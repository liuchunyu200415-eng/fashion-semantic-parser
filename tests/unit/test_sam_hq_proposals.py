"""Tests for class-agnostic SAM-HQ proposal generation."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from fashion_semantic_parser.common.exceptions import ModelNotReadyError
from fashion_semantic_parser.service.sam_hq_proposals import (
    SAMHQAutomaticProposalGenerator,
    SAMHQMaskProposal,
    SAMHQProposalSettings,
    best_proposal_mask_iou,
    load_sam_hq_proposal_settings,
    validate_local_sam_hq_assets,
)


class _FakeGenerator:
    def __init__(self, proposals: list[object]) -> None:
        self.proposals = proposals

    def generate(self, image: np.ndarray) -> list[object]:
        assert image.shape == (20, 30, 3)
        return self.proposals


def _record(
    box: tuple[int, int, int, int],
    *,
    predicted_iou: float,
    stability_score: float,
) -> dict[str, object]:
    x_min, y_min, x_max, y_max = box
    mask = np.zeros((20, 30), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    return {
        "segmentation": mask,
        "predicted_iou": predicted_iou,
        "stability_score": stability_score,
    }


def test_project_config_uses_high_recall_official_sam_hq_path() -> None:
    settings = load_sam_hq_proposal_settings()

    assert settings.sam_hq_model_type == "vit_b"
    assert settings.sam_hq_repo_commit == "e696978d60352dc9a26b12631cd91781502c6546"
    assert settings.points_per_side == 32
    assert settings.crop_n_layers == 1
    assert settings.max_regions == 200
    assert settings.hq_token_only is False

    hq_only = load_sam_hq_proposal_settings(
        "configs/localization_sam_hq_proposals_hq_only.yaml"
    )
    assert hq_only.hq_token_only is True
    assert hq_only.sam_hq_weights == settings.sam_hq_weights


def test_proposals_are_validated_ranked_and_capped() -> None:
    generator = SAMHQAutomaticProposalGenerator(
        SAMHQProposalSettings(
            device="cpu",
            precision="fp32",
            min_mask_region_area=2,
            max_regions=2,
        ),
        generator=_FakeGenerator(
            [
                _record((1, 2, 4, 6), predicted_iou=0.7, stability_score=0.9),
                _record((10, 4, 15, 8), predicted_iou=0.9, stability_score=0.8),
                _record((20, 10, 22, 12), predicted_iou=0.8, stability_score=0.9),
            ]
        ),
    )

    proposals = generator.generate(np.zeros((20, 30, 3), dtype=np.uint8))

    assert [proposal.predicted_iou for proposal in proposals] == [0.9, 0.8]
    assert proposals[0].box == (10.0, 4.0, 15.0, 8.0)
    assert proposals[0].area == 20


def test_small_and_empty_proposals_are_filtered() -> None:
    generator = SAMHQAutomaticProposalGenerator(
        SAMHQProposalSettings(
            device="cpu",
            precision="fp32",
            min_mask_region_area=5,
        ),
        generator=_FakeGenerator(
            [_record((1, 1, 2, 2), predicted_iou=0.9, stability_score=0.9)]
        ),
    )

    assert generator.generate(np.zeros((20, 30, 3), dtype=np.uint8)) == []


def test_wrong_mask_shape_is_rejected() -> None:
    generator = SAMHQAutomaticProposalGenerator(
        SAMHQProposalSettings(device="cpu", precision="fp32"),
        generator=_FakeGenerator(
            [
                {
                    "segmentation": np.ones((2, 2), dtype=bool),
                    "predicted_iou": 0.9,
                    "stability_score": 0.9,
                }
            ]
        ),
    )

    with pytest.raises(ModelNotReadyError, match="dimensions"):
        generator.generate(np.zeros((20, 30, 3), dtype=np.uint8))


def test_unbounded_finite_predicted_iou_is_retained() -> None:
    generator = SAMHQAutomaticProposalGenerator(
        SAMHQProposalSettings(device="cpu", precision="fp32"),
        generator=_FakeGenerator(
            [_record((1, 1, 6, 6), predicted_iou=1.1, stability_score=0.9)]
        ),
    )

    proposals = generator.generate(np.zeros((20, 30, 3), dtype=np.uint8))

    assert proposals[0].predicted_iou == 1.1


@pytest.mark.parametrize(
    ("predicted_iou", "stability_score", "message"),
    [
        (float("nan"), 0.9, "predicted_iou must be finite"),
        (0.9, 1.1, "stability_score must be in"),
    ],
)
def test_invalid_quality_score_is_rejected(
    predicted_iou: float,
    stability_score: float,
    message: str,
) -> None:
    generator = SAMHQAutomaticProposalGenerator(
        SAMHQProposalSettings(device="cpu", precision="fp32"),
        generator=_FakeGenerator(
            [
                _record(
                    (1, 1, 6, 6),
                    predicted_iou=predicted_iou,
                    stability_score=stability_score,
                )
            ]
        ),
    )

    with pytest.raises(ModelNotReadyError, match=message):
        generator.generate(np.zeros((20, 30, 3), dtype=np.uint8))


def test_best_proposal_iou_retains_misses_in_recall_ceiling() -> None:
    target = np.zeros((4, 4), dtype=bool)
    target[:2, :2] = True
    miss = np.zeros((4, 4), dtype=bool)
    miss[2:, 2:] = True
    exact = target.copy()
    proposals = [
        SAMHQMaskProposal((2, 2, 4, 4), miss, 4, 0.9, 0.9),
        SAMHQMaskProposal((0, 0, 2, 2), exact, 4, 0.8, 0.8),
    ]

    best_iou, best_index = best_proposal_mask_iou(target, proposals)

    assert best_iou == 1.0
    assert best_index == 1
    assert best_proposal_mask_iou(target, []) == (0.0, None)


def test_local_assets_require_pinned_source_and_checkpoint(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "sam-hq"
    head_path = repo_path / ".git" / "HEAD"
    head_path.parent.mkdir(parents=True)
    commit = "a" * 40
    head_path.write_text(commit + "\n", encoding="utf-8")
    weights_path = tmp_path / "weights.pth"
    weights_path.write_bytes(b"official-test-weights")
    checksum = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    generator = SAMHQAutomaticProposalGenerator(
        SAMHQProposalSettings(
            device="cpu",
            precision="fp32",
            sam_hq_repo_commit=commit,
            sam_hq_weights_sha256=checksum,
        )
    )

    validate_local_sam_hq_assets(generator.settings, repo_path, weights_path)

    weights_path.write_bytes(b"drifted")
    with pytest.raises(ModelNotReadyError, match="checksum mismatch"):
        validate_local_sam_hq_assets(generator.settings, repo_path, weights_path)
