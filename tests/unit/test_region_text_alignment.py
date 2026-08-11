"""Tests for multi-positive DINOv2/BGE-M3 alignment."""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from fashion_semantic_parser.service.region_text_alignment import (
    RegionTextAlignmentSettings,
    build_positive_region_mask,
    build_text_projection,
    evaluate_image_candidate_retrieval,
    extract_unique_region_features,
    load_region_text_alignment_settings,
    load_text_projection_checkpoint,
    multi_positive_contrastive_loss,
    positive_top1_accuracy,
)


def test_positive_mask_preserves_multi_target_and_shared_regions() -> None:
    """Broad and modified queries can share positives without false negatives."""
    mask = build_positive_region_mask(
        [(11, 12), (12,), (13,)],
        [11, 12, 13],
    )

    assert mask.tolist() == [
        [True, True, False],
        [False, True, False],
        [False, False, True],
    ]


def test_positive_mask_rejects_missing_candidate() -> None:
    """Every source target must be present in the candidate region bank."""
    with pytest.raises(ValueError, match="missing candidate"):
        build_positive_region_mask([(11, 12)], [11])


def test_alignment_config_matches_encoder_dimensions() -> None:
    """The committed head must bridge the two pinned encoder geometries."""
    settings = load_region_text_alignment_settings()

    assert settings.text_dimension == 1024
    assert settings.region_dimension == 384
    assert settings.hidden_dimension == 512
    assert settings.temperature == 0.07


def test_alignment_hidden_dimension_cannot_undercut_region_dimension() -> None:
    """The smoke head cannot silently introduce an unintended bottleneck."""
    with pytest.raises(ValidationError, match="hidden_dimension"):
        RegionTextAlignmentSettings(hidden_dimension=128)


def test_multi_positive_loss_rewards_correct_alignment() -> None:
    """Aligned pairs must score better than a deliberately swapped pairing."""
    torch = pytest.importorskip("torch")
    text = torch.eye(3)
    regions = torch.eye(3)
    positives = torch.eye(3, dtype=torch.bool)

    aligned_loss, aligned_logits = multi_positive_contrastive_loss(
        text,
        regions,
        positives,
        temperature=0.1,
    )
    swapped_loss, _ = multi_positive_contrastive_loss(
        text,
        regions[[1, 2, 0]],
        positives,
        temperature=0.1,
    )

    assert aligned_loss.item() < swapped_loss.item()
    assert positive_top1_accuracy(aligned_logits, positives) == 1.0


def test_retrieval_reports_competitive_and_grouped_metrics_separately() -> None:
    """Trivial single-candidate cases cannot inflate competitive accuracy."""
    summary, cases = evaluate_image_candidate_retrieval(
        query_ids=["q-spatial", "q-basic", "q-relation"],
        projected_text_features=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        query_image_ids=[1, 1, 2],
        query_target_ids=[(11,), (12,), (21,)],
        query_dimensions=[
            ("basic", "spatial"),
            ("basic",),
            ("basic", "relation"),
        ],
        query_languages=["zh", "en", "en"],
        region_annotation_ids=[11, 12, 21],
        region_image_ids=[1, 1, 2],
        region_features=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
    )

    assert summary["query_count"] == 3
    assert summary["top1_correct_count"] == 2
    assert summary["top1_accuracy"] == pytest.approx(2 / 3)
    assert summary["competitive_query_count"] == 2
    assert summary["competitive_top1_accuracy"] == 0.5
    assert summary["mean_reciprocal_rank"] == pytest.approx(5 / 6)
    assert summary["by_dimension"]["spatial"]["top1_accuracy"] == 1.0
    assert summary["by_language"]["zh"]["query_count"] == 1
    assert cases[2]["competitive"] is False


def test_unique_region_extraction_reuses_duplicate_source_mask() -> None:
    """Alternative queries for one annotation must not repeat DINOv2 inference."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.ones((1, 4, 4), dtype=np.uint8)
    items = [
        SimpleNamespace(
            sample=SimpleNamespace(source_image_id=7),
            image_rgb=image,
            source_annotation_ids=(101,),
            target_masks=mask,
        ),
        SimpleNamespace(
            sample=SimpleNamespace(source_image_id=7),
            image_rgb=image.copy(),
            source_annotation_ids=(101,),
            target_masks=mask.copy(),
        ),
    ]

    class FakeEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, image_rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
            self.calls += 1
            assert image_rgb.shape == (4, 4, 3)
            return np.ones((len(masks), 3), dtype=np.float32)

    encoder = FakeEncoder()
    features = extract_unique_region_features(items, encoder)

    assert encoder.calls == 1
    assert list(features) == [101]
    assert features[101].shape == (3,)


def test_checkpoint_loader_rejects_unfrozen_metadata(tmp_path, monkeypatch) -> None:
    """An evaluator cannot silently accept a different training contract."""
    torch = pytest.importorskip("torch")
    settings = RegionTextAlignmentSettings(
        text_dimension=4,
        hidden_dimension=3,
        region_dimension=3,
    )
    checkpoint = tmp_path / "alignment.pt"
    torch.save(
        {
            "schema_version": 1,
            "alignment_settings": settings.model_dump(mode="json"),
            "state_dict": build_text_projection(settings).state_dict(),
            "base_encoders_frozen": False,
            "dinov2_model": "dinov2_vits14",
            "text_model": "BAAI/bge-m3",
        },
        checkpoint,
    )
    monkeypatch.setattr(
        "fashion_semantic_parser.service.region_text_alignment.resolve_project_path",
        lambda _: checkpoint,
    )

    with pytest.raises(ValueError, match="frozen"):
        load_text_projection_checkpoint(checkpoint, device="cpu")
