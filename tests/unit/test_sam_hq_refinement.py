"""Tests for official SAM-HQ Box-prompt refinement."""

from types import SimpleNamespace

import numpy as np
import pytest

from fashion_semantic_parser.common.exceptions import ModelNotReadyError
from fashion_semantic_parser.service.sam_hq_proposals import SAMHQProposalSettings
from fashion_semantic_parser.service.sam_hq_refinement import SAMHQBoxPromptRefiner


class _FakeTorch:
    float32 = np.float32
    int64 = np.int64

    @staticmethod
    def as_tensor(value: object, **_: object) -> np.ndarray:
        """Convert test prompt boxes to an array."""
        return np.asarray(value, dtype=np.float32)


class _FakePredictor:
    device = "cpu"

    def __init__(self) -> None:
        """Create an identity transform and deterministic Mask outputs."""
        self.transform = SimpleNamespace(
            apply_boxes_torch=lambda boxes, _shape: boxes,
            apply_coords_torch=lambda points, _shape: points,
        )
        self.image_shape: tuple[int, ...] | None = None
        self.last_point_coords: object = None
        self.last_point_labels: object = None

    def set_image(self, image: np.ndarray) -> None:
        """Retain the image shape for output construction."""
        self.image_shape = image.shape

    def predict_torch(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, None]:
        """Return one rectangular Mask per transformed Box prompt."""
        boxes = np.asarray(kwargs["boxes"], dtype=np.int32)
        self.last_point_coords = kwargs["point_coords"]
        self.last_point_labels = kwargs["point_labels"]
        assert self.image_shape is not None
        candidate_count = 3 if kwargs["multimask_output"] else 1
        masks = np.zeros(
            (len(boxes), candidate_count, *self.image_shape[:2]),
            dtype=bool,
        )
        for index, (x_min, y_min, x_max, y_max) in enumerate(boxes):
            masks[index, :, y_min:y_max, x_min:x_max] = True
        return masks, np.full((len(boxes), candidate_count), 0.9), None


def _refiner() -> SAMHQBoxPromptRefiner:
    """Return a CPU refiner backed by deterministic test doubles."""
    return SAMHQBoxPromptRefiner(
        SAMHQProposalSettings(device="cpu", precision="fp32"),
        predictor=_FakePredictor(),
        torch_module=_FakeTorch(),
    )


def test_box_prompt_refinement_preserves_order_and_mask_geometry() -> None:
    """Each valid Box should produce its corresponding binary Mask."""
    results = _refiner().refine(
        np.zeros((20, 30, 3), dtype=np.uint8),
        [(1.0, 2.0, 5.0, 7.0), (10.0, 4.0, 15.0, 8.0)],
    )

    assert [result.prompt_box for result in results] == [
        (1.0, 2.0, 5.0, 7.0),
        (10.0, 4.0, 15.0, 8.0),
    ]
    assert results[0].mask_box == (1.0, 2.0, 5.0, 7.0)
    assert results[0].mask.sum() == 20
    assert results[0].mask_quality == 0.9


def test_box_prompt_refinement_rejects_invalid_or_empty_boxes() -> None:
    """Invalid prompts fail while an empty prompt batch returns immediately."""
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    assert _refiner().refine(image, []) == []
    with pytest.raises(ValueError, match="positive in-image area"):
        _refiner().refine(image, [(5.0, 5.0, 2.0, 2.0)])


def test_multimask_refinement_preserves_candidate_groups() -> None:
    """Ambiguity-aware inference should retain every candidate per prompt."""
    groups = _refiner().refine_candidates(
        np.zeros((20, 30, 3), dtype=np.uint8),
        [(1.0, 2.0, 5.0, 7.0), (10.0, 4.0, 15.0, 8.0)],
        multimask_output=True,
    )

    assert [len(group) for group in groups] == [3, 3]
    assert all(candidate.mask.sum() > 0 for group in groups for candidate in group)


def test_positive_points_are_transformed_with_matching_labels() -> None:
    """One foreground point per Box should reach the external predictor."""
    predictor = _FakePredictor()
    refiner = SAMHQBoxPromptRefiner(
        SAMHQProposalSettings(device="cpu", precision="fp32"),
        predictor=predictor,
        torch_module=_FakeTorch(),
    )

    groups = refiner.refine_candidates(
        np.zeros((20, 30, 3), dtype=np.uint8),
        [(1.0, 2.0, 5.0, 7.0)],
        multimask_output=True,
        positive_points=[(3.0, 4.0)],
    )

    assert len(groups[0]) == 3
    assert np.array_equal(predictor.last_point_coords, [[[3.0, 4.0]]])
    assert np.array_equal(predictor.last_point_labels, [[1]])
    with pytest.raises(ValueError, match="count must match"):
        refiner.refine_candidates(
            np.zeros((20, 30, 3), dtype=np.uint8),
            [(1.0, 2.0, 5.0, 7.0)],
            multimask_output=True,
            positive_points=[],
        )


def test_box_prompt_refinement_rejects_invalid_output_shape() -> None:
    """External runtime shape drift must fail before evaluation."""
    predictor = _FakePredictor()
    predictor.predict_torch = lambda **_kwargs: (  # type: ignore[method-assign]
        np.zeros((1, 1, 2, 2), dtype=bool),
        np.ones((1, 1), dtype=float),
        None,
    )
    refiner = SAMHQBoxPromptRefiner(
        SAMHQProposalSettings(device="cpu", precision="fp32"),
        predictor=predictor,
        torch_module=_FakeTorch(),
    )

    with pytest.raises(ModelNotReadyError, match="Mask dimensions"):
        refiner.refine(
            np.zeros((20, 30, 3), dtype=np.uint8),
            [(1.0, 1.0, 5.0, 5.0)],
        )
