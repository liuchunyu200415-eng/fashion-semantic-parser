"""Tests for the optional isolated SAM 3 referring benchmark runner."""

from argparse import Namespace

import numpy as np
import pytest

from scripts.predict_referring_sam3 import (
    _latency_summary,
    _normalize_masks,
    _result_to_prediction,
    _validate_args,
)


def test_result_to_prediction_preserves_prompt_masks_boxes_and_scores() -> None:
    """SAM 3 results must retain the full prompt and evaluator-compatible masks."""
    mask = np.zeros((1, 1, 10, 12), dtype=bool)
    mask[0, 0, 2:7, 3:9] = True
    prediction = _result_to_prediction(
        {
            "masks": mask,
            "boxes": np.asarray([[3.0, 2.0, 9.0, 7.0]]),
            "scores": np.asarray([0.8]),
        },
        image_path="data/image.jpg",
        query="衣服右边的口袋",
        grounding_prompt="the pocket on the right side of the garment",
        image_size=(12, 10),
        max_regions=10,
    )

    assert prediction.query == "衣服右边的口袋"
    assert len(prediction.regions) == 1
    region = prediction.regions[0]
    assert region.region_label == "sam3_text"
    assert region.matched_text == "the pocket on the right side of the garment"
    assert region.confidence == pytest.approx(0.8)
    assert region.box.model_dump() == {
        "x_min": 3.0,
        "y_min": 2.0,
        "x_max": 9.0,
        "y_max": 7.0,
    }
    assert region.mask


def test_result_to_prediction_sorts_limits_and_retains_empty_results() -> None:
    """Candidate limits must be score-based and an empty result must remain a miss."""
    masks = np.zeros((2, 6, 6), dtype=bool)
    masks[:, 1:5, 1:5] = True
    prediction = _result_to_prediction(
        {
            "masks": masks,
            "boxes": [[1, 1, 5, 5], [1, 1, 5, 5]],
            "scores": [0.2, 0.9],
        },
        image_path="data/image.jpg",
        query="zipper",
        grounding_prompt="the silver zipper",
        image_size=(6, 6),
        max_regions=1,
    )
    assert [region.confidence for region in prediction.regions] == [0.9]

    empty = _result_to_prediction(
        {"masks": [], "boxes": [], "scores": []},
        image_path="data/image.jpg",
        query="zipper",
        grounding_prompt="zipper",
        image_size=(6, 6),
        max_regions=1,
    )
    assert empty.regions == []


def test_sam3_argument_validation_and_mask_shape() -> None:
    """Reject invalid thresholds and unexpected post-processing shapes early."""
    valid = Namespace(
        score_threshold=0.5,
        mask_threshold=0.5,
        max_regions=10,
        progress_every=1,
    )
    _validate_args(valid)
    assert _normalize_masks(np.ones((1, 1, 2, 3))).shape == (1, 2, 3)

    invalid = Namespace(
        score_threshold=1.1,
        mask_threshold=0.5,
        max_regions=10,
        progress_every=1,
    )
    with pytest.raises(ValueError, match="score-threshold"):
        _validate_args(invalid)
    with pytest.raises(ValueError, match="shape"):
        _normalize_masks(np.ones((1, 2, 2, 3)))


def test_sam3_latency_excludes_model_loading() -> None:
    """The runner reports request latency separately from model initialization."""
    assert _latency_summary([]) == {"count": 0, "mean": None, "p95": None}
    summary = _latency_summary([0.2, 0.4])
    assert summary["count"] == 2
    assert summary["mean"] == pytest.approx(0.3)
    assert summary["p95"] == pytest.approx(0.39)
