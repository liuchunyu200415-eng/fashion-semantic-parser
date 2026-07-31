"""Tests for saved-prediction mask IoU evaluation helpers."""

from pathlib import Path
from typing import Any

import pytest

from scripts.evaluate_segmentation_predictions import (
    _category_thresholds_by_id,
    _coco_ap_summary,
    _coco_class_names,
    _filter_predictions,
    _parse_category_thresholds,
    _resolve_path,
)


class _FakeCOCO:
    """Minimal COCO API stand-in for category-name ordering."""

    def loadCats(self, category_ids: list[int]) -> list[dict[str, Any]]:
        """Return deliberately reversed records to test id-based ordering."""
        assert category_ids == [1, 2]
        return [
            {"id": 2, "name": "pants"},
            {"id": 1, "name": "top"},
        ]

    @staticmethod
    def getCatIds() -> list[int]:
        """Return the fake taxonomy IDs."""
        return [1, 2]


def test_prediction_filter_keeps_threshold_boundary() -> None:
    """Saved predictions at the selected operating threshold should remain."""
    predictions = [
        {"score": 0.09, "image_id": 1},
        {"score": 0.10, "image_id": 2},
        {"score": 0.80, "image_id": 3},
    ]

    filtered = _filter_predictions(predictions, score_threshold=0.10)

    assert [prediction["image_id"] for prediction in filtered] == [2, 3]


def test_prediction_filter_rejects_invalid_threshold() -> None:
    """The deployment score threshold must use a zero-to-one value."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        _filter_predictions([], score_threshold=1.1)


def test_prediction_filter_supports_category_thresholds() -> None:
    """Small classes may use lower thresholds without changing other classes."""
    predictions = [
        {"score": 0.35, "category_id": 1},
        {"score": 0.35, "category_id": 2},
        {"score": 0.65, "category_id": 2},
    ]

    filtered = _filter_predictions(
        predictions,
        score_threshold=0.6,
        category_score_thresholds={1: 0.3},
    )

    assert filtered == [predictions[0], predictions[2]]


def test_category_threshold_arguments_resolve_to_coco_ids() -> None:
    """CLI labels should map to the category IDs stored in predictions."""
    thresholds = _parse_category_thresholds(["top=0.6", "pants=0.4"])

    assert _category_thresholds_by_id(_FakeCOCO(), thresholds) == {
        1: 0.6,
        2: 0.4,
    }


def test_category_threshold_arguments_reject_unknown_categories() -> None:
    """Typos must fail instead of silently using the global threshold."""
    with pytest.raises(ValueError, match="Unknown COCO categories"):
        _category_thresholds_by_id(_FakeCOCO(), {"shoe": 0.3})


def test_coco_class_names_follow_evaluator_category_order() -> None:
    """Per-category IoU labels should align with COCOeval matrix columns."""
    assert _coco_class_names(_FakeCOCO(), [1, 2]) == ["top", "pants"]


def test_coco_ap_summary_converts_fractions_to_percentages() -> None:
    """Saved-prediction AP should use the same percentage scale as other reports."""
    metrics = _coco_ap_summary([0.4, 0.6, 0.3, -1.0, 0.2, 0.5])

    assert metrics["AP"] == pytest.approx(40.0)
    assert metrics["AP50"] == pytest.approx(60.0)
    assert metrics["AP75"] == pytest.approx(30.0)
    assert metrics["APs"] != metrics["APs"]
    assert metrics["APm"] == pytest.approx(20.0)
    assert metrics["APl"] == pytest.approx(50.0)


def test_output_path_allows_external_evaluation_directory(tmp_path: Path) -> None:
    """Saved-prediction metrics may be written beside external run artifacts."""
    output_path = tmp_path / "metrics.json"

    assert _resolve_path(output_path, pytest.fail) == output_path
