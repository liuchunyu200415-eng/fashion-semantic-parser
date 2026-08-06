"""Tests for offline PRD 3.1.2 operating-point scans."""

from pathlib import Path

import pytest

from scripts.scan_localization_operating_points import (
    _build_scan_result,
    _normalize_thresholds,
    _normalize_top_k,
    _summarize_operating_point,
)


def test_scan_grid_is_sorted_deduplicated_and_supports_unlimited_top_k() -> None:
    """Repeated CLI values should produce one deterministic scan grid."""
    assert _normalize_thresholds([0.2, 0.0, 0.2]) == [0.0, 0.2]
    assert _normalize_top_k([5, 0, 1, 5]) == [1, 5, None]


@pytest.mark.parametrize("values", [[-0.1], [1.1], []])
def test_invalid_threshold_grid_is_rejected(values: list[float]) -> None:
    """Offline score thresholds remain probabilities."""
    with pytest.raises(ValueError):
        _normalize_thresholds(values)


@pytest.mark.parametrize("values", [[-1], []])
def test_invalid_top_k_grid_is_rejected(values: list[int]) -> None:
    """Only positive limits and zero-as-unlimited are accepted."""
    with pytest.raises(ValueError):
        _normalize_top_k(values)


def test_operating_point_summary_keeps_macro_and_per_category_recall() -> None:
    """Macro recall must stay auditable down to every exact source class."""
    evaluation = {
        "categories": ["collar", "pocket"],
        "score_threshold": 0.1,
        "top_k": 3,
        "candidate_count_after_filter": 9,
        "segm_coco": {"AP50": 45.0},
        "segm_direct_iou": {
            "Recall50-collar": 80.0,
            "Recall50-pocket": 40.0,
            "Precision50": 50.0,
            "Recall50": 60.0,
            "F1_50": 54.55,
            "MatchedMeanIoU": 75.0,
            "AllGTMeanIoU": 45.0,
            "AllGTIoU85Rate": 20.0,
        },
    }

    row = _summarize_operating_point(evaluation)

    assert row["macro_recall50"] == 60.0
    assert row["per_category_recall50"] == {
        "collar": 80.0,
        "pocket": 40.0,
    }


def test_scan_result_selects_recall_and_f1_independently(tmp_path: Path) -> None:
    """High-recall diagnosis and deployable F1 can choose different settings."""
    rows = [
        {"macro_recall50": 93.0, "f1_50": 30.0},
        {"macro_recall50": 80.0, "f1_50": 70.0},
    ]

    result = _build_scan_result(
        rows=rows,
        validation_path=tmp_path / "validation.json",
        prediction_path=tmp_path / "predictions.json",
        summary_path=tmp_path / "missing_summary.json",
        category_names=["collar"],
    )

    assert result["any_operating_point_passed"] is True
    assert result["best_macro_recall"] is rows[0]
    assert result["best_micro_f1"] is rows[1]
