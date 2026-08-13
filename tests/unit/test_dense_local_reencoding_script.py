"""Tests for coarse-to-fine local re-encoding evaluation metrics."""

import importlib.util
from pathlib import Path

import pytest


def _load_script():
    """Load the evaluation script as a test module."""
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluate_dense_local_reencoding.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_dense_local_reencoding",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_retains_misses_and_reports_area_ratio() -> None:
    """Each fixed branch must retain misses and expose its eligible count."""
    module = _load_script()
    cases = [
        {
            "target_area": 10,
            "coarse_mask_iou": 0.60,
            "coarse_box_iou": 0.80,
            "coarse_predicted_area": 20,
            "local_only_mask_iou": 0.40,
            "local_only_box_iou": 0.30,
            "local_only_predicted_area": 10,
            "coarse_local_max_mask_iou": 0.75,
            "coarse_local_max_box_iou": 0.90,
            "coarse_local_max_predicted_area": 30,
        },
        {
            "target_area": 10,
            "coarse_mask_iou": 0.00,
            "coarse_box_iou": 0.00,
            "coarse_predicted_area": 0,
            "local_only_mask_iou": 0.50,
            "local_only_box_iou": 0.60,
            "local_only_predicted_area": 20,
            "coarse_local_max_mask_iou": 0.50,
            "coarse_local_max_box_iou": 0.60,
            "coarse_local_max_predicted_area": 20,
        },
    ]

    summary = module._summarize(cases)

    assert summary["query_count"] == 2
    assert summary["coarse"]["mask_recall50_count"] == 1
    assert summary["coarse"]["mask_recall50"] == 0.5
    assert summary["local_only"]["mask_recall50"] == 0.5
    assert summary["coarse_local_max"]["mask_recall75_count"] == 1
    assert summary["coarse"]["median_predicted_to_target_area_ratio"] == 1.0


def test_summary_rejects_empty_case_set() -> None:
    """Undefined empty-set accuracy must not be serialized as a valid score."""
    module = _load_script()

    with pytest.raises(ValueError, match="at least one"):
        module._summarize([])
