"""Tests for the DINOv2 dense localization smoke metrics."""

import argparse

import pytest

from scripts.smoke_dinov2_dense_localization import (
    DenseRunMetadata,
    _parse_quantiles,
    _quantile_metrics,
    _summarize,
)


def test_score_quantiles_require_unique_ascending_values() -> None:
    """Threshold scans must remain ordered and reproducible."""
    assert _parse_quantiles("0.9,0.95,0.99") == (0.9, 0.95, 0.99)

    with pytest.raises(argparse.ArgumentTypeError, match="unique, ascending"):
        _parse_quantiles("0.95,0.9")
    with pytest.raises(argparse.ArgumentTypeError, match=r"in \(0, 1\)"):
        _parse_quantiles("0,0.9")


def test_quantile_metrics_keep_counts_and_failed_queries() -> None:
    """Every query must remain in Mask and Box accuracy denominators."""
    cases = [
        {
            "mask_iou_by_quantile": {"0.900": 0.80},
            "box_iou_by_quantile": {"0.900": 0.90},
        },
        {
            "mask_iou_by_quantile": {"0.900": 0.20},
            "box_iou_by_quantile": {"0.900": 0.10},
        },
    ]

    metrics = _quantile_metrics(cases, 0.9)

    assert metrics["query_count"] == 2
    assert metrics["mask_recall50_count"] == 1
    assert metrics["mask_recall50"] == 0.5
    assert metrics["mean_mask_iou"] == 0.5
    assert metrics["box_recall50_count"] == 1


def test_summary_marks_dense_smoke_as_non_acceptance_evidence() -> None:
    """A development threshold scan cannot claim PRD acceptance."""
    cases = [
        {
            "mask_iou_by_quantile": {"0.900": 1.0},
            "box_iou_by_quantile": {"0.900": 1.0},
        }
    ]
    image_rows = [
        {
            "dinov2_encode_seconds": 0.01,
            "dense_scoring_seconds": 0.001,
        }
    ]
    metadata = DenseRunMetadata(
        split="validation",
        image_offset=0,
        checkpoint="checkpoint.pt",
        text_seconds=0.02,
        projection_seconds=0.003,
    )

    summary = _summarize(cases, image_rows, (0.9,), metadata)

    assert summary["full_image_candidate_coverage"] is True
    assert summary["uses_oracle_candidates"] is False
    assert summary["independent_manual_test_set"] is False
    assert summary["selected_score_quantile"] is None
    assert summary["prd_accuracy_92_passed"] is None
    assert summary["prd_localization_30ms_passed"] is None
