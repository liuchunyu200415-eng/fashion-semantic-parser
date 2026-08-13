"""Tests for supervised dense patch localization reporting."""

from scripts.evaluate_dense_patch_localization import (
    _score_cases,
    _summarize,
    _warm_mean,
)


def test_dense_case_scoring_retains_failed_queries() -> None:
    """Mask and Box denominators must include every full-query miss."""
    cases = [
        {
            "mask_iou": 0.8,
            "box_iou": 0.9,
            "dimensions": ["basic", "spatial"],
            "language": "zh",
            "target_label": "pocket",
        },
        {
            "mask_iou": 0.1,
            "box_iou": 0.2,
            "dimensions": ["basic"],
            "language": "en",
            "target_label": "zipper",
        },
    ]

    metrics = _score_cases(cases)

    assert metrics["query_count"] == 2
    assert metrics["mask_recall50_count"] == 1
    assert metrics["mask_recall50"] == 0.5
    assert metrics["mean_mask_iou"] == 0.45
    assert metrics["box_recall50_count"] == 1


def test_dense_summary_reports_overlapping_dimensions_and_groups() -> None:
    """Spatial queries should remain in both basic and spatial diagnostics."""
    cases = [
        {
            "mask_iou": 0.8,
            "box_iou": 0.9,
            "dimensions": ["basic", "spatial"],
            "language": "zh",
            "target_label": "pocket",
            "oracle_pixel_area_topk_mask_iou": 0.6,
            "oracle_support_topk_mask_iou": 0.4,
        }
    ]
    image_rows = [{"total_image_seconds": 0.2}]

    summary = _summarize(cases, image_rows)

    assert summary["by_dimension"]["basic"]["query_count"] == 1
    assert summary["by_dimension"]["spatial"]["query_count"] == 1
    assert summary["by_language"]["zh"]["mask_recall50"] == 1.0
    assert summary["by_target_label"]["pocket"]["mask_recall50"] == 1.0
    assert summary["oracle_pixel_area_topk_mask_recall50"] == 1.0
    assert summary["oracle_support_topk_mask_recall50"] == 0.0


def test_warm_latency_excludes_first_model_load_image() -> None:
    """The cold first image cannot be averaged into warm image latency."""
    assert _warm_mean([{"total_image_seconds": 1.0}]) is None
    assert (
        _warm_mean(
            [
                {"total_image_seconds": 1.0},
                {"total_image_seconds": 0.1},
                {"total_image_seconds": 0.3},
            ]
        )
        == 0.2
    )
