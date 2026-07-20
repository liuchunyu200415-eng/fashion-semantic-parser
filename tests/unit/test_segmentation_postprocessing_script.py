"""Tests for saved-prediction category-conflict post-processing."""

from typing import Any

from scripts.postprocess_segmentation_predictions import _postprocess_predictions


def _prediction(
    image_id: int,
    category_id: int,
    score: float,
    bbox: list[float],
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "category_id": category_id,
        "score": score,
        "bbox": bbox,
        "segmentation": {"size": [10, 10], "counts": "test"},
    }


def test_postprocessing_filters_score_and_composite_dress() -> None:
    """The output should retain masks while removing only diagnosed dresses."""
    predictions = [
        _prediction(1, 5, 0.82, [0.0, 0.0, 100.0, 200.0]),
        _prediction(1, 1, 0.94, [0.0, 0.0, 100.0, 90.0]),
        _prediction(1, 3, 0.86, [0.0, 90.0, 100.0, 110.0]),
        _prediction(2, 5, 0.95, [0.0, 0.0, 100.0, 200.0]),
        _prediction(2, 1, 0.79, [0.0, 0.0, 100.0, 90.0]),
    ]

    from fashion_semantic_parser.service.segmentation_postprocessing import (
        find_composite_dress_conflicts,
    )

    output, report = _postprocess_predictions(
        predictions,
        score_threshold=0.8,
        min_union_iou=0.8,
        min_component_coverage=0.8,
        score_margin=0.0,
        conflict_finder=find_composite_dress_conflicts,
    )

    assert [(row["image_id"], row["category_id"]) for row in output] == [
        (1, 1),
        (1, 3),
        (2, 5),
    ]
    assert output[0]["segmentation"] == predictions[1]["segmentation"]
    assert report["prediction_count_before_score_filter"] == 5
    assert report["prediction_count_after_score_filter"] == 4
    assert report["suppressed_dress_count"] == 1
    assert report["affected_image_count"] == 1
    assert report["prediction_count_after_conflict_filter"] == 3
