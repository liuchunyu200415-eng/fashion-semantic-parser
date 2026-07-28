"""Tests for dataset-level segmentation prediction helpers."""

from pathlib import Path

import pytest

from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
)
from scripts.predict_segmentation_dataset import (
    _build_settings_overrides,
    _summary_path,
    prediction_to_coco_results,
)


def test_prediction_to_coco_results_preserves_masks_boxes_and_scores() -> None:
    """Typed API output should become valid flat COCO result records."""
    prediction = SegmentationPrediction(
        image_path="data/example.jpg",
        instances=[
            SegmentationInstance(
                category_id=7,
                category_label="bag",
                confidence=0.81,
                box=SegmentationBoundingBox(
                    x_min=10.0,
                    y_min=20.0,
                    x_max=40.0,
                    y_max=70.0,
                ),
                mask=[[10.0, 20.0, 40.0, 20.0, 40.0, 70.0]],
            )
        ],
    )

    results = prediction_to_coco_results(prediction, image_id=123)

    assert results == [
        {
            "image_id": 123,
            "category_id": 7,
            "bbox": [10.0, 20.0, 30.0, 50.0],
            "score": 0.81,
            "segmentation": [[10.0, 20.0, 40.0, 20.0, 40.0, 70.0]],
        }
    ]


def test_prediction_to_coco_results_skips_empty_masks() -> None:
    """COCO segmentation evaluation cannot use instances without polygons."""
    prediction = SegmentationPrediction(
        image_path="data/example.jpg",
        instances=[
            SegmentationInstance(
                category_id=1,
                category_label="top",
                confidence=0.9,
                box=SegmentationBoundingBox(
                    x_min=1.0,
                    y_min=1.0,
                    x_max=10.0,
                    y_max=10.0,
                ),
                mask=[],
            )
        ],
    )

    assert prediction_to_coco_results(prediction, image_id=1) == []


def test_dataset_prediction_summary_uses_output_stem() -> None:
    """Run metadata should sit beside the potentially large result list."""
    output = Path("outputs/auto_roi_predictions.json")

    assert _summary_path(output) == Path("outputs/auto_roi_predictions_summary.json")


def test_dataset_prediction_builds_auto_roi_margin_override() -> None:
    """Automatic ROI experiments should override context without new configs."""
    assert _build_settings_overrides(
        roi_mode="auto",
        subject_roi_margin=0.25,
    ) == {"subject_roi_margin": 0.25}


@pytest.mark.parametrize(
    ("roi_mode", "margin"),
    [
        ("full", 0.25),
        ("auto", -0.1),
        ("auto", 1.1),
    ],
)
def test_dataset_prediction_rejects_invalid_roi_margin(
    roi_mode: str,
    margin: float,
) -> None:
    """Margin sweeps should fail fast for misleading or invalid settings."""
    with pytest.raises(ValueError, match="subject-roi-margin"):
        _build_settings_overrides(
            roi_mode=roi_mode,
            subject_roi_margin=margin,
        )
