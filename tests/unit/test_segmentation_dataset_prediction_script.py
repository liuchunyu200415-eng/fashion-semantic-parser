"""Tests for dataset-level segmentation prediction helpers."""

from pathlib import Path

from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
)
from scripts.predict_segmentation_dataset import (
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
