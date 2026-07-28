"""Tests for single-image segmentation visualization helpers."""

import numpy as np

from scripts.visualize_segmentation_prediction import draw_prediction


def test_roi_visualization_keeps_predictions_in_expanded_context() -> None:
    """ROI context may contain shoes or bags that must remain visible."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    prediction = {
        "instances": [
            {
                "category_label": "bag",
                "confidence": 0.9,
                "box": {
                    "x_min": 70.0,
                    "y_min": 70.0,
                    "x_max": 85.0,
                    "y_max": 85.0,
                },
                "mask": [[70.0, 70.0, 85.0, 70.0, 85.0, 85.0, 70.0, 85.0]],
            }
        ]
    }

    result = draw_prediction(
        image,
        prediction,
        alpha=0.5,
        subject_roi={
            "x_min": 10.0,
            "y_min": 10.0,
            "x_max": 50.0,
            "y_max": 50.0,
        },
    )

    assert np.any(result[75, 75] > 0)
