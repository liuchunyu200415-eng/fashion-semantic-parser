"""Tests for automatic primary-person ROI selection."""

from fashion_semantic_parser.service.subject_roi import select_primary_person_roi


def test_primary_person_prefers_large_central_fashion_subject() -> None:
    """A central model should outrank a similarly large edge bystander."""
    roi = select_primary_person_roi(
        boxes=[
            [250.0, 50.0, 750.0, 950.0],
            [0.0, 0.0, 500.0, 1000.0],
            [100.0, 100.0, 900.0, 900.0],
        ],
        scores=[0.9, 0.95, 0.99],
        classes=[0, 0, 2],
        image_width=1000,
        image_height=1000,
    )

    assert roi is not None
    assert roi.model_dump() == {
        "x_min": 250.0,
        "y_min": 50.0,
        "x_max": 750.0,
        "y_max": 950.0,
    }


def test_primary_person_filters_low_confidence_and_tiny_boxes() -> None:
    """Weak or extremely small person detections should trigger fallback."""
    roi = select_primary_person_roi(
        boxes=[
            [100.0, 100.0, 900.0, 900.0],
            [490.0, 490.0, 510.0, 510.0],
        ],
        scores=[0.69, 0.99],
        classes=[0, 0],
        image_width=1000,
        image_height=1000,
        score_threshold=0.7,
        min_area_ratio=0.005,
    )

    assert roi is None


def test_primary_person_clamps_detection_to_image_bounds() -> None:
    """Detector boxes should never create invalid crop coordinates."""
    roi = select_primary_person_roi(
        boxes=[[-20.0, -10.0, 220.0, 310.0]],
        scores=[0.95],
        classes=[0],
        image_width=200,
        image_height=300,
    )

    assert roi is not None
    assert roi.model_dump() == {
        "x_min": 0.0,
        "y_min": 0.0,
        "x_max": 200.0,
        "y_max": 300.0,
    }
