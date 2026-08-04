"""Tests for the PRD 3.1.2 API acceptance overview renderer."""

import numpy as np
import pytest

from scripts.visualize_localization_api_acceptance import (
    CARD_HEADER_HEIGHT,
    CARD_IMAGE_HEIGHT,
    HEADER_HEIGHT,
    PANEL_LABEL_HEIGHT,
    _build_case_card,
    _build_overview,
    _fit_to_canvas,
    _row_passed,
)


def test_fit_to_canvas_letterboxes_without_distortion() -> None:
    """A wide source should be centered with vertical padding."""
    image = np.full((20, 40, 3), 100, dtype=np.uint8)

    canvas = _fit_to_canvas(image, width=40, height=40)

    assert canvas.shape == (40, 40, 3)
    assert np.all(canvas[10:30] == 100)
    assert np.all(canvas[:10] == 0)
    assert np.all(canvas[30:] == 0)


def test_case_card_has_stable_dimensions() -> None:
    """Different source dimensions must not change acceptance card geometry."""
    original = np.zeros((60, 40, 3), dtype=np.uint8)
    prediction = np.zeros((20, 80, 3), dtype=np.uint8)
    row = {
        "target_region": "collar",
        "expected_detected": True,
        "source_matched": True,
        "all_masks_present": True,
        "all_boxes_valid": True,
        "subject_roi_source": "detected",
        "subject_roi_present": True,
        "segmentation_present": True,
        "region_count": 1,
        "elapsed_seconds": 1.25,
    }

    card = _build_case_card(
        original,
        None,
        prediction,
        row,
        index=1,
        card_width=960,
    )

    assert card.shape == (
        CARD_HEADER_HEIGHT + PANEL_LABEL_HEIGHT + CARD_IMAGE_HEIGHT,
        960,
        3,
    )


def test_overview_arranges_eight_cards_in_two_columns() -> None:
    """The default layout should create four rows for eight PRD cases."""
    cards = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(8)]

    overview = _build_overview(
        cards,
        columns=2,
        accepted=True,
        passed_count=8,
        total_seconds=11.51,
    )

    assert overview.shape[0] > HEADER_HEIGHT + 4 * 100
    assert overview.shape[1] > 2 * 200


def test_overview_rejects_mixed_card_dimensions() -> None:
    """Cards with inconsistent dimensions should fail before assignment."""
    cards = [
        np.zeros((100, 200, 3), dtype=np.uint8),
        np.zeros((90, 200, 3), dtype=np.uint8),
    ]

    with pytest.raises(ValueError, match="same dimensions"):
        _build_overview(
            cards,
            columns=2,
            accepted=False,
            passed_count=1,
            total_seconds=1.0,
        )


def test_row_pass_requires_valid_roi_and_segmentation() -> None:
    """A visual PASS must match the API runner's complete functional checks."""
    row = {
        "expected_detected": True,
        "source_matched": True,
        "subject_roi_source": "detected",
        "subject_roi_present": True,
        "segmentation_present": True,
        "all_masks_present": True,
        "all_boxes_valid": True,
    }

    assert _row_passed(row) is True
    assert _row_passed({**row, "segmentation_present": False}) is False
    assert _row_passed({**row, "subject_roi_present": False}) is False
