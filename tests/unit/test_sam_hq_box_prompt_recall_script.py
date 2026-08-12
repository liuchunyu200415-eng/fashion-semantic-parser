"""Tests for controlled SAM-HQ oracle Box diagnostics."""

import argparse

import pytest

from scripts.smoke_sam_hq_box_prompt_recall import (
    _expand_box,
    _parse_expansion_ratios,
    _scaled_crop_box,
)


def test_expansion_ratios_require_unique_zero_baseline() -> None:
    """A valid sweep should preserve ordered unique ratios including zero."""
    assert _parse_expansion_ratios("0,0.1,0.2") == (0.0, 0.1, 0.2)

    with pytest.raises(argparse.ArgumentTypeError, match="include zero"):
        _parse_expansion_ratios("0.1,0.2")
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        _parse_expansion_ratios("0,0")


def test_box_expansion_clamps_each_side_to_the_image() -> None:
    """Per-side margins must never create out-of-image coordinates."""
    assert _expand_box(
        (5.0, 10.0, 15.0, 30.0),
        0.1,
        image_width=100,
        image_height=100,
    ) == (4.0, 8.0, 16.0, 32.0)
    assert _expand_box(
        (0.0, 0.0, 10.0, 10.0),
        0.2,
        image_width=100,
        image_height=100,
    ) == (0.0, 0.0, 12.0, 12.0)


def test_scaled_crop_box_preserves_context_and_image_bounds() -> None:
    """ROI crops should add symmetric context and clamp at image edges."""
    assert _scaled_crop_box(
        (10.0, 20.0, 30.0, 40.0),
        2.0,
        image_width=100,
        image_height=100,
    ) == (0, 10, 40, 50)
    assert _scaled_crop_box(
        (0.0, 0.0, 10.0, 10.0),
        4.0,
        image_width=100,
        image_height=100,
    ) == (0, 0, 25, 25)
