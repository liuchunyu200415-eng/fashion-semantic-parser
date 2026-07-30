"""Tests for single-image localization experiment overrides."""

import pytest

from scripts.predict_localization import _build_settings_overrides


def test_localization_prediction_builds_only_explicit_overrides() -> None:
    """CLI experiments should not silently replace committed config values."""
    assert _build_settings_overrides(
        box_threshold=0.29,
        text_threshold=None,
        max_regions=1,
        subject_roi_margin=None,
        full_image=False,
    ) == {
        "box_threshold": 0.29,
        "max_regions": 1,
    }


def test_full_image_rejects_unused_roi_margin() -> None:
    """An ROI margin on full-image inference is a contradictory experiment."""
    with pytest.raises(ValueError, match="cannot be used"):
        _build_settings_overrides(
            box_threshold=None,
            text_threshold=None,
            max_regions=None,
            subject_roi_margin=0.2,
            full_image=True,
        )
