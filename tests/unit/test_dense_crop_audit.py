"""Tests for category-free full-query coarse crop diagnostics."""

import numpy as np
import pytest

from fashion_semantic_parser.service.dense_crop_audit import (
    CoarseCropBox,
    crop_target_coverage,
    select_query_peak_crops,
)
from fashion_semantic_parser.service.dinov2_region_encoder import (
    DinoV2LetterboxGeometry,
)


def _geometry() -> DinoV2LetterboxGeometry:
    """Return a square identity geometry for crop tests."""
    return DinoV2LetterboxGeometry(
        original_height=100,
        original_width=100,
        resized_height=100,
        resized_width=100,
        top=0,
        left=0,
        output_size=100,
    )


def test_query_peak_crops_are_score_driven_and_distinct() -> None:
    """Top crops should follow score peaks without target or category input."""
    scores = np.zeros((4, 4), dtype=np.float32)
    scores[0, 0] = 1.0
    scores[3, 3] = 0.9

    crops = select_query_peak_crops(
        scores,
        _geometry(),
        crop_fraction=0.30,
        max_crops=2,
    )

    assert len(crops) == 2
    assert crops[0] == CoarseCropBox(0, 0, 30, 30)
    assert crops[1] == CoarseCropBox(70, 70, 100, 100)


def test_crop_target_coverage_reports_target_and_image_area() -> None:
    """Coverage must distinguish target capture from crop area cost."""
    target = np.zeros((100, 100), dtype=bool)
    target[10:20, 10:20] = True

    coverage, area = crop_target_coverage(
        target,
        (CoarseCropBox(0, 0, 15, 20),),
    )

    assert coverage == 0.5
    assert area == 0.03


def test_crop_audit_rejects_invalid_geometry() -> None:
    """Invalid crop settings and out-of-bounds crops must fail early."""
    with pytest.raises(ValueError, match="invalid"):
        select_query_peak_crops(
            np.asarray([[np.nan]], dtype=np.float32),
            _geometry(),
            crop_fraction=0.30,
            max_crops=1,
        )
    target = np.ones((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="outside"):
        crop_target_coverage(
            target,
            (CoarseCropBox(0, 0, 11, 10),),
        )
