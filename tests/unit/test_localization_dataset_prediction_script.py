"""Tests for category-specific localization dataset prediction helpers."""

from types import SimpleNamespace

import pytest

from scripts.predict_localization_dataset import (
    _build_settings_overrides,
    _select_category_images,
    prediction_to_coco_results,
)


def test_select_category_images_is_exact_sorted_and_limitable() -> None:
    """Evaluation must use deterministic images with exact-category ground truth."""
    source = {
        "categories": [
            {"id": 1, "name": "collar"},
            {"id": 2, "name": "lapel"},
        ],
        "images": [
            {"id": 20, "file_name": "second.jpg"},
            {"id": 10, "file_name": "first.jpg"},
            {"id": 30, "file_name": "lapel.jpg"},
        ],
        "annotations": [
            {"image_id": 20, "category_id": 1, "iscrowd": 0},
            {"image_id": 10, "category_id": 1, "iscrowd": 0},
            {"image_id": 30, "category_id": 2, "iscrowd": 0},
        ],
    }

    category_id, images = _select_category_images(
        source,
        category_name="collar",
        image_limit=1,
    )

    assert category_id == 1
    assert [image["id"] for image in images] == [10]


def test_localization_prediction_converts_regions_to_coco() -> None:
    """Saved candidates should retain score, mask, and mask-derived box."""
    prediction = SimpleNamespace(
        regions=[
            SimpleNamespace(
                mask=[[1.0, 2.0, 5.0, 2.0, 5.0, 8.0, 1.0, 8.0]],
                box=SimpleNamespace(
                    x_min=1.0,
                    y_min=2.0,
                    x_max=5.0,
                    y_max=8.0,
                ),
                confidence=0.81,
            )
        ]
    )

    results = prediction_to_coco_results(
        prediction,
        image_id=12,
        category_id=3,
    )

    assert results == [
        {
            "image_id": 12,
            "category_id": 3,
            "bbox": [1.0, 2.0, 4.0, 6.0],
            "score": 0.81,
            "segmentation": [[1.0, 2.0, 5.0, 2.0, 5.0, 8.0, 1.0, 8.0]],
        }
    ]


def test_subject_roi_margin_requires_auto_mode() -> None:
    """A crop margin should not appear in a full-image run summary."""
    with pytest.raises(ValueError, match="requires --roi-mode auto"):
        _build_settings_overrides(
            roi_mode="full",
            box_threshold=None,
            text_threshold=None,
            max_regions=None,
            subject_roi_margin=0.35,
        )
