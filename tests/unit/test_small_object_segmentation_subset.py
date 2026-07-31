"""Tests for targeted small-object COCO training subsets."""

import json
from pathlib import Path

import pytest

from fashion_semantic_parser.dao.segmentation.small_objects import (
    build_small_object_coco_subset,
)


def _write_source(path: Path) -> None:
    """Write a compact eight-class-like COCO fixture."""
    source = {
        "info": {"description": "fixture"},
        "images": [
            {"id": 1, "file_name": "one.jpg"},
            {"id": 2, "file_name": "two.jpg"},
            {"id": 3, "file_name": "three.jpg"},
        ],
        "categories": [
            {"id": 1, "name": "top"},
            {"id": 6, "name": "shoes"},
            {"id": 7, "name": "bag"},
            {"id": 8, "name": "accessory"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 6, "area": 400},
            {"id": 2, "image_id": 1, "category_id": 1, "area": 5000},
            {"id": 3, "image_id": 2, "category_id": 7, "area": 1600},
            {"id": 4, "image_id": 3, "category_id": 8, "bbox": [0, 0, 20, 20]},
        ],
    }
    path.write_text(json.dumps(source), encoding="utf-8")


def test_small_object_subset_keeps_complete_selected_images(tmp_path: Path) -> None:
    """A selected image should retain target and contextual annotations."""
    source_path = tmp_path / "source.json"
    output_path = tmp_path / "subset.json"
    _write_source(source_path)

    summary = build_small_object_coco_subset(
        source_path=source_path,
        output_path=output_path,
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert [image["id"] for image in output["images"]] == [1, 3]
    assert [annotation["id"] for annotation in output["annotations"]] == [1, 2, 4]
    assert (
        output["categories"]
        == json.loads(source_path.read_text(encoding="utf-8"))["categories"]
    )
    assert summary.selected_image_count == 2
    assert summary.selected_annotation_count == 3
    assert summary.target_small_annotation_count == 2
    assert summary.target_small_category_counts == {
        "shoes": 1,
        "bag": 0,
        "accessory": 1,
    }


def test_small_object_subset_audit_does_not_write_output(tmp_path: Path) -> None:
    """Audit mode is represented by an absent output path."""
    source_path = tmp_path / "source.json"
    _write_source(source_path)

    summary = build_small_object_coco_subset(
        source_path=source_path,
        output_path=None,
        target_categories=["shoes"],
    )

    assert summary.output_path is None
    assert summary.selected_image_count == 1


def test_small_object_subset_rejects_unknown_category(tmp_path: Path) -> None:
    """A category typo must fail before writing a misleading subset."""
    source_path = tmp_path / "source.json"
    _write_source(source_path)

    with pytest.raises(ValueError, match="Unknown COCO target categories"):
        build_small_object_coco_subset(
            source_path=source_path,
            output_path=None,
            target_categories=["shoe"],
        )
