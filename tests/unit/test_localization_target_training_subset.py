"""Tests for targeted PRD 3.1.2 class-replay subsets."""

import json
from pathlib import Path

import pytest

from fashion_semantic_parser.dao.localization.training_subset import (
    build_localization_target_coco_subset,
)


def _write_source(path: Path) -> None:
    """Write a compact localization COCO fixture."""
    source = {
        "images": [
            {"id": 1, "file_name": "one.jpg"},
            {"id": 2, "file_name": "two.jpg"},
        ],
        "categories": [
            {"id": 1, "name": "collar"},
            {"id": 2, "name": "rivet"},
            {"id": 3, "name": "tassel"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2},
            {"id": 2, "image_id": 1, "category_id": 1},
            {"id": 3, "image_id": 2, "category_id": 3, "iscrowd": 1},
        ],
    }
    path.write_text(json.dumps(source), encoding="utf-8")


def test_target_subset_keeps_complete_selected_images(tmp_path: Path) -> None:
    """Replay records retain contextual classes from every selected image."""
    source_path = tmp_path / "source.json"
    output_path = tmp_path / "subset.json"
    _write_source(source_path)

    summary = build_localization_target_coco_subset(
        source_path=source_path,
        output_path=output_path,
        target_categories=["rivet"],
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert [image["id"] for image in output["images"]] == [1]
    assert [annotation["id"] for annotation in output["annotations"]] == [1, 2]
    assert summary.target_annotation_counts == {"rivet": 1}
    assert summary.selected_annotation_count == 2


def test_target_subset_audit_does_not_write_output(tmp_path: Path) -> None:
    """Audit mode returns counts without creating a training artifact."""
    source_path = tmp_path / "source.json"
    _write_source(source_path)

    summary = build_localization_target_coco_subset(
        source_path=source_path,
        output_path=None,
        target_categories=["rivet"],
    )

    assert summary.output_path is None


def test_target_subset_rejects_unknown_category(tmp_path: Path) -> None:
    """Category typos fail before a long training run starts."""
    source_path = tmp_path / "source.json"
    _write_source(source_path)

    with pytest.raises(ValueError, match="Unknown localization target"):
        build_localization_target_coco_subset(
            source_path=source_path,
            output_path=None,
            target_categories=["button"],
        )
