"""Tests for Fashionpedia query-region DINOv2 training data preparation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fashion_semantic_parser.dao.localization.referring_training import (
    ReferringTrainingSample,
    prepare_fashionpedia_referring_training_data,
)


def test_prepare_referring_training_data_covers_composable_language(
    tmp_path: Path,
) -> None:
    """Generate broad, spatial, direct-attribute, and reliable relation records."""
    root = _fashionpedia_root(tmp_path)
    output_path = tmp_path / "processed" / "referring.jsonl"
    summary_path = tmp_path / "outputs" / "summary.json"
    source = {
        "categories": [
            {"id": 4, "name": "jacket"},
            {"id": 32, "name": "pocket"},
            {"id": 35, "name": "zipper"},
        ],
        "attributes": [
            {"id": 100, "name": "red"},
            {"id": 101, "name": "striped"},
        ],
        "images": [{"id": 10, "file_name": "a.jpg", "width": 200, "height": 200}],
        "annotations": [
            _annotation(1, 10, 4, [20, 10, 160, 180], attributes=[]),
            _annotation(2, 10, 32, [40, 60, 20, 20], attributes=[100]),
            _annotation(3, 10, 32, [140, 64, 20, 20], attributes=[100, 101]),
            _annotation(4, 10, 35, [95, 30, 10, 100], attributes=None),
        ],
    }
    _write_train_source(root, source, image_names=("a.jpg",))

    summary = prepare_fashionpedia_referring_training_data(
        root=root,
        split="train",
        output_path=output_path,
        summary_output_path=summary_path,
    )
    samples = _read_jsonl(output_path)

    broad_pocket = next(
        sample
        for sample in samples
        if sample["template_id"] == "basic-zh" and sample["target_label"] == "pocket"
    )
    assert broad_pocket["query"] == "这件衣服的口袋"
    assert [target["source_annotation_id"] for target in broad_pocket["targets"]] == [
        2,
        3,
    ]

    right_pocket = next(
        sample
        for sample in samples
        if sample["template_id"] == "spatial-right-en"
        and sample["target_label"] == "pocket"
    )
    assert right_pocket["reference_frame"] == "image"
    assert right_pocket["targets"][0]["source_annotation_id"] == 3
    assert not any(
        sample["template_id"].startswith("spatial-upper")
        and sample["target_label"] == "pocket"
        for sample in samples
    )

    red_pocket = next(
        sample
        for sample in samples
        if sample["template_id"] == "attribute-100-en"
        and sample["target_label"] == "pocket"
    )
    assert red_pocket["query"] == "the pocket with red"
    assert len(red_pocket["targets"]) == 2
    striped_pocket = next(
        sample for sample in samples if sample["template_id"] == "attribute-101-en"
    )
    assert striped_pocket["targets"][0]["source_annotation_id"] == 3

    jacket_pocket = next(
        sample
        for sample in samples
        if sample["template_id"] == "relation-jacket-zh"
        and sample["target_label"] == "pocket"
    )
    assert jacket_pocket["query"] == "这件夹克上的口袋"
    assert jacket_pocket["reference_category"] == "jacket"
    assert len(jacket_pocket["targets"]) == 2

    assert summary.valid_part_annotation_count == 3
    assert summary.relation_association_count == 3
    assert summary.unmatched_relation_part_count == 0
    assert summary.dimension_counts["basic"] == summary.output_sample_count
    assert summary.dimension_counts["spatial"] == 4
    assert summary.dimension_counts["attribute"] == 2
    assert summary.dimension_counts["relation"] == 4
    assert (
        json.loads(summary_path.read_text(encoding="utf-8"))["mask_storage"]
        == "source_annotation_reference"
    )


def test_ambiguous_containment_does_not_create_relation_labels(tmp_path: Path) -> None:
    """Nested garments must not silently create a guessed part relationship."""
    root = _fashionpedia_root(tmp_path)
    source = {
        "categories": [
            {"id": 0, "name": "shirt, blouse"},
            {"id": 4, "name": "jacket"},
            {"id": 32, "name": "pocket"},
        ],
        "images": [{"id": 10, "file_name": "a.jpg", "width": 200, "height": 200}],
        "annotations": [
            _annotation(1, 10, 0, [10, 10, 180, 180]),
            _annotation(2, 10, 4, [20, 20, 160, 160]),
            _annotation(3, 10, 32, [60, 60, 20, 20]),
        ],
    }
    _write_train_source(root, source, image_names=("a.jpg",))
    output_path = tmp_path / "referring.jsonl"

    summary = prepare_fashionpedia_referring_training_data(
        root=root,
        split="train",
        output_path=output_path,
        summary_output_path=tmp_path / "summary.json",
    )
    samples = _read_jsonl(output_path)

    assert summary.relation_association_count == 0
    assert summary.unmatched_relation_part_count == 1
    assert not any("relation" in sample["dimensions"] for sample in samples)


def test_missing_part_image_keeps_final_output_atomic(tmp_path: Path) -> None:
    """A missing image must fail without leaving a partial JSONL index."""
    root = _fashionpedia_root(tmp_path)
    source = {
        "categories": [{"id": 32, "name": "pocket"}],
        "images": [{"id": 10, "file_name": "missing.jpg", "width": 100, "height": 100}],
        "annotations": [_annotation(1, 10, 32, [10, 10, 20, 20])],
    }
    _write_train_source(root, source, image_names=())
    output_path = tmp_path / "referring.jsonl"

    with pytest.raises(FileNotFoundError, match="missing 1 image"):
        prepare_fashionpedia_referring_training_data(
            root=root,
            split="train",
            output_path=output_path,
            summary_output_path=tmp_path / "summary.json",
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".jsonl.tmp").exists()


def test_training_sample_rejects_spatial_without_reference_frame() -> None:
    """Training records must not leave left/right coordinate meaning implicit."""
    with pytest.raises(ValidationError, match="reference frame"):
        ReferringTrainingSample.model_validate(
            {
                "id": "fashionpedia-train-1-pocket-spatial-left-en-2",
                "split": "train",
                "image_path": "data/raw/fashionpedia/train/a.jpg",
                "source_image_id": 1,
                "query": "the left pocket",
                "language": "en",
                "dimensions": ["basic", "spatial"],
                "target_label": "pocket",
                "targets": [
                    {
                        "source_annotation_id": 2,
                        "label": "pocket",
                        "box": {"x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2},
                    }
                ],
                "template_id": "spatial-left-en",
            }
        )


def _fashionpedia_root(tmp_path: Path) -> Path:
    root = tmp_path / "fashionpedia"
    (root / "annotations").mkdir(parents=True)
    (root / "train").mkdir(parents=True)
    return root


def _write_train_source(
    root: Path,
    source: dict[str, object],
    *,
    image_names: tuple[str, ...],
) -> None:
    for image_name in image_names:
        (root / "train" / image_name).write_bytes(b"fixture")
    (root / "annotations" / "instances_attributes_train2020.json").write_text(
        json.dumps(source),
        encoding="utf-8",
    )


def _annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    bbox: list[int],
    *,
    attributes: object = (),
) -> dict[str, object]:
    if attributes is None:
        attributes = None
    elif attributes == ():
        attributes = []
    x, y, width, height = bbox
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "segmentation": [[x, y, x + width, y, x + width, y + height, x, y + height]],
        "area": width * height,
        "iscrowd": 0,
        "attribute_ids": attributes,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
