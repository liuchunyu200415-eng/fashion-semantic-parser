"""Tests for Fashionpedia PRD segmentation preparation."""

import json
from pathlib import Path

import pytest

from fashion_semantic_parser.dao.segmentation.fashionpedia import (
    audit_fashionpedia_annotations,
    convert_fashionpedia_to_coco,
)
from fashion_semantic_parser.dao.segmentation.taxonomy import (
    fashionpedia_category_exclusion_reason,
    map_fashionpedia_category,
)


@pytest.mark.parametrize(
    ("source_name", "target_name"),
    [
        ("shirt, blouse", "top"),
        ("cardigan", "outerwear"),
        ("shorts", "pants"),
        ("shoe", "shoes"),
        ("bag, wallet", "bag"),
        ("hat", "accessory"),
    ],
)
def test_map_fashionpedia_main_categories(
    source_name: str,
    target_name: str,
) -> None:
    """Fashionpedia main apparel should map to stable PRD category IDs."""
    category = map_fashionpedia_category(source_name)

    assert category is not None
    assert category.english_name == target_name


def test_fashionpedia_parts_and_ambiguous_apparel_are_explicitly_excluded() -> None:
    """Parts and jumpsuits require different handling from mapped apparel."""
    assert map_fashionpedia_category("sleeve") is None
    assert map_fashionpedia_category("jumpsuit") is None
    assert fashionpedia_category_exclusion_reason("sleeve") == "garment_part"
    assert (
        fashionpedia_category_exclusion_reason("jumpsuit") == "ambiguous_main_apparel"
    )
    assert fashionpedia_category_exclusion_reason("new category") == "unknown"


def test_convert_fashionpedia_writes_only_unambiguous_main_apparel(
    tmp_path: Path,
) -> None:
    """Conversion should remap targets without turning jumpsuits into background."""
    root = tmp_path / "fashionpedia"
    annotation_path = root / "annotations" / "instances_attributes_train2020.json"
    image_root = root / "train"
    output_path = tmp_path / "processed" / "fashionpedia_train.json"
    annotation_path.parent.mkdir(parents=True)
    image_root.mkdir(parents=True)
    for file_name in ("a.jpg", "b.jpg", "c.jpg"):
        (image_root / file_name).write_bytes(b"fixture")

    source = {
        "categories": [
            {"id": 0, "name": "shirt, blouse"},
            {"id": 11, "name": "jumpsuit"},
            {"id": 14, "name": "hat"},
            {"id": 23, "name": "shoe"},
            {"id": 24, "name": "bag, wallet"},
            {"id": 31, "name": "sleeve"},
        ],
        "images": [
            {
                "id": 100,
                "file_name": "a.jpg",
                "width": 100,
                "height": 200,
                "license": 1,
            },
            {"id": 200, "file_name": "b.jpg", "width": 100, "height": 200},
            {"id": 300, "file_name": "c.jpg", "width": 100, "height": 200},
        ],
        "annotations": [
            _annotation(1, 100, 23),
            _annotation(
                2,
                100,
                24,
                segmentation={"size": [200, 100], "counts": "encoded"},
            ),
            _annotation(3, 100, 14),
            _annotation(4, 100, 31),
            _annotation(5, 200, 11),
            _annotation(6, 200, 0),
            _annotation(7, 300, 0, segmentation=[]),
        ],
        "licenses": [{"id": 1, "name": "fixture"}],
    }
    annotation_path.write_text(json.dumps(source), encoding="utf-8")

    summary = convert_fashionpedia_to_coco(
        root=root,
        split="train",
        output_path=output_path,
    )
    converted = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary.source_image_count == 3
    assert summary.source_annotation_count == 7
    assert summary.output_image_count == 1
    assert summary.output_annotation_count == 3
    assert summary.dropped_ambiguous_image_count == 1
    assert summary.dropped_ambiguous_image_annotation_count == 2
    assert summary.dropped_empty_image_count == 1
    assert summary.excluded_part_annotation_count == 1
    assert summary.invalid_annotation_count == 1
    assert summary.category_counts["shoes"] == 1
    assert summary.category_counts["bag"] == 1
    assert summary.category_counts["accessory"] == 1
    assert converted["images"][0]["source_image_id"] == 100
    assert converted["images"][0]["file_name"].endswith("train/a.jpg")
    assert [row["category_id"] for row in converted["annotations"]] == [6, 7, 8]
    assert converted["annotations"][1]["segmentation"]["counts"] == "encoded"
    assert len(converted["categories"]) == 8


def test_audit_allows_missing_images_but_conversion_rejects_them(
    tmp_path: Path,
) -> None:
    """Annotation audits may run first, while training files require images."""
    root = tmp_path / "fashionpedia"
    annotation_path = root / "annotations" / "instances_attributes_val2020.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 23, "name": "shoe"}],
                "images": [
                    {
                        "id": 1,
                        "file_name": "missing.jpg",
                        "width": 100,
                        "height": 200,
                    }
                ],
                "annotations": [_annotation(1, 1, 23)],
            }
        ),
        encoding="utf-8",
    )

    summary = audit_fashionpedia_annotations(root=root, split="validation")

    assert summary.output_image_count == 1
    assert summary.output_annotation_count == 1
    assert summary.missing_image_count == 1
    with pytest.raises(FileNotFoundError, match="--audit-only"):
        convert_fashionpedia_to_coco(
            root=root,
            split="validation",
            output_path=tmp_path / "output.json",
        )
    assert not (tmp_path / "output.json").exists()


def test_convert_fashionpedia_accepts_legacy_image_directory(
    tmp_path: Path,
) -> None:
    """Older train2020/val2020 layouts should remain usable."""
    root = tmp_path / "fashionpedia"
    annotation_path = root / "annotations" / "instances_attributes_val2020.json"
    image_root = root / "val2020"
    output_path = tmp_path / "output.json"
    annotation_path.parent.mkdir(parents=True)
    image_root.mkdir(parents=True)
    (image_root / "legacy.jpg").write_bytes(b"fixture")
    annotation_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 23, "name": "shoe"}],
                "images": [
                    {
                        "id": 1,
                        "file_name": "legacy.jpg",
                        "width": 100,
                        "height": 200,
                    }
                ],
                "annotations": [_annotation(1, 1, 23)],
            }
        ),
        encoding="utf-8",
    )

    summary = convert_fashionpedia_to_coco(
        root=root,
        split="validation",
        output_path=output_path,
    )
    converted = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary.missing_image_count == 0
    assert converted["images"][0]["file_name"].endswith("val2020/legacy.jpg")


def _annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    *,
    segmentation: object | None = None,
) -> dict[str, object]:
    """Build one compact official-style Fashionpedia annotation."""
    if segmentation is None:
        segmentation = [[0, 0, 10, 0, 10, 10, 0, 10]]
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "bbox": [0, 0, 10, 10],
        "area": 100,
        "iscrowd": 0,
    }
