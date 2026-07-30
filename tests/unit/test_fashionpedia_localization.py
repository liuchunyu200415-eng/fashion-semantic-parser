"""Tests for Fashionpedia PRD 3.1.2 local-part preparation."""

import json
from pathlib import Path

import pytest

from fashion_semantic_parser.dao.localization.fashionpedia import (
    audit_fashionpedia_part_annotations,
    convert_fashionpedia_parts_to_coco,
)
from fashion_semantic_parser.dao.localization.taxonomy import (
    PRD_LOCALIZATION_REGION_COVERAGE,
    map_fashionpedia_part_category,
    resolve_localization_prompt,
)


@pytest.mark.parametrize(
    ("source_name", "target_name", "region_group"),
    [
        ("collar", "collar", "collar"),
        ("neckline", "neckline", "collar"),
        ("epaulette", "epaulette", "shoulder"),
        ("pocket", "pocket", "pocket"),
        ("ruffle", "ruffle", "decoration"),
    ],
)
def test_map_fashionpedia_local_parts(
    source_name: str,
    target_name: str,
    region_group: str,
) -> None:
    """Official part labels should retain their direct semantic meaning."""
    category = map_fashionpedia_part_category(source_name)

    assert category is not None
    assert category.english_name == target_name
    assert category.region_group == region_group


def test_prd_coverage_does_not_relabel_sleeves_as_cuffs() -> None:
    """The audit must expose missing targets instead of inventing supervision."""
    coverage = {
        region.english_name: region for region in PRD_LOCALIZATION_REGION_COVERAGE
    }

    assert coverage["collar"].status == "exact"
    assert coverage["pocket"].status == "exact"
    assert coverage["shoulder"].status == "partial"
    assert coverage["cuff"].status == "missing"
    assert coverage["cuff"].source_categories == ()


@pytest.mark.parametrize(
    ("query", "label", "prompt"),
    [
        ("这件衣服的领口", "neckline", "neckline"),
        ("请找出袖口", "cuff", "cuff . sleeve cuff"),
        ("where is the pocket?", "pocket", "pocket"),
        ("定位拉链", "zipper", "zipper"),
        ("下摆区域", "hem", "garment hem . lower hem"),
    ],
)
def test_resolve_localization_prompt_normalizes_known_queries(
    query: str,
    label: str,
    prompt: str,
) -> None:
    """Known Chinese and English terms should reach the English text encoder."""
    resolved = resolve_localization_prompt(query)

    assert resolved.region_label == label
    assert resolved.grounding_prompt == prompt


def test_resolve_localization_prompt_preserves_unknown_free_form_text() -> None:
    """Open-vocabulary English remains available beyond the fixed taxonomy."""
    resolved = resolve_localization_prompt("silver logo near chest")

    assert resolved.region_label == "custom"
    assert resolved.grounding_prompt == "silver logo near chest"


def test_convert_fashionpedia_parts_preserves_masks_and_prompt_metadata(
    tmp_path: Path,
) -> None:
    """Conversion should keep only direct local-part masks in a separate COCO."""
    root = tmp_path / "fashionpedia"
    annotation_path = root / "annotations" / "instances_attributes_train2020.json"
    image_root = root / "train"
    output_path = tmp_path / "processed" / "fashionpedia_parts_train.json"
    annotation_path.parent.mkdir(parents=True)
    image_root.mkdir(parents=True)
    for file_name in ("a.jpg", "b.jpg", "c.jpg"):
        (image_root / file_name).write_bytes(b"fixture")

    source = {
        "categories": [
            {"id": 0, "name": "shirt, blouse"},
            {"id": 28, "name": "collar"},
            {"id": 31, "name": "sleeve"},
            {"id": 32, "name": "pocket"},
            {"id": 43, "name": "ruffle"},
        ],
        "images": [
            {"id": 100, "file_name": "a.jpg", "width": 100, "height": 200},
            {"id": 200, "file_name": "b.jpg", "width": 100, "height": 200},
            {"id": 300, "file_name": "c.jpg", "width": 100, "height": 200},
        ],
        "annotations": [
            _annotation(1, 100, 28, attribute_ids=[3, True, "invalid"]),
            _annotation(
                2,
                100,
                32,
                segmentation={"size": [200, 100], "counts": "encoded"},
            ),
            _annotation(3, 100, 0),
            _annotation(4, 200, 31, segmentation=[]),
            _annotation(5, 300, 43),
        ],
        "licenses": [{"id": 1, "name": "fixture"}],
    }
    annotation_path.write_text(json.dumps(source), encoding="utf-8")

    summary = convert_fashionpedia_parts_to_coco(
        root=root,
        split="train",
        output_path=output_path,
    )
    converted = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary.selected_part_annotation_count == 4
    assert summary.output_image_count == 2
    assert summary.output_annotation_count == 3
    assert summary.excluded_non_part_annotation_count == 1
    assert summary.invalid_part_annotation_count == 1
    assert summary.category_counts["collar"] == 1
    assert summary.category_counts["pocket"] == 1
    assert summary.category_counts["ruffle"] == 1
    assert summary.prd_region_coverage["cuff"] == "missing"
    assert summary.uncovered_prd_regions == ["cuff", "hem", "waist", "pattern"]
    assert converted["annotations"][0]["attribute_ids"] == [3]
    assert converted["annotations"][1]["segmentation"]["counts"] == "encoded"
    assert len(converted["categories"]) == 19
    collar = next(
        category for category in converted["categories"] if category["name"] == "collar"
    )
    assert "领子" in collar["prompt_terms"]
    assert converted["info"]["description"].endswith("PRD 3.1.2 localization")


def test_part_audit_allows_missing_images_but_conversion_rejects_them(
    tmp_path: Path,
) -> None:
    """Part labels can be audited before the official image archive is present."""
    root = tmp_path / "fashionpedia"
    annotation_path = root / "annotations" / "instances_attributes_val2020.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 32, "name": "pocket"}],
                "images": [
                    {
                        "id": 1,
                        "file_name": "missing.jpg",
                        "width": 100,
                        "height": 200,
                    }
                ],
                "annotations": [_annotation(1, 1, 32)],
            }
        ),
        encoding="utf-8",
    )

    summary = audit_fashionpedia_part_annotations(
        root=root,
        split="validation",
    )

    assert summary.output_image_count == 1
    assert summary.output_annotation_count == 1
    assert summary.missing_image_count == 1
    with pytest.raises(FileNotFoundError, match="--audit-only"):
        convert_fashionpedia_parts_to_coco(
            root=root,
            split="validation",
            output_path=tmp_path / "output.json",
        )
    assert not (tmp_path / "output.json").exists()


def _annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    *,
    segmentation: object | None = None,
    attribute_ids: object | None = None,
) -> dict[str, object]:
    """Build one compact official-style Fashionpedia annotation."""
    if segmentation is None:
        segmentation = [[0, 0, 10, 0, 10, 10, 0, 10]]
    if attribute_ids is None:
        attribute_ids = []
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "attribute_ids": attribute_ids,
        "segmentation": segmentation,
        "bbox": [0, 0, 10, 10],
        "area": 100,
        "iscrowd": 0,
    }
