"""Prepare Fashionpedia local-part masks for PRD 3.1.2 localization."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fashion_semantic_parser.common.paths import to_project_relative_path
from fashion_semantic_parser.dao.fashionpedia import (
    annotation_area,
    category_records_by_id,
    dict_records,
    image_sort_key,
    is_integer,
    is_positive_number,
    normalize_coco_bbox_xywh,
    normalize_coco_segmentation,
    read_fashionpedia_json,
    resolve_fashionpedia_split_paths,
    safe_image_path,
    source_category_name,
)
from fashion_semantic_parser.dao.localization.taxonomy import (
    FASHIONPEDIA_PART_CATEGORIES,
    PRD_LOCALIZATION_REGION_COVERAGE,
    localization_coco_categories,
    map_fashionpedia_part_category,
)


class FashionpediaPartPreparationSummary(BaseModel):
    """Audit and conversion counts for one Fashionpedia part split."""

    split: str
    source_image_count: int
    source_annotation_count: int
    selected_image_count: int
    selected_annotation_count: int
    selected_part_annotation_count: int
    output_image_count: int
    output_annotation_count: int
    missing_image_count: int
    dropped_invalid_image_count: int
    dropped_empty_image_count: int
    excluded_non_part_annotation_count: int
    invalid_part_annotation_count: int
    category_counts: dict[str, int]
    region_group_counts: dict[str, int]
    prd_region_coverage: dict[str, str]
    uncovered_prd_regions: list[str]
    annotation_path: str
    output_path: str | None = None


def audit_fashionpedia_part_annotations(
    root: Path,
    split: str,
    limit: int | None = None,
) -> FashionpediaPartPreparationSummary:
    """Audit Fashionpedia local-part masks without requiring image files."""
    annotation_path, image_root = resolve_fashionpedia_split_paths(root, split)
    _, summary = _prepare_fashionpedia_part_coco(
        annotation_path=annotation_path,
        image_root=image_root,
        split=split,
        limit=limit,
    )
    return summary


def convert_fashionpedia_parts_to_coco(
    root: Path,
    split: str,
    output_path: Path,
    limit: int | None = None,
) -> FashionpediaPartPreparationSummary:
    """Convert one Fashionpedia split to a local-part COCO dataset."""
    annotation_path, image_root = resolve_fashionpedia_split_paths(root, split)
    coco, summary = _prepare_fashionpedia_part_coco(
        annotation_path=annotation_path,
        image_root=image_root,
        split=split,
        limit=limit,
    )
    if summary.missing_image_count:
        raise FileNotFoundError(
            f"Fashionpedia {split} is missing {summary.missing_image_count} "
            f"selected part image file(s) under {image_root}. Run --audit-only "
            "until the official image archive has been extracted."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    converted_summary: FashionpediaPartPreparationSummary = summary.model_copy(
        update={"output_path": to_project_relative_path(output_path)}
    )
    return converted_summary


def _prepare_fashionpedia_part_coco(
    *,
    annotation_path: Path,
    image_root: Path,
    split: str,
    limit: int | None,
) -> tuple[dict[str, object], FashionpediaPartPreparationSummary]:
    """Build local-part records while preserving annotation coverage limits."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to zero")

    source = read_fashionpedia_json(annotation_path)
    source_images = dict_records(source.get("images"))
    source_annotations = dict_records(source.get("annotations"))
    source_categories = dict_records(source.get("categories"))
    category_by_id = category_records_by_id(source_categories)

    selected_images = sorted(source_images, key=image_sort_key)
    if limit is not None:
        selected_images = selected_images[:limit]
    selected_image_ids = {
        image["id"] for image in selected_images if is_integer(image.get("id"))
    }
    selected_annotations = [
        annotation
        for annotation in source_annotations
        if annotation.get("image_id") in selected_image_ids
    ]
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in selected_annotations:
        image_id = annotation.get("image_id")
        if is_integer(image_id):
            annotations_by_image[image_id].append(annotation)

    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    region_group_counts: Counter[str] = Counter()
    selected_part_annotation_count = 0
    missing_image_count = 0
    dropped_invalid_image_count = 0
    dropped_empty_image_count = 0
    excluded_non_part_annotation_count = 0
    invalid_part_annotation_count = 0

    for source_image in selected_images:
        source_image_id = source_image.get("id")
        file_name = source_image.get("file_name")
        width = source_image.get("width")
        height = source_image.get("height")
        if not (
            is_integer(source_image_id)
            and isinstance(file_name, str)
            and is_positive_number(width)
            and is_positive_number(height)
        ):
            dropped_invalid_image_count += 1
            continue

        converted_rows: list[dict[str, object]] = []
        for source_annotation in annotations_by_image.get(source_image_id, []):
            source_category_name_value = source_category_name(
                source_annotation,
                category_by_id,
            )
            category = map_fashionpedia_part_category(source_category_name_value)
            if category is None:
                excluded_non_part_annotation_count += 1
                continue

            selected_part_annotation_count += 1
            segmentation = normalize_coco_segmentation(
                source_annotation.get("segmentation")
            )
            bbox = normalize_coco_bbox_xywh(source_annotation.get("bbox"))
            if segmentation is None or bbox is None:
                invalid_part_annotation_count += 1
                continue

            converted_rows.append(
                {
                    "category_id": category.id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": annotation_area(source_annotation.get("area"), bbox),
                    "iscrowd": int(source_annotation.get("iscrowd") == 1),
                    "attribute_ids": _attribute_ids(
                        source_annotation.get("attribute_ids")
                    ),
                    "source_annotation_id": source_annotation.get("id"),
                    "source_category_id": source_annotation.get("category_id"),
                    "target_category_name": category.english_name,
                    "region_group": category.region_group,
                }
            )

        if not converted_rows:
            dropped_empty_image_count += 1
            continue

        image_path = safe_image_path(image_root, file_name)
        if not image_path.is_file():
            missing_image_count += 1

        output_image_id = len(images) + 1
        image_record: dict[str, object] = {
            "id": output_image_id,
            "file_name": to_project_relative_path(image_path),
            "width": int(width),
            "height": int(height),
            "source_dataset": "fashionpedia",
            "source_image_id": source_image_id,
        }
        for optional_key in ("license", "original_url", "kaggle_id"):
            optional_value = source_image.get(optional_key)
            if optional_value is not None:
                image_record[optional_key] = optional_value
        images.append(image_record)

        for converted_row in converted_rows:
            category_name = str(converted_row.pop("target_category_name"))
            region_group = str(converted_row["region_group"])
            converted_row.update(
                {
                    "id": len(annotations) + 1,
                    "image_id": output_image_id,
                }
            )
            annotations.append(converted_row)
            category_counts[category_name] += 1
            region_group_counts[region_group] += 1

    coverage: dict[str, str] = {
        region.english_name: region.status
        for region in PRD_LOCALIZATION_REGION_COVERAGE
    }
    coco: dict[str, object] = {
        "info": {
            "description": "Fashionpedia parts for PRD 3.1.2 localization",
            "source": "https://fashionpedia.github.io/",
            "coverage_note": (
                "Direct part masks do not cover cuff, hem, waist, or general "
                "pattern regions."
            ),
        },
        "images": images,
        "annotations": annotations,
        "categories": localization_coco_categories(),
        "licenses": source.get("licenses", []),
    }
    summary = FashionpediaPartPreparationSummary(
        split=split,
        source_image_count=len(source_images),
        source_annotation_count=len(source_annotations),
        selected_image_count=len(selected_images),
        selected_annotation_count=len(selected_annotations),
        selected_part_annotation_count=selected_part_annotation_count,
        output_image_count=len(images),
        output_annotation_count=len(annotations),
        missing_image_count=missing_image_count,
        dropped_invalid_image_count=dropped_invalid_image_count,
        dropped_empty_image_count=dropped_empty_image_count,
        excluded_non_part_annotation_count=excluded_non_part_annotation_count,
        invalid_part_annotation_count=invalid_part_annotation_count,
        category_counts={
            category.english_name: category_counts[category.english_name]
            for category in FASHIONPEDIA_PART_CATEGORIES
        },
        region_group_counts=dict(sorted(region_group_counts.items())),
        prd_region_coverage=coverage,
        uncovered_prd_regions=[
            region.english_name
            for region in PRD_LOCALIZATION_REGION_COVERAGE
            if region.status == "missing"
        ],
        annotation_path=to_project_relative_path(annotation_path),
    )
    return coco, summary


def _attribute_ids(value: object) -> list[int]:
    """Preserve valid Fashionpedia attribute IDs for later 3.1.3 reuse."""
    if not isinstance(value, list):
        return []
    return [item for item in value if is_integer(item)]
