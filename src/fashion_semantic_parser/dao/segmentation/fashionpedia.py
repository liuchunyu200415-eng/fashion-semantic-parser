"""Prepare Fashionpedia annotations for PRD 3.1.1 segmentation."""

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
from fashion_semantic_parser.dao.segmentation.coco import coco_categories
from fashion_semantic_parser.dao.segmentation.taxonomy import (
    PRD_SEGMENTATION_CATEGORIES,
    fashionpedia_category_exclusion_reason,
    map_fashionpedia_category,
)


class FashionpediaPreparationSummary(BaseModel):
    """Audit and conversion counts for one Fashionpedia split."""

    split: str
    source_image_count: int
    source_annotation_count: int
    selected_image_count: int
    selected_annotation_count: int
    output_image_count: int
    output_annotation_count: int
    missing_image_count: int
    dropped_invalid_image_count: int
    dropped_ambiguous_image_count: int
    dropped_ambiguous_image_annotation_count: int
    dropped_empty_image_count: int
    excluded_part_annotation_count: int
    excluded_unknown_annotation_count: int
    invalid_annotation_count: int
    category_counts: dict[str, int]
    source_category_counts: dict[str, int]
    annotation_path: str
    output_path: str | None = None


def audit_fashionpedia_annotations(
    root: Path,
    split: str,
    limit: int | None = None,
) -> FashionpediaPreparationSummary:
    """Audit Fashionpedia mapping without requiring image downloads."""
    annotation_path, image_root = resolve_fashionpedia_split_paths(root, split)
    _, summary = _prepare_fashionpedia_coco(
        annotation_path=annotation_path,
        image_root=image_root,
        split=split,
        limit=limit,
    )
    return summary


def convert_fashionpedia_to_coco(
    root: Path,
    split: str,
    output_path: Path,
    limit: int | None = None,
) -> FashionpediaPreparationSummary:
    """Convert one Fashionpedia split to the project's eight-class COCO schema."""
    annotation_path, image_root = resolve_fashionpedia_split_paths(root, split)
    coco, summary = _prepare_fashionpedia_coco(
        annotation_path=annotation_path,
        image_root=image_root,
        split=split,
        limit=limit,
    )
    if summary.missing_image_count:
        raise FileNotFoundError(
            f"Fashionpedia {split} is missing {summary.missing_image_count} "
            f"selected image file(s) under {image_root}. Run --audit-only until "
            "the official image archive has been extracted."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary.model_copy(
        update={"output_path": to_project_relative_path(output_path)}
    )


def _prepare_fashionpedia_coco(
    *,
    annotation_path: Path,
    image_root: Path,
    split: str,
    limit: int | None,
) -> tuple[dict[str, object], FashionpediaPreparationSummary]:
    """Build mapped records and detailed audit counts."""
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

    source_category_counts: Counter[str] = Counter()
    for annotation in selected_annotations:
        source_category_counts[_source_category_name(annotation, category_by_id)] += 1

    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    target_category_counts: Counter[str] = Counter()
    missing_image_count = 0
    dropped_invalid_image_count = 0
    dropped_ambiguous_image_count = 0
    dropped_ambiguous_image_annotation_count = 0
    dropped_empty_image_count = 0
    excluded_part_annotation_count = 0
    excluded_unknown_annotation_count = 0
    invalid_annotation_count = 0

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

        source_rows = annotations_by_image.get(source_image_id, [])
        if _contains_ambiguous_main_apparel(source_rows, category_by_id):
            dropped_ambiguous_image_count += 1
            dropped_ambiguous_image_annotation_count += len(source_rows)
            continue

        converted_rows: list[dict[str, object]] = []
        for source_annotation in source_rows:
            source_category_id = source_annotation.get("category_id")
            category_name = _source_category_name(
                source_annotation,
                category_by_id,
            )
            exclusion_reason = fashionpedia_category_exclusion_reason(category_name)
            if exclusion_reason == "garment_part":
                excluded_part_annotation_count += 1
                continue

            category = map_fashionpedia_category(category_name)
            if category is None:
                excluded_unknown_annotation_count += 1
                continue

            segmentation = normalize_coco_segmentation(
                source_annotation.get("segmentation")
            )
            bbox = normalize_coco_bbox_xywh(source_annotation.get("bbox"))
            if segmentation is None or bbox is None:
                invalid_annotation_count += 1
                continue

            converted_rows.append(
                {
                    "category_id": category.id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": annotation_area(source_annotation.get("area"), bbox),
                    "iscrowd": int(source_annotation.get("iscrowd") == 1),
                    "source_annotation_id": source_annotation.get("id"),
                    "source_category_id": source_category_id,
                    "target_category_name": category.english_name,
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
            converted_row.update(
                {
                    "id": len(annotations) + 1,
                    "image_id": output_image_id,
                }
            )
            annotations.append(converted_row)
            target_category_counts[category_name] += 1

    coco: dict[str, object] = {
        "info": {
            "description": "Fashionpedia mapped to PRD 3.1.1 segmentation",
            "source": "https://fashionpedia.github.io/",
        },
        "images": images,
        "annotations": annotations,
        "categories": coco_categories(PRD_SEGMENTATION_CATEGORIES),
        "licenses": source.get("licenses", []),
    }
    summary = FashionpediaPreparationSummary(
        split=split,
        source_image_count=len(source_images),
        source_annotation_count=len(source_annotations),
        selected_image_count=len(selected_images),
        selected_annotation_count=len(selected_annotations),
        output_image_count=len(images),
        output_annotation_count=len(annotations),
        missing_image_count=missing_image_count,
        dropped_invalid_image_count=dropped_invalid_image_count,
        dropped_ambiguous_image_count=dropped_ambiguous_image_count,
        dropped_ambiguous_image_annotation_count=(
            dropped_ambiguous_image_annotation_count
        ),
        dropped_empty_image_count=dropped_empty_image_count,
        excluded_part_annotation_count=excluded_part_annotation_count,
        excluded_unknown_annotation_count=excluded_unknown_annotation_count,
        invalid_annotation_count=invalid_annotation_count,
        category_counts={
            category.english_name: target_category_counts[category.english_name]
            for category in PRD_SEGMENTATION_CATEGORIES
        },
        source_category_counts=dict(sorted(source_category_counts.items())),
        annotation_path=to_project_relative_path(annotation_path),
    )
    return coco, summary


def _source_category_name(
    annotation: dict[str, Any],
    category_by_id: dict[int, dict[str, Any]],
) -> str:
    """Resolve one source category name or a stable unknown marker."""
    return str(source_category_name(annotation, category_by_id))


def _contains_ambiguous_main_apparel(
    annotations: list[dict[str, Any]],
    category_by_id: dict[int, dict[str, Any]],
) -> bool:
    """Keep excluded main apparel from becoming unlabeled background."""
    return any(
        fashionpedia_category_exclusion_reason(
            _source_category_name(annotation, category_by_id)
        )
        == "ambiguous_main_apparel"
        for annotation in annotations
    )
