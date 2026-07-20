"""Prepare Fashionpedia annotations for PRD 3.1.1 segmentation."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypeGuard

from pydantic import BaseModel

from fashion_semantic_parser.common.paths import to_project_relative_path
from fashion_semantic_parser.dao.segmentation.coco import coco_categories
from fashion_semantic_parser.dao.segmentation.taxonomy import (
    PRD_SEGMENTATION_CATEGORIES,
    fashionpedia_category_exclusion_reason,
    map_fashionpedia_category,
)

_SPLIT_LAYOUT = {
    "train": ("instances_attributes_train2020.json", "train2020"),
    "validation": ("instances_attributes_val2020.json", "val2020"),
}


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
    annotation_path, image_root = _split_paths(root, split)
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
    annotation_path, image_root = _split_paths(root, split)
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


def _split_paths(root: Path, split: str) -> tuple[Path, Path]:
    """Resolve official Fashionpedia annotation and image directories."""
    try:
        annotation_name, image_directory = _SPLIT_LAYOUT[split]
    except KeyError as error:
        supported = ", ".join(sorted(_SPLIT_LAYOUT))
        raise ValueError(
            f"Unsupported Fashionpedia split {split!r}; expected one of {supported}"
        ) from error
    return root / "annotations" / annotation_name, root / image_directory


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

    source = _read_json(annotation_path)
    source_images = _dict_records(source.get("images"))
    source_annotations = _dict_records(source.get("annotations"))
    source_categories = _dict_records(source.get("categories"))
    category_by_id = {
        category["id"]: category
        for category in source_categories
        if _integer(category.get("id")) and isinstance(category.get("name"), str)
    }

    selected_images = sorted(source_images, key=_image_sort_key)
    if limit is not None:
        selected_images = selected_images[:limit]
    selected_image_ids = {
        image["id"] for image in selected_images if _integer(image.get("id"))
    }
    selected_annotations = [
        annotation
        for annotation in source_annotations
        if annotation.get("image_id") in selected_image_ids
    ]
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in selected_annotations:
        image_id = annotation.get("image_id")
        if _integer(image_id):
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
            _integer(source_image_id)
            and isinstance(file_name, str)
            and _positive_number(width)
            and _positive_number(height)
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

            segmentation = _segmentation(source_annotation.get("segmentation"))
            bbox = _bbox_xywh(source_annotation.get("bbox"))
            if segmentation is None or bbox is None:
                invalid_annotation_count += 1
                continue

            converted_rows.append(
                {
                    "category_id": category.id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": _annotation_area(source_annotation.get("area"), bbox),
                    "iscrowd": int(source_annotation.get("iscrowd") == 1),
                    "source_annotation_id": source_annotation.get("id"),
                    "source_category_id": source_category_id,
                    "target_category_name": category.english_name,
                }
            )

        if not converted_rows:
            dropped_empty_image_count += 1
            continue

        image_path = _safe_image_path(image_root, file_name)
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


def _read_json(path: Path) -> dict[str, Any]:
    """Read one official Fashionpedia annotation object."""
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _dict_records(value: object) -> list[dict[str, Any]]:
    """Return only object records from an external JSON list."""
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]


def _integer(value: object) -> TypeGuard[int]:
    """Return whether a value is an integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_number(value: object) -> TypeGuard[int | float]:
    """Return whether a value is a positive finite-size number."""
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _image_sort_key(image: dict[str, Any]) -> tuple[str, str]:
    """Sort source images deterministically for limited smoke tests."""
    return str(image.get("file_name", "")), str(image.get("id", ""))


def _source_category_name(
    annotation: dict[str, Any],
    category_by_id: dict[int, dict[str, Any]],
) -> str:
    """Resolve one source category name or a stable unknown marker."""
    category_id = annotation.get("category_id")
    category = category_by_id.get(category_id) if _integer(category_id) else None
    category_name = category.get("name") if category is not None else None
    if isinstance(category_name, str):
        return category_name
    return f"<unknown:{category_id}>"


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


def _safe_image_path(image_root: Path, file_name: str) -> Path:
    """Resolve a source image path without allowing traversal."""
    relative_path = Path(file_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe Fashionpedia image path: {file_name}")
    return image_root / relative_path


def _segmentation(value: object) -> list[list[float]] | dict[str, object] | None:
    """Validate COCO polygon or RLE segmentation data."""
    if isinstance(value, list):
        polygons: list[list[float]] = []
        for polygon in value:
            if not isinstance(polygon, list):
                continue
            if len(polygon) < 6 or len(polygon) % 2 != 0:
                continue
            if not all(
                isinstance(point, int | float) and not isinstance(point, bool)
                for point in polygon
            ):
                continue
            polygons.append([float(point) for point in polygon])
        return polygons or None

    if isinstance(value, dict):
        size = value.get("size")
        counts = value.get("counts")
        if (
            isinstance(size, list)
            and len(size) == 2
            and all(_positive_number(dimension) for dimension in size)
            and isinstance(counts, str | list)
        ):
            return {"size": [int(size[0]), int(size[1])], "counts": counts}
    return None


def _bbox_xywh(value: object) -> list[float] | None:
    """Validate an existing COCO xywh bounding box."""
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(
        isinstance(coordinate, int | float) and not isinstance(coordinate, bool)
        for coordinate in value
    ):
        return None
    x, y, width, height = (float(coordinate) for coordinate in value)
    if width <= 0.0 or height <= 0.0:
        return None
    return [x, y, width, height]


def _annotation_area(value: object, bbox: list[float]) -> float:
    """Prefer official mask area and fall back to bounding-box area."""
    if _positive_number(value):
        return float(value)
    return bbox[2] * bbox[3]
