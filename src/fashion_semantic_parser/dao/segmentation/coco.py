"""Convert DeepFashion2 annotations to COCO instance segmentation format."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fashion_semantic_parser.common.paths import to_project_relative_path
from fashion_semantic_parser.dao.segmentation.taxonomy import (
    PRD_SEGMENTATION_CATEGORIES,
    SegmentationCategory,
    map_deepfashion2_category,
)


class COCOCategory(BaseModel):
    """COCO category record."""

    id: int
    name: str
    supercategory: str = "garment"


class COCOConversionSummary(BaseModel):
    """Summary for one DeepFashion2 to COCO conversion."""

    split: str
    image_count: int
    annotation_count: int
    skipped_item_count: int
    output_path: str


def convert_deepfashion2_to_coco(
    root: Path,
    split: str,
    output_path: Path,
    limit: int | None = None,
) -> COCOConversionSummary:
    """Convert one DeepFashion2 split to COCO instance segmentation format.

    Args:
        root: DeepFashion2 dataset root directory.
        split: Dataset split name, such as ``train`` or ``validation``.
        output_path: Output COCO JSON file path.
        limit: Optional maximum number of images for smoke tests.

    Returns:
        Conversion summary.
    """
    image_root = root / split / "image"
    annotation_root = root / split / "annos"
    image_paths = sorted(image_root.glob("*.jpg")) if image_root.exists() else []
    if limit is not None:
        image_paths = image_paths[:limit]

    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    skipped_item_count = 0
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        annotation_path = annotation_root / f"{image_path.stem}.json"
        images.append(
            {
                "id": image_id,
                "file_name": to_project_relative_path(image_path),
            }
        )
        if not annotation_path.exists():
            continue

        raw_annotation = _read_json(annotation_path)
        for item in _iter_raw_items(raw_annotation):
            category = _mapped_category(item)
            segmentation = _segmentation(item)
            bbox = _bbox_xywh(item)
            if category is None or not segmentation or bbox is None:
                skipped_item_count += 1
                continue

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category.id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": _bbox_area(bbox),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": _coco_categories(PRD_SEGMENTATION_CATEGORIES),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return COCOConversionSummary(
        split=split,
        image_count=len(images),
        annotation_count=len(annotations),
        skipped_item_count=skipped_item_count,
        output_path=to_project_relative_path(output_path),
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)
    return data


def _iter_raw_items(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return garment item dictionaries from a DeepFashion2 annotation."""
    items: list[dict[str, Any]] = []
    for key in sorted(annotation):
        value = annotation[key]
        if key.startswith("item") and isinstance(value, dict):
            items.append(value)
    return items


def _mapped_category(item: dict[str, Any]) -> SegmentationCategory | None:
    """Map one raw item to the PRD segmentation taxonomy."""
    category_name = item.get("category_name")
    if not isinstance(category_name, str):
        return None
    return map_deepfashion2_category(category_name)


def _segmentation(item: dict[str, Any]) -> list[list[float]]:
    """Read polygon segmentation from one raw item."""
    segmentation = item.get("segmentation", [])
    if not isinstance(segmentation, list):
        return []

    polygons: list[list[float]] = []
    for polygon in segmentation:
        if not isinstance(polygon, list):
            continue
        points = [point for point in polygon if isinstance(point, int | float)]
        if len(points) >= 6:
            polygons.append(points)
    return polygons


def _bbox_xywh(item: dict[str, Any]) -> list[float] | None:
    """Convert a DeepFashion2 bounding box to COCO xywh format."""
    bounding_box = item.get("bounding_box")
    if not isinstance(bounding_box, list) or len(bounding_box) != 4:
        return None

    values = [value for value in bounding_box if isinstance(value, int | float)]
    if len(values) != 4:
        return None

    x_min, y_min, x_max, y_max = values
    width = max(float(x_max) - float(x_min), 0.0)
    height = max(float(y_max) - float(y_min), 0.0)
    if width <= 0 or height <= 0:
        return None
    return [float(x_min), float(y_min), width, height]


def _bbox_area(bbox: list[float]) -> float:
    """Compute COCO bbox area."""
    return bbox[2] * bbox[3]


def _coco_categories(
    categories: list[SegmentationCategory],
) -> list[dict[str, object]]:
    """Convert PRD categories to COCO categories."""
    return [
        COCOCategory(id=category.id, name=category.english_name).model_dump()
        for category in categories
    ]
