"""Shared helpers for reading official Fashionpedia COCO annotations."""

import json
from pathlib import Path
from typing import Any, TypeGuard

FASHIONPEDIA_SPLIT_LAYOUT: dict[str, tuple[str, tuple[str, ...]]] = {
    "train": ("instances_attributes_train2020.json", ("train", "train2020")),
    "validation": (
        "instances_attributes_val2020.json",
        ("test", "val", "val2020"),
    ),
}


def resolve_fashionpedia_split_paths(root: Path, split: str) -> tuple[Path, Path]:
    """Resolve official Fashionpedia annotation and image directories."""
    try:
        annotation_name, image_directories = FASHIONPEDIA_SPLIT_LAYOUT[split]
    except KeyError as error:
        supported = ", ".join(sorted(FASHIONPEDIA_SPLIT_LAYOUT))
        raise ValueError(
            f"Unsupported Fashionpedia split {split!r}; expected one of {supported}"
        ) from error
    image_root = next(
        (
            root / image_directory
            for image_directory in image_directories
            if (root / image_directory).is_dir()
        ),
        root / image_directories[0],
    )
    return root / "annotations" / annotation_name, image_root


def read_fashionpedia_json(path: Path) -> dict[str, Any]:
    """Read one official Fashionpedia annotation object."""
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def dict_records(value: object) -> list[dict[str, Any]]:
    """Return only object records from an external JSON list."""
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]


def category_records_by_id(
    source_categories: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Index valid Fashionpedia category records by integer ID."""
    return {
        category["id"]: category
        for category in source_categories
        if is_integer(category.get("id")) and isinstance(category.get("name"), str)
    }


def is_integer(value: object) -> TypeGuard[int]:
    """Return whether a value is an integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_positive_number(value: object) -> TypeGuard[int | float]:
    """Return whether a value is a positive finite-size number."""
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def image_sort_key(image: dict[str, Any]) -> tuple[str, str]:
    """Sort source images deterministically for limited smoke tests."""
    return str(image.get("file_name", "")), str(image.get("id", ""))


def source_category_name(
    annotation: dict[str, Any],
    category_by_id: dict[int, dict[str, Any]],
) -> str:
    """Resolve one source category name or a stable unknown marker."""
    category_id = annotation.get("category_id")
    category = category_by_id.get(category_id) if is_integer(category_id) else None
    category_name = category.get("name") if category is not None else None
    if isinstance(category_name, str):
        return category_name
    return f"<unknown:{category_id}>"


def safe_image_path(image_root: Path, file_name: str) -> Path:
    """Resolve a source image path without allowing traversal."""
    relative_path = Path(file_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe Fashionpedia image path: {file_name}")
    return image_root / relative_path


def normalize_coco_segmentation(
    value: object,
) -> list[list[float]] | dict[str, object] | None:
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
            and all(is_positive_number(dimension) for dimension in size)
            and isinstance(counts, str | list)
        ):
            return {"size": [int(size[0]), int(size[1])], "counts": counts}
    return None


def normalize_coco_bbox_xywh(value: object) -> list[float] | None:
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


def annotation_area(value: object, bbox: list[float]) -> float:
    """Prefer official mask area and fall back to bounding-box area."""
    if is_positive_number(value):
        return float(value)
    return bbox[2] * bbox[3]
