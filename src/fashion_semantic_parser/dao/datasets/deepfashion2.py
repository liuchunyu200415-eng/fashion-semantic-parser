"""DeepFashion2 dataset inspection utilities."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fashion_semantic_parser.models.datasets import (
    FashionItemAnnotation,
    FashionSample,
)


class DeepFashion2SplitSummary(BaseModel):
    """Summary of one DeepFashion2 split directory."""

    name: str
    image_count: int = 0
    annotation_count: int = 0
    sample_image: str | None = None
    sample_annotation: str | None = None
    sample_item_keys: list[str] = Field(default_factory=list)


class DeepFashion2Summary(BaseModel):
    """Summary of a DeepFashion2 dataset directory."""

    root: str
    exists: bool
    splits: list[DeepFashion2SplitSummary] = Field(default_factory=list)
    json_directories: list[str] = Field(default_factory=list)


def inspect_deepfashion2_dataset(root: Path) -> DeepFashion2Summary:
    """Inspect the DeepFashion2 dataset without loading image bytes.

    Args:
        root: Dataset root directory.

    Returns:
        Dataset summary with split counts and JSON metadata directories.
    """
    if not root.exists():
        return DeepFashion2Summary(root=str(root), exists=False)

    splits = [
        _inspect_split(root / split_name)
        for split_name in ("train", "validation", "test")
        if (root / split_name).exists()
    ]
    json_directories = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("json_for_")
    )

    return DeepFashion2Summary(
        root=str(root),
        exists=True,
        splits=splits,
        json_directories=json_directories,
    )


def _inspect_split(split_root: Path) -> DeepFashion2SplitSummary:
    """Inspect one DeepFashion2 split directory."""
    image_root = split_root / "image"
    annotation_root = split_root / "annos"
    images = sorted(image_root.glob("*.jpg")) if image_root.exists() else []
    annotations = (
        sorted(annotation_root.glob("*.json")) if annotation_root.exists() else []
    )
    sample_keys = _read_sample_item_keys(annotations[0]) if annotations else []

    return DeepFashion2SplitSummary(
        name=split_root.name,
        image_count=len(images),
        annotation_count=len(annotations),
        sample_image=images[0].name if images else None,
        sample_annotation=annotations[0].name if annotations else None,
        sample_item_keys=sample_keys,
    )


def _read_sample_item_keys(annotation_path: Path) -> list[str]:
    """Read representative item keys from a DeepFashion2 annotation file."""
    with annotation_path.open("r", encoding="utf-8") as file:
        annotation: dict[str, Any] = json.load(file)

    item_keys = sorted(key for key in annotation if key.startswith("item"))
    return item_keys[:5]


def load_deepfashion2_samples(
    root: Path,
    split: str,
    limit: int | None = None,
) -> list[FashionSample]:
    """Load DeepFashion2 split metadata into normalized samples.

    Args:
        root: DeepFashion2 dataset root directory.
        split: Dataset split name, such as ``train``, ``validation``, or ``test``.
        limit: Optional maximum number of samples to return.

    Returns:
        Normalized image samples with item annotations when available.
    """
    split_root = root / split
    image_root = split_root / "image"
    annotation_root = split_root / "annos"
    if not image_root.exists():
        return []

    image_paths = sorted(image_root.glob("*.jpg"))
    if limit is not None:
        image_paths = image_paths[:limit]

    samples: list[FashionSample] = []
    for image_path in image_paths:
        annotation_path = annotation_root / f"{image_path.stem}.json"
        items: list[FashionItemAnnotation] = []
        metadata: dict[str, Any] = {}
        sample_annotation_path: str | None = None

        if annotation_path.exists():
            annotation = _read_annotation(annotation_path)
            items = _parse_item_annotations(annotation)
            metadata = _parse_sample_metadata(annotation)
            sample_annotation_path = str(annotation_path)

        samples.append(
            FashionSample(
                dataset_name="deepfashion2",
                split=split,
                image_path=str(image_path),
                annotation_path=sample_annotation_path,
                items=items,
                metadata=metadata,
            )
        )

    return samples


def _read_annotation(annotation_path: Path) -> dict[str, Any]:
    """Read one DeepFashion2 annotation file."""
    with annotation_path.open("r", encoding="utf-8") as file:
        annotation: dict[str, Any] = json.load(file)
    return annotation


def _parse_item_annotations(
    annotation: dict[str, Any],
) -> list[FashionItemAnnotation]:
    """Parse garment item entries from one DeepFashion2 annotation."""
    items: list[FashionItemAnnotation] = []
    item_keys = sorted(key for key in annotation if key.startswith("item"))
    for item_key in item_keys:
        raw_item = annotation.get(item_key)
        if not isinstance(raw_item, dict):
            continue

        items.append(
            FashionItemAnnotation(
                item_id=item_key,
                category_name=_optional_str(raw_item.get("category_name")),
                category_id=_optional_int(raw_item.get("category_id")),
                style=_optional_int(raw_item.get("style")),
                bounding_box=_int_list(raw_item.get("bounding_box")),
                raw_attributes=raw_item,
            )
        )
    return items


def _parse_sample_metadata(annotation: dict[str, Any]) -> dict[str, Any]:
    """Keep non-item DeepFashion2 fields as sample metadata."""
    return {
        key: value for key, value in annotation.items() if not key.startswith("item")
    }


def _optional_str(value: Any) -> str | None:
    """Return a string value when available."""
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    """Return an integer value when available."""
    return value if isinstance(value, int) else None


def _int_list(value: Any) -> list[int]:
    """Return a list of integers when the raw field is list-like."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int)]
