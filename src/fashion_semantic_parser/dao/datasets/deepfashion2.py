"""DeepFashion2 dataset inspection utilities."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


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
