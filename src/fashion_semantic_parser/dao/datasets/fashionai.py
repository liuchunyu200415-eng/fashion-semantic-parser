"""FashionAI attribute dataset inspection utilities."""

from pathlib import Path

from pydantic import BaseModel, Field


class FashionAIAttributeSplit(BaseModel):
    """Summary of one FashionAI attribute image directory."""

    name: str
    image_count: int
    sample_image: str | None = None


class FashionAISummary(BaseModel):
    """Summary of a FashionAI attribute dataset directory."""

    root: str
    exists: bool
    attribute_splits: list[FashionAIAttributeSplit] = Field(default_factory=list)
    test_file_count: int = 0
    test_files: list[str] = Field(default_factory=list)


def inspect_fashionai_dataset(root: Path) -> FashionAISummary:
    """Inspect the FashionAI attribute dataset without loading image bytes.

    Args:
        root: Dataset root directory.

    Returns:
        Dataset summary with split counts and available test files.
    """
    if not root.exists():
        return FashionAISummary(root=str(root), exists=False)

    image_root = root / "Images"
    test_root = root / "Tests"
    attribute_splits = _inspect_attribute_splits(image_root)
    test_files = sorted(path.name for path in test_root.glob("*") if path.is_file())

    return FashionAISummary(
        root=str(root),
        exists=True,
        attribute_splits=attribute_splits,
        test_file_count=len(test_files),
        test_files=test_files,
    )


def _inspect_attribute_splits(image_root: Path) -> list[FashionAIAttributeSplit]:
    """Inspect FashionAI attribute image subdirectories."""
    if not image_root.exists():
        return []

    splits: list[FashionAIAttributeSplit] = []
    for split_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        images = sorted(split_dir.glob("*.jpg"))
        sample_image = images[0].name if images else None
        splits.append(
            FashionAIAttributeSplit(
                name=split_dir.name,
                image_count=len(images),
                sample_image=sample_image,
            )
        )
    return splits
