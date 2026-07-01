"""FashionAI attribute dataset inspection utilities."""

import csv
from pathlib import Path

from pydantic import BaseModel, Field

from fashion_semantic_parser.models.datasets import FashionSample


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


class FashionAIQuestion(BaseModel):
    """One row from a FashionAI test CSV file."""

    row_index: int
    fields: dict[str, str]


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


def load_fashionai_attribute_samples(
    root: Path,
    limit: int | None = None,
) -> list[FashionSample]:
    """Load FashionAI attribute images into normalized samples.

    Args:
        root: FashionAI dataset root directory.
        limit: Optional maximum number of samples to return.

    Returns:
        Normalized samples with the attribute directory stored as metadata.
    """
    image_root = root / "Images"
    if not image_root.exists():
        return []

    samples: list[FashionSample] = []
    for attribute_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        for image_path in sorted(attribute_dir.glob("*.jpg")):
            samples.append(
                FashionSample(
                    dataset_name="fashionai",
                    split="test",
                    image_path=str(image_path),
                    attributes={"attribute_group": attribute_dir.name},
                    metadata={"attribute_group": attribute_dir.name},
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def load_fashionai_questions(
    root: Path,
    file_name: str = "question.csv",
    limit: int | None = None,
) -> list[FashionAIQuestion]:
    """Load a FashionAI CSV file while preserving its original fields.

    Args:
        root: FashionAI dataset root directory.
        file_name: CSV file name under the ``Tests`` directory.
        limit: Optional maximum number of rows to return.

    Returns:
        CSV rows represented as typed question records.
    """
    csv_path = root / "Tests" / file_name
    if not csv_path.exists():
        return []

    questions: list[FashionAIQuestion] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader):
            fields = {
                key: value
                for key, value in row.items()
                if key is not None and value is not None
            }
            questions.append(FashionAIQuestion(row_index=row_index, fields=fields))
            if limit is not None and len(questions) >= limit:
                return questions

    return questions
