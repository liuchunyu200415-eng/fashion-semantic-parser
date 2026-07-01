"""Combined dataset inspection helpers."""

from pathlib import Path

from pydantic import BaseModel

from fashion_semantic_parser.dao.datasets.deepfashion2 import (
    DeepFashion2Summary,
    inspect_deepfashion2_dataset,
)
from fashion_semantic_parser.dao.datasets.fashionai import (
    FashionAISummary,
    inspect_fashionai_dataset,
)


class DatasetSummary(BaseModel):
    """Combined project dataset summary."""

    fashionai: FashionAISummary
    deepfashion2: DeepFashion2Summary


def inspect_project_datasets(
    fashionai_root: Path,
    deepfashion2_root: Path,
) -> DatasetSummary:
    """Inspect all configured project datasets.

    Args:
        fashionai_root: FashionAI dataset root directory.
        deepfashion2_root: DeepFashion2 dataset root directory.

    Returns:
        Combined dataset summary.
    """
    return DatasetSummary(
        fashionai=inspect_fashionai_dataset(fashionai_root),
        deepfashion2=inspect_deepfashion2_dataset(deepfashion2_root),
    )
