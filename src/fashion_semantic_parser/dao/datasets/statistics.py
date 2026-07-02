"""Compute statistics from generated dataset indexes."""

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from fashion_semantic_parser.dao.datasets.index_reader import DatasetIndexReader


class DeepFashion2SplitStatistics(BaseModel):
    """Statistics for one DeepFashion2 split index."""

    index_name: str
    sample_count: int = 0
    item_count: int = 0
    category_counts: dict[str, int] = Field(default_factory=dict)
    source_counts: dict[str, int] = Field(default_factory=dict)


class FashionAIStatistics(BaseModel):
    """Statistics for FashionAI index files."""

    attribute_sample_count: int = 0
    question_count: int = 0
    attribute_group_counts: dict[str, int] = Field(default_factory=dict)


class DatasetStatistics(BaseModel):
    """Combined statistics for all generated dataset indexes."""

    record_counts: dict[str, int] = Field(default_factory=dict)
    fashionai: FashionAIStatistics = Field(default_factory=FashionAIStatistics)
    deepfashion2: dict[str, DeepFashion2SplitStatistics] = Field(default_factory=dict)


def compute_dataset_statistics(reader: DatasetIndexReader) -> DatasetStatistics:
    """Compute project dataset statistics from generated indexes.

    Args:
        reader: Reader for generated dataset indexes.

    Returns:
        Combined statistics across FashionAI and DeepFashion2 indexes.
    """
    return DatasetStatistics(
        record_counts=reader.record_counts(),
        fashionai=_compute_fashionai_statistics(reader),
        deepfashion2={
            split: _compute_deepfashion2_split_statistics(
                reader,
                f"deepfashion2_{split}",
            )
            for split in ("train", "validation", "test")
            if f"deepfashion2_{split}" in reader.list_indexes()
        },
    )


def _compute_fashionai_statistics(
    reader: DatasetIndexReader,
) -> FashionAIStatistics:
    """Compute FashionAI attribute and question statistics."""
    attribute_group_counts: Counter[str] = Counter()
    attribute_sample_count = 0
    question_count = 0

    if "fashionai_attributes" in reader.list_indexes():
        for record in reader.iter_records("fashionai_attributes"):
            attribute_sample_count += 1
            attribute_group = _nested_str(record, "attributes", "attribute_group")
            if attribute_group is not None:
                attribute_group_counts[attribute_group] += 1

    if "fashionai_questions" in reader.list_indexes():
        for _record in reader.iter_records("fashionai_questions"):
            question_count += 1

    return FashionAIStatistics(
        attribute_sample_count=attribute_sample_count,
        question_count=question_count,
        attribute_group_counts=dict(attribute_group_counts),
    )


def _compute_deepfashion2_split_statistics(
    reader: DatasetIndexReader,
    index_name: str,
) -> DeepFashion2SplitStatistics:
    """Compute statistics for one DeepFashion2 split index."""
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    sample_count = 0
    item_count = 0

    for record in reader.iter_records(index_name):
        sample_count += 1
        source = _nested_str(record, "metadata", "source")
        if source is not None:
            source_counts[source] += 1

        items = record.get("items", [])
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            item_count += 1
            category_name = item.get("category_name")
            if isinstance(category_name, str):
                category_counts[category_name] += 1

    return DeepFashion2SplitStatistics(
        index_name=index_name,
        sample_count=sample_count,
        item_count=item_count,
        category_counts=dict(category_counts),
        source_counts=dict(source_counts),
    )


def _nested_str(record: dict[str, Any], outer_key: str, inner_key: str) -> str | None:
    """Read a nested string value from a record dictionary."""
    outer_value = record.get(outer_key, {})
    if not isinstance(outer_value, dict):
        return None
    inner_value = outer_value.get(inner_key)
    return inner_value if isinstance(inner_value, str) else None
