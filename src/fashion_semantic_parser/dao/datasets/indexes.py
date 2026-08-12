"""Build lightweight JSONL indexes for raw fashion datasets."""

import json
from pathlib import Path
from typing import Iterable, cast

from pydantic import BaseModel, Field

from fashion_semantic_parser.common.paths import to_project_relative_path
from fashion_semantic_parser.dao.datasets.deepfashion2 import (
    iter_deepfashion2_samples,
)
from fashion_semantic_parser.dao.datasets.fashionai import (
    FashionAIQuestion,
    iter_fashionai_attribute_samples,
    iter_fashionai_questions,
)
from fashion_semantic_parser.models.datasets import (
    FashionItemAnnotation,
    FashionSample,
)


class DatasetIndexFile(BaseModel):
    """Metadata for one generated dataset index file."""

    name: str
    path: str
    record_count: int


class DatasetIndexManifest(BaseModel):
    """Manifest describing generated dataset index files."""

    output_dir: str
    files: list[DatasetIndexFile] = Field(default_factory=list)


def build_dataset_indexes(
    fashionai_root: Path,
    deepfashion2_root: Path,
    output_dir: Path,
    limit: int | None = None,
) -> DatasetIndexManifest:
    """Build JSONL indexes for configured raw datasets.

    Args:
        fashionai_root: FashionAI dataset root directory.
        deepfashion2_root: DeepFashion2 dataset root directory.
        output_dir: Directory for generated JSONL index files.
        limit: Optional maximum records per source, mainly for smoke tests.

    Returns:
        Manifest with generated file paths and record counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [
        _write_fashionai_attribute_index(fashionai_root, output_dir, limit),
        _write_fashionai_question_index(fashionai_root, output_dir, limit),
    ]
    for split in ("train", "validation", "test"):
        files.append(
            _write_deepfashion2_split_index(
                deepfashion2_root,
                output_dir,
                split,
                limit,
            )
        )

    manifest = DatasetIndexManifest(
        output_dir=to_project_relative_path(output_dir),
        files=files,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _write_fashionai_attribute_index(
    fashionai_root: Path,
    output_dir: Path,
    limit: int | None,
) -> DatasetIndexFile:
    """Write the FashionAI attribute image index."""
    output_path = output_dir / "fashionai_attributes.jsonl"
    samples = iter_fashionai_attribute_samples(fashionai_root, limit=limit)
    records = (_sample_to_index_record(sample) for sample in samples)
    record_count = _write_jsonl(records, output_path)
    return _index_file("fashionai_attributes", output_path, record_count)


def _write_fashionai_question_index(
    fashionai_root: Path,
    output_dir: Path,
    limit: int | None,
) -> DatasetIndexFile:
    """Write the FashionAI question CSV index."""
    output_path = output_dir / "fashionai_questions.jsonl"
    questions = iter_fashionai_questions(fashionai_root, limit=limit)
    records = (_question_to_index_record(question) for question in questions)
    record_count = _write_jsonl(records, output_path)
    return _index_file("fashionai_questions", output_path, record_count)


def _write_deepfashion2_split_index(
    deepfashion2_root: Path,
    output_dir: Path,
    split: str,
    limit: int | None,
) -> DatasetIndexFile:
    """Write one DeepFashion2 split index."""
    output_path = output_dir / f"deepfashion2_{split}.jsonl"
    samples = iter_deepfashion2_samples(deepfashion2_root, split=split, limit=limit)
    records = (_sample_to_index_record(sample) for sample in samples)
    record_count = _write_jsonl(records, output_path)
    return _index_file(f"deepfashion2_{split}", output_path, record_count)


def _write_jsonl(records: Iterable[dict[str, object]], output_path: Path) -> int:
    """Write records as JSON Lines.

    Args:
        records: JSON-serializable record dictionaries.
        output_path: Output JSONL file path.

    Returns:
        Number of records written.
    """
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")
            count += 1
    tmp_path.replace(output_path)
    return count


def _sample_to_index_record(sample: FashionSample) -> dict[str, object]:
    """Convert a normalized sample to a portable index record."""
    record: dict[str, object] = {
        "dataset_name": sample.dataset_name,
        "split": sample.split,
        "image_path": to_project_relative_path(sample.image_path),
        "items": [_item_to_index_record(item) for item in sample.items],
        "attributes": sample.attributes,
        "metadata": sample.metadata,
    }
    if sample.annotation_path is not None:
        record["annotation_path"] = to_project_relative_path(sample.annotation_path)
    return record


def _item_to_index_record(item: FashionItemAnnotation) -> dict[str, object]:
    """Convert an item annotation to compact index fields."""
    return {
        "item_id": item.item_id,
        "category_name": item.category_name,
        "category_id": item.category_id,
        "style": item.style,
        "bounding_box": item.bounding_box,
    }


def _question_to_index_record(question: FashionAIQuestion) -> dict[str, object]:
    """Convert a FashionAI question row to an index record."""
    return cast(dict[str, object], question.model_dump())


def _index_file(
    name: str,
    output_path: Path,
    record_count: int,
) -> DatasetIndexFile:
    """Create metadata for one generated index file."""
    return DatasetIndexFile(
        name=name,
        path=to_project_relative_path(output_path),
        record_count=record_count,
    )
