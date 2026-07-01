"""Tests for dataset inspection utilities."""

import json
from pathlib import Path

from fashion_semantic_parser.common.paths import (
    PROJECT_ROOT,
    to_project_relative_path,
)
from fashion_semantic_parser.dao.datasets.deepfashion2 import (
    inspect_deepfashion2_dataset,
    load_deepfashion2_samples,
)
from fashion_semantic_parser.dao.datasets.fashionai import (
    inspect_fashionai_dataset,
    load_fashionai_attribute_samples,
    load_fashionai_questions,
)
from fashion_semantic_parser.dao.datasets.indexes import build_dataset_indexes


def test_inspect_fashionai_dataset_counts_splits(tmp_path: Path) -> None:
    """FashionAI inspection should count attribute images and test files."""
    root = tmp_path / "fashionai"
    image_split = root / "Images" / "sleeve_length_labels"
    tests_root = root / "Tests"
    image_split.mkdir(parents=True)
    tests_root.mkdir(parents=True)
    (image_split / "sample.jpg").write_bytes(b"fake-image")
    (tests_root / "question.csv").write_text("image_id,label\n", encoding="utf-8")

    summary = inspect_fashionai_dataset(root)

    assert summary.exists is True
    assert summary.attribute_splits[0].name == "sleeve_length_labels"
    assert summary.attribute_splits[0].image_count == 1
    assert summary.test_files == ["question.csv"]


def test_inspect_deepfashion2_dataset_counts_split_files(tmp_path: Path) -> None:
    """DeepFashion2 inspection should count split images and annotations."""
    root = tmp_path / "deepfashion2"
    image_root = root / "train" / "image"
    annotation_root = root / "train" / "annos"
    json_root = root / "json_for_validation"
    image_root.mkdir(parents=True)
    annotation_root.mkdir(parents=True)
    json_root.mkdir(parents=True)
    (image_root / "000001.jpg").write_bytes(b"fake-image")
    annotation = {"item1": {"category_name": "short sleeve top"}}
    (annotation_root / "000001.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    summary = inspect_deepfashion2_dataset(root)

    assert summary.exists is True
    assert summary.json_directories == ["json_for_validation"]
    assert summary.splits[0].name == "train"
    assert summary.splits[0].image_count == 1
    assert summary.splits[0].annotation_count == 1
    assert summary.splits[0].sample_item_keys == ["item1"]


def test_load_fashionai_attribute_samples_and_questions(tmp_path: Path) -> None:
    """FashionAI loader should normalize images and preserve CSV fields."""
    root = tmp_path / "fashionai"
    image_split = root / "Images" / "sleeve_length_labels"
    tests_root = root / "Tests"
    image_split.mkdir(parents=True)
    tests_root.mkdir(parents=True)
    (image_split / "sample.jpg").write_bytes(b"fake-image")
    (tests_root / "question.csv").write_text(
        "image_id,attribute_name\nsample.jpg,sleeve_length\n",
        encoding="utf-8",
    )

    samples = load_fashionai_attribute_samples(root)
    questions = load_fashionai_questions(root)

    assert len(samples) == 1
    assert samples[0].dataset_name == "fashionai"
    assert samples[0].attributes["attribute_group"] == "sleeve_length_labels"
    assert len(questions) == 1
    assert questions[0].fields["image_id"] == "sample.jpg"


def test_load_deepfashion2_samples_parses_item_annotations(
    tmp_path: Path,
) -> None:
    """DeepFashion2 loader should normalize image and item annotations."""
    root = tmp_path / "deepfashion2"
    image_root = root / "train" / "image"
    annotation_root = root / "train" / "annos"
    image_root.mkdir(parents=True)
    annotation_root.mkdir(parents=True)
    (image_root / "000001.jpg").write_bytes(b"fake-image")
    annotation = {
        "source": "shop",
        "pair_id": 1,
        "item1": {
            "category_name": "short sleeve top",
            "category_id": 1,
            "style": 2,
            "bounding_box": [1, 2, 10, 20],
        },
    }
    (annotation_root / "000001.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    samples = load_deepfashion2_samples(root, split="train")

    assert len(samples) == 1
    assert samples[0].dataset_name == "deepfashion2"
    assert samples[0].metadata["source"] == "shop"
    assert samples[0].items[0].category_name == "short sleeve top"
    assert samples[0].items[0].bounding_box == [1, 2, 10, 20]


def test_to_project_relative_path_converts_project_paths() -> None:
    """Project paths should be serialized as portable relative paths."""
    path = PROJECT_ROOT / "data" / "raw" / "sample.jpg"

    assert to_project_relative_path(path) == "data/raw/sample.jpg"


def test_build_dataset_indexes_writes_jsonl_files(tmp_path: Path) -> None:
    """Index builder should write portable JSONL files and a manifest."""
    fashionai_root = tmp_path / "fashionai"
    fashionai_image_root = fashionai_root / "Images" / "coat_length_labels"
    fashionai_tests_root = fashionai_root / "Tests"
    deepfashion2_root = tmp_path / "deepfashion2"
    deepfashion2_image_root = deepfashion2_root / "train" / "image"
    deepfashion2_annotation_root = deepfashion2_root / "train" / "annos"
    output_dir = tmp_path / "processed" / "indexes"
    fashionai_image_root.mkdir(parents=True)
    fashionai_tests_root.mkdir(parents=True)
    deepfashion2_image_root.mkdir(parents=True)
    deepfashion2_annotation_root.mkdir(parents=True)
    (fashionai_image_root / "sample.jpg").write_bytes(b"fake-image")
    (fashionai_tests_root / "question.csv").write_text(
        "image_id,attribute_name\nsample.jpg,coat_length\n",
        encoding="utf-8",
    )
    (deepfashion2_image_root / "000001.jpg").write_bytes(b"fake-image")
    (deepfashion2_annotation_root / "000001.json").write_text(
        json.dumps({"item1": {"category_name": "trousers"}}),
        encoding="utf-8",
    )

    manifest = build_dataset_indexes(
        fashionai_root=fashionai_root,
        deepfashion2_root=deepfashion2_root,
        output_dir=output_dir,
    )

    record_counts = {
        index_file.name: index_file.record_count for index_file in manifest.files
    }
    assert record_counts["fashionai_attributes"] == 1
    assert record_counts["fashionai_questions"] == 1
    assert record_counts["deepfashion2_train"] == 1
    assert record_counts["deepfashion2_validation"] == 0
    assert record_counts["deepfashion2_test"] == 0
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "fashionai_attributes.jsonl").read_text(
        encoding="utf-8"
    ).count("\n") == 1
