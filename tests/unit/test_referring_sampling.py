"""Tests for deterministic balanced referring-expression sampling."""

import json
from pathlib import Path

import pytest

from fashion_semantic_parser.dao.localization.referring_sampling import (
    build_balanced_referring_subset,
)


def test_balanced_subset_water_fills_strata_and_restores_source_order(
    tmp_path: Path,
) -> None:
    """Scarce strata survive and selected rows retain image-complete ordering."""
    index_path = tmp_path / "input.jsonl"
    rows = [
        _sample(index, image_id=index // 2, label="sleeve", language="en")
        for index in range(6)
    ]
    rows.extend(
        [
            _sample(6, image_id=3, label="rivet", language="zh"),
            _sample(7, image_id=3, label="rivet", language="zh"),
        ]
    )
    _write_jsonl(index_path, rows)
    output_path = tmp_path / "balanced.jsonl"

    summary = build_balanced_referring_subset(
        index_path=index_path,
        output_path=output_path,
        summary_output_path=tmp_path / "summary.json",
        sample_count=4,
        seed="test-seed",
    )
    selected = _read_jsonl(output_path)

    assert summary.output_sample_count == 4
    assert summary.category_counts == {"rivet": 2, "sleeve": 2}
    assert summary.weak_part_counts["rivet"] == 2
    assert [row["source_image_id"] for row in selected] == sorted(
        row["source_image_id"] for row in selected
    )


def test_balanced_subset_is_reproducible(tmp_path: Path) -> None:
    """The same seed and source index must produce byte-identical records."""
    index_path = tmp_path / "input.jsonl"
    _write_jsonl(
        index_path,
        [
            _sample(
                index,
                image_id=index,
                label="pocket",
                language="zh" if index % 2 else "en",
                dimensions=["basic", "spatial"] if index % 3 else ["basic"],
            )
            for index in range(20)
        ],
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    for output in (first, second):
        build_balanced_referring_subset(
            index_path=index_path,
            output_path=output,
            summary_output_path=output.with_suffix(".summary.json"),
            sample_count=8,
            seed="stable-seed",
        )

    assert first.read_bytes() == second.read_bytes()


def test_balanced_subset_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Duplicate source IDs cannot silently enter a balanced core."""
    index_path = tmp_path / "input.jsonl"
    duplicate = _sample(1, image_id=1, label="zipper", language="en")
    _write_jsonl(index_path, [duplicate, duplicate])

    with pytest.raises(ValueError, match="duplicate ID"):
        build_balanced_referring_subset(
            index_path=index_path,
            output_path=tmp_path / "output.jsonl",
            summary_output_path=tmp_path / "summary.json",
            sample_count=1,
        )


def test_balanced_subset_rejects_request_larger_than_source(tmp_path: Path) -> None:
    """The selector cannot report a requested scale that the input lacks."""
    index_path = tmp_path / "input.jsonl"
    _write_jsonl(index_path, [_sample(1, image_id=1, label="zipper", language="en")])

    with pytest.raises(ValueError, match="Requested 2 samples"):
        build_balanced_referring_subset(
            index_path=index_path,
            output_path=tmp_path / "output.jsonl",
            summary_output_path=tmp_path / "summary.json",
            sample_count=2,
        )


def _sample(
    index: int,
    *,
    image_id: int,
    label: str,
    language: str,
    dimensions: list[str] | None = None,
) -> dict[str, object]:
    selected_dimensions = dimensions or ["basic"]
    payload: dict[str, object] = {
        "schema_version": 1,
        "id": f"fashionpedia-train-{image_id}-{label}-{language}-{index}",
        "source_dataset": "fashionpedia",
        "split": "train",
        "image_path": f"data/raw/fashionpedia/train/{image_id}.jpg",
        "source_image_id": image_id,
        "query": f"{language} query {index}",
        "language": language,
        "dimensions": selected_dimensions,
        "target_label": label,
        "targets": [
            {
                "source_annotation_id": index,
                "label": label,
                "box": {"x_min": 1, "y_min": 1, "x_max": 5, "y_max": 5},
            }
        ],
        "template_id": f"template-{language}-{index}",
    }
    if "spatial" in selected_dimensions:
        payload["reference_frame"] = "image"
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
