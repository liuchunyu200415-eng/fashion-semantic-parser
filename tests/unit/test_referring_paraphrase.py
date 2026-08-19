"""Tests for auditable Fashionpedia LLM paraphrase expansion."""

# Script bootstrap helpers intentionally share the repository's direct-run pattern.
# pylint: disable=duplicate-code

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fashion_semantic_parser.dao.localization.referring_paraphrase import (
    export_referring_paraphrase_jobs,
    merge_referring_paraphrases,
)
from fashion_semantic_parser.dao.localization.referring_training import (
    ReferringTrainingSample,
)


def test_export_jobs_records_referent_fingerprint_and_constraints(
    tmp_path: Path,
) -> None:
    """LLM jobs must describe the rewrite without exposing mutable GT output."""
    base_path = _write_base_index(tmp_path)
    job_path = tmp_path / "jobs.jsonl"

    count = export_referring_paraphrase_jobs(
        index_path=base_path,
        output_path=job_path,
        paraphrases_per_sample=2,
        limit=1,
    )
    job = _read_jsonl(job_path)[0]

    assert count == 1
    assert job["source_sample_id"] == "fashionpedia-train-10-pocket-basic-zh-1"
    assert job["schema_version"] == 2
    assert len(job["source_fingerprint"]) == 64
    assert job["requested_paraphrase_count"] == 2
    assert "Preserve target 'pocket'" in job["instruction"]
    assert "target count 1" in job["instruction"]


def test_targeted_export_prioritizes_weak_modifier_queries(tmp_path: Path) -> None:
    """A bounded rewrite batch starts with weak labels plus rich modifiers."""
    rows = _base_samples()
    weak_spatial = dict(rows[1])
    weak_spatial.update(
        {
            "id": "fashionpedia-train-20-zipper-spatial-en-3",
            "query": "the zipper on the lower side of the garment",
            "dimensions": ["basic", "spatial"],
            "reference_frame": "image",
            "template_id": "spatial-lower-en",
        }
    )
    rows.append(weak_spatial)
    base_path = tmp_path / "base.jsonl"
    _write_jsonl(base_path, rows)
    job_path = tmp_path / "jobs.jsonl"

    count = export_referring_paraphrase_jobs(
        index_path=base_path,
        output_path=job_path,
        limit=1,
        selection_policy="weak_complex_balanced",
    )
    job = _read_jsonl(job_path)[0]

    assert count == 1
    assert job["source_sample_id"] == weak_spatial["id"]
    assert job["dimensions"] == ["basic", "spatial"]
    assert job["spatial_modifier"] == "lower"


def test_merge_reviewed_paraphrases_preserves_source_targets_and_provenance(
    tmp_path: Path,
) -> None:
    """Only query text changes; image, Mask IDs, dimensions, and label stay fixed."""
    base_path = _write_base_index(tmp_path)
    jobs_path = tmp_path / "jobs.jsonl"
    export_referring_paraphrase_jobs(
        index_path=base_path,
        output_path=jobs_path,
        paraphrases_per_sample=2,
        limit=1,
    )
    job = _read_jsonl(jobs_path)[0]
    result_path = tmp_path / "results.jsonl"
    _write_jsonl(
        result_path,
        [
            {
                "schema_version": 1,
                "source_sample_id": job["source_sample_id"],
                "source_fingerprint": job["source_fingerprint"],
                "language": "zh",
                "generator_model": "reviewed-model-v1",
                "review_status": "reviewed",
                "reviewed_by": "dataset-reviewer",
                "reviewed_at": "2026-08-14T12:00:00+08:00",
                "paraphrases": ["请定位这件衣服的口袋", "这件衣服的口袋"],
            }
        ],
    )
    output_path = tmp_path / "expanded.jsonl"

    summary = merge_referring_paraphrases(
        base_index_path=base_path,
        result_path=result_path,
        output_path=output_path,
        summary_output_path=tmp_path / "summary.json",
        minimum_sample_count=3,
    )
    output = _read_jsonl(output_path)
    source = output[0]
    paraphrase = output[1]

    assert summary.base_sample_count == 2
    assert summary.llm_paraphrase_sample_count == 1
    assert summary.output_sample_count == 3
    assert summary.skipped_duplicate_paraphrase_count == 1
    assert paraphrase["query"] == "请定位这件衣服的口袋"
    assert paraphrase["targets"] == source["targets"]
    assert paraphrase["image_path"] == source["image_path"]
    assert paraphrase["augmentation_method"] == "llm_paraphrase"
    assert paraphrase["source_sample_id"] == source["id"]
    assert paraphrase["generator_model"] == "reviewed-model-v1"
    assert summary.weak_part_counts["pocket"] == 2


def test_merge_rejects_tampered_source_fingerprint(tmp_path: Path) -> None:
    """A rewrite cannot attach to a source record that changed after export."""
    base_path = _write_base_index(tmp_path)
    result_path = tmp_path / "results.jsonl"
    _write_jsonl(
        result_path,
        [
            {
                "source_sample_id": "fashionpedia-train-10-pocket-basic-zh-1",
                "source_fingerprint": "0" * 64,
                "language": "zh",
                "generator_model": "model-v1",
                "review_status": "reviewed",
                "reviewed_by": "dataset-reviewer",
                "reviewed_at": "2026-08-14T12:00:00+08:00",
                "paraphrases": ["找到这件衣服的口袋"],
            }
        ],
    )

    with pytest.raises(ValueError, match="fingerprint differs"):
        merge_referring_paraphrases(
            base_index_path=base_path,
            result_path=result_path,
            output_path=tmp_path / "expanded.jsonl",
            summary_output_path=tmp_path / "summary.json",
            minimum_sample_count=1,
        )


def test_merge_blocks_unreviewed_results_by_default(tmp_path: Path) -> None:
    """Raw LLM output must not silently become trusted training supervision."""
    base_path = _write_base_index(tmp_path)
    jobs_path = tmp_path / "jobs.jsonl"
    export_referring_paraphrase_jobs(
        index_path=base_path,
        output_path=jobs_path,
        limit=1,
    )
    job = _read_jsonl(jobs_path)[0]
    result_path = tmp_path / "results.jsonl"
    _write_jsonl(
        result_path,
        [
            {
                "source_sample_id": job["source_sample_id"],
                "source_fingerprint": job["source_fingerprint"],
                "language": "zh",
                "generator_model": "model-v1",
                "review_status": "unreviewed",
                "paraphrases": ["找到这件衣服的口袋"],
            }
        ],
    )

    with pytest.raises(ValueError, match="is not reviewed"):
        merge_referring_paraphrases(
            base_index_path=base_path,
            result_path=result_path,
            output_path=tmp_path / "expanded.jsonl",
            summary_output_path=tmp_path / "summary.json",
            minimum_sample_count=1,
        )


def test_merge_removes_output_when_minimum_scale_is_not_met(tmp_path: Path) -> None:
    """A small successful merge cannot be reported as the 100k data milestone."""
    base_path = _write_base_index(tmp_path)
    jobs_path = tmp_path / "jobs.jsonl"
    export_referring_paraphrase_jobs(
        index_path=base_path,
        output_path=jobs_path,
        limit=1,
    )
    job = _read_jsonl(jobs_path)[0]
    result_path = tmp_path / "results.jsonl"
    _write_jsonl(
        result_path,
        [
            {
                "source_sample_id": job["source_sample_id"],
                "source_fingerprint": job["source_fingerprint"],
                "language": "zh",
                "generator_model": "model-v1",
                "review_status": "reviewed",
                "reviewed_by": "dataset-reviewer",
                "reviewed_at": "2026-08-14T12:00:00+08:00",
                "paraphrases": ["找到这件衣服的口袋"],
            }
        ],
    )
    output_path = tmp_path / "expanded.jsonl"

    with pytest.raises(ValueError, match="minimum is 100000"):
        merge_referring_paraphrases(
            base_index_path=base_path,
            result_path=result_path,
            output_path=output_path,
            summary_output_path=tmp_path / "summary.json",
        )

    assert not output_path.exists()


def test_training_sample_rejects_incomplete_llm_provenance() -> None:
    """Augmented rows require both their immutable parent ID and model name."""
    payload = _base_samples()[0]
    payload["augmentation_method"] = "llm_paraphrase"

    with pytest.raises(ValidationError, match="require source_sample_id"):
        ReferringTrainingSample.model_validate(payload)


def test_reviewed_result_requires_reviewer_identity(tmp_path: Path) -> None:
    """A reviewed flag without reviewer provenance cannot enter the merge."""
    base_path = _write_base_index(tmp_path)
    jobs_path = tmp_path / "jobs.jsonl"
    export_referring_paraphrase_jobs(
        index_path=base_path,
        output_path=jobs_path,
        limit=1,
    )
    job = _read_jsonl(jobs_path)[0]
    result_path = tmp_path / "results.jsonl"
    _write_jsonl(
        result_path,
        [
            {
                "source_sample_id": job["source_sample_id"],
                "source_fingerprint": job["source_fingerprint"],
                "language": "zh",
                "generator_model": "model-v1",
                "review_status": "reviewed",
                "paraphrases": ["找到这件衣服的口袋"],
            }
        ],
    )

    with pytest.raises(ValidationError, match="require reviewed_by"):
        merge_referring_paraphrases(
            base_index_path=base_path,
            result_path=result_path,
            output_path=tmp_path / "expanded.jsonl",
            summary_output_path=tmp_path / "summary.json",
            minimum_sample_count=1,
        )


def test_export_rejects_blank_jsonl_records(tmp_path: Path) -> None:
    """Blank rows cannot silently reduce the exported rewrite-job count."""
    base_path = _write_base_index(tmp_path)
    base_path.write_text(
        base_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="blank record"):
        export_referring_paraphrase_jobs(
            index_path=base_path,
            output_path=tmp_path / "jobs.jsonl",
        )


def _write_base_index(tmp_path: Path) -> Path:
    path = tmp_path / "base.jsonl"
    _write_jsonl(path, _base_samples())
    return path


def _base_samples() -> list[dict[str, object]]:
    return [
        {
            "schema_version": 1,
            "id": "fashionpedia-train-10-pocket-basic-zh-1",
            "source_dataset": "fashionpedia",
            "split": "train",
            "image_path": "data/raw/fashionpedia/train/a.jpg",
            "source_image_id": 10,
            "query": "这件衣服的口袋",
            "language": "zh",
            "dimensions": ["basic"],
            "target_label": "pocket",
            "targets": [
                {
                    "source_annotation_id": 1,
                    "label": "pocket",
                    "box": {"x_min": 1, "y_min": 1, "x_max": 5, "y_max": 5},
                }
            ],
            "template_id": "basic-zh",
        },
        {
            "schema_version": 1,
            "id": "fashionpedia-train-20-zipper-basic-en-2",
            "source_dataset": "fashionpedia",
            "split": "train",
            "image_path": "data/raw/fashionpedia/train/b.jpg",
            "source_image_id": 20,
            "query": "the zipper on the garment",
            "language": "en",
            "dimensions": ["basic"],
            "target_label": "zipper",
            "targets": [
                {
                    "source_annotation_id": 2,
                    "label": "zipper",
                    "box": {"x_min": 2, "y_min": 2, "x_max": 3, "y_max": 8},
                }
            ],
            "template_id": "basic-en",
        },
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
