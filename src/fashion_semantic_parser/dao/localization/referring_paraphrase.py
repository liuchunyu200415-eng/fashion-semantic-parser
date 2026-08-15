"""Auditable LLM paraphrase expansion for Fashionpedia referring records."""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from fashion_semantic_parser.dao.localization.referring_sampling import (
    select_targeted_paraphrase_samples,
)
from fashion_semantic_parser.dao.localization.referring_smoke import (
    ReferringQueryDimension,
)
from fashion_semantic_parser.dao.localization.referring_training import (
    ReferringTrainingSample,
    TrainingLanguage,
)

WEAK_PART_LABELS = ("zipper", "rivet", "neckline", "pocket")


class ReferringParaphraseJob(BaseModel):
    """One vendor-neutral request to paraphrase without changing the referent."""

    schema_version: Literal[1] = 1
    source_sample_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_query: str = Field(min_length=1)
    language: TrainingLanguage
    dimensions: list[ReferringQueryDimension] = Field(min_length=1)
    target_label: str = Field(min_length=1)
    target_count: int = Field(ge=1)
    requested_paraphrase_count: int = Field(ge=1, le=20)
    instruction: str = Field(min_length=1)


class ReferringParaphraseResult(BaseModel):
    """Reviewed model output tied to one immutable source-record fingerprint."""

    schema_version: Literal[1] = 1
    source_sample_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    language: TrainingLanguage
    generator_model: str = Field(min_length=1)
    review_status: Literal["unreviewed", "reviewed"] = "unreviewed"
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    paraphrases: list[str] = Field(min_length=1, max_length=20)

    @field_validator("generator_model", "reviewed_by")
    @classmethod
    def normalize_optional_identity(cls, value: str | None) -> str | None:
        """Store stable non-empty generator and reviewer identities."""
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Paraphrase identity fields cannot be empty.")
        return normalized

    @field_validator("paraphrases")
    @classmethod
    def normalize_paraphrases(cls, values: list[str]) -> list[str]:
        """Reject empty or duplicate rewrites before they reach training."""
        normalized = [" ".join(value.strip().split()) for value in values]
        if any(not value for value in normalized):
            raise ValueError("Paraphrases cannot be empty.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Paraphrases cannot contain duplicates.")
        return normalized

    @model_validator(mode="after")
    def validate_review_provenance(self) -> "ReferringParaphraseResult":
        """Require an auditable reviewer identity before trusting LLM output."""
        if self.review_status == "reviewed" and (
            self.reviewed_by is None or self.reviewed_at is None
        ):
            raise ValueError(
                "Reviewed paraphrases require reviewed_by and reviewed_at."
            )
        return self


class ReferringParaphraseExpansionSummary(BaseModel):
    """Bounded counts and provenance for one expanded JSONL dataset."""

    base_sample_count: int
    llm_paraphrase_sample_count: int
    output_sample_count: int
    minimum_sample_count: int
    minimum_sample_count_passed: bool
    reviewed_result_count: int
    unreviewed_result_count: int
    skipped_duplicate_paraphrase_count: int
    language_counts: dict[str, int]
    dimension_counts: dict[str, int]
    category_counts: dict[str, int]
    augmentation_method_counts: dict[str, int]
    weak_part_counts: dict[str, int]
    base_index_path: str
    result_path: str
    output_path: str


def export_referring_paraphrase_jobs(
    *,
    index_path: Path,
    output_path: Path,
    paraphrases_per_sample: int = 3,
    limit: int | None = None,
    selection_policy: Literal["prefix", "weak_complex_balanced"] = "prefix",
) -> int:
    """Write atomic JSONL jobs whose target identity cannot be rewritten."""
    if not 1 <= paraphrases_per_sample <= 20:
        raise ValueError("paraphrases_per_sample must be between 1 and 20.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least one when provided.")
    if selection_policy not in ("prefix", "weak_complex_balanced"):
        raise ValueError(f"Unsupported paraphrase selection policy: {selection_policy}")
    samples = _read_training_samples(index_path)
    if selection_policy == "weak_complex_balanced":
        samples = select_targeted_paraphrase_samples(samples, limit=limit)
    elif selection_policy == "prefix" and limit is not None:
        samples = samples[:limit]
    jobs = [
        ReferringParaphraseJob(
            source_sample_id=sample.id,
            source_fingerprint=referring_sample_fingerprint(sample),
            source_query=sample.query,
            language=sample.language,
            dimensions=list(sample.dimensions),
            target_label=sample.target_label,
            target_count=len(sample.targets),
            requested_paraphrase_count=paraphrases_per_sample,
            instruction=_paraphrase_instruction(sample, paraphrases_per_sample),
        )
        for sample in samples
    ]
    _write_jsonl_atomic(output_path, [job.model_dump(mode="json") for job in jobs])
    return len(jobs)


# The merge boundary intentionally exposes CLI provenance and audit controls.
# pylint: disable-next=too-many-arguments,too-many-locals,too-many-branches
def merge_referring_paraphrases(
    *,
    base_index_path: Path,
    result_path: Path,
    output_path: Path,
    summary_output_path: Path,
    minimum_sample_count: int = 100_000,
    allow_unreviewed: bool = False,
) -> ReferringParaphraseExpansionSummary:
    """Merge reviewed rewrites while preserving source image, target, and Mask IDs."""
    if minimum_sample_count < 1:
        raise ValueError("minimum_sample_count must be at least one.")
    base_samples = _read_training_samples(base_index_path)
    base_by_id = {sample.id: sample for sample in base_samples}
    if len(base_by_id) != len(base_samples):
        raise ValueError("Base referring index contains duplicate sample IDs.")
    results = _read_paraphrase_results(result_path)
    result_by_id = {result.source_sample_id: result for result in results}
    if len(result_by_id) != len(results):
        raise ValueError("Paraphrase results contain duplicate source_sample_id.")
    unknown_ids = sorted(set(result_by_id).difference(base_by_id))
    if unknown_ids:
        raise ValueError(
            f"Paraphrase results reference unknown samples: {unknown_ids[:10]}"
        )

    output_samples: list[ReferringTrainingSample] = []
    seen_query_keys: set[tuple[int, str, tuple[int, ...], str]] = set()
    skipped_duplicates = 0
    reviewed_count = 0
    unreviewed_count = 0
    for base_sample in base_samples:
        output_samples.append(base_sample)
        seen_query_keys.add(_query_identity(base_sample, base_sample.query))
        result = result_by_id.get(base_sample.id)
        if result is None:
            continue
        if result.review_status == "reviewed":
            reviewed_count += 1
        else:
            unreviewed_count += 1
            if not allow_unreviewed:
                raise ValueError(
                    f"Paraphrase result {base_sample.id!r} is not reviewed."
                )
        if result.language != base_sample.language:
            raise ValueError(
                f"Paraphrase language differs for sample {base_sample.id!r}."
            )
        if result.source_fingerprint != referring_sample_fingerprint(base_sample):
            raise ValueError(
                f"Paraphrase source fingerprint differs for {base_sample.id!r}."
            )
        for index, query in enumerate(result.paraphrases, start=1):
            identity = _query_identity(base_sample, query)
            if identity in seen_query_keys:
                skipped_duplicates += 1
                continue
            seen_query_keys.add(identity)
            payload = base_sample.model_dump(mode="json")
            payload.update(
                {
                    "id": f"{base_sample.id}-llm-{index:02d}",
                    "query": query,
                    "template_id": f"llm-paraphrase-{index:02d}",
                    "augmentation_method": "llm_paraphrase",
                    "source_sample_id": base_sample.id,
                    "generator_model": result.generator_model,
                }
            )
            output_samples.append(ReferringTrainingSample.model_validate(payload))

    output_payloads = [sample.model_dump(mode="json") for sample in output_samples]
    _write_jsonl_atomic(output_path, output_payloads)
    summary = _build_expansion_summary(
        samples=output_samples,
        base_sample_count=len(base_samples),
        minimum_sample_count=minimum_sample_count,
        reviewed_result_count=reviewed_count,
        unreviewed_result_count=unreviewed_count,
        skipped_duplicate_paraphrase_count=skipped_duplicates,
        base_index_path=base_index_path,
        result_path=result_path,
        output_path=output_path,
    )
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if not summary.minimum_sample_count_passed:
        output_path.unlink(missing_ok=True)
        raise ValueError(
            f"Expanded dataset has {summary.output_sample_count} samples; "
            f"minimum is {minimum_sample_count}."
        )
    return summary


def referring_sample_fingerprint(sample: ReferringTrainingSample) -> str:
    """Hash every field that defines the source expression and its referent."""
    payload = {
        "id": sample.id,
        "image_path": sample.image_path,
        "source_image_id": sample.source_image_id,
        "query": sample.query,
        "language": sample.language,
        "dimensions": sample.dimensions,
        "reference_frame": sample.reference_frame,
        "target_label": sample.target_label,
        "targets": [target.model_dump(mode="json") for target in sample.targets],
        "source_attribute_ids": sample.source_attribute_ids,
        "reference_category": sample.reference_category,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_training_samples(path: Path) -> list[ReferringTrainingSample]:
    """Read a non-empty validated JSONL training index."""
    samples = [
        ReferringTrainingSample.model_validate_json(line)
        for line in _read_nonblank_jsonl_lines(path)
    ]
    if not samples:
        raise ValueError(f"Referring training index is empty: {path}")
    return samples


def _read_paraphrase_results(path: Path) -> list[ReferringParaphraseResult]:
    """Read non-empty validated vendor-neutral paraphrase results."""
    results = [
        ReferringParaphraseResult.model_validate_json(line)
        for line in _read_nonblank_jsonl_lines(path)
    ]
    if not results:
        raise ValueError(f"Paraphrase result index is empty: {path}")
    return results


def _read_nonblank_jsonl_lines(path: Path) -> list[str]:
    """Reject blank records so sample counts cannot drift silently."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f"JSONL input contains a blank record: {path}")
    return lines


def _write_jsonl_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    """Replace one JSONL artifact only after every record serializes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _paraphrase_instruction(
    sample: ReferringTrainingSample,
    paraphrase_count: int,
) -> str:
    """Build a bounded instruction that forbids referent-changing rewrites."""
    return (
        f"Rewrite the query into exactly {paraphrase_count} natural "
        f"{sample.language} referring expressions. Preserve target "
        f"'{sample.target_label}', target count {len(sample.targets)}, all "
        f"spatial/attribute/relation modifiers, and the reference frame. "
        "Do not add objects, properties, directions, or relationships."
    )


def _query_identity(
    sample: ReferringTrainingSample,
    query: str,
) -> tuple[int, str, tuple[int, ...], str]:
    """Deduplicate rewrites only when image, target set, and language agree."""
    normalized_query = " ".join(query.strip().split()).casefold()
    annotation_ids = tuple(
        sorted(target.source_annotation_id for target in sample.targets)
    )
    return (
        sample.source_image_id,
        sample.language,
        annotation_ids,
        normalized_query,
    )


# Summary inputs are explicit so no audit count is inferred from hidden state.
# pylint: disable-next=too-many-arguments
def _build_expansion_summary(
    *,
    samples: list[ReferringTrainingSample],
    base_sample_count: int,
    minimum_sample_count: int,
    reviewed_result_count: int,
    unreviewed_result_count: int,
    skipped_duplicate_paraphrase_count: int,
    base_index_path: Path,
    result_path: Path,
    output_path: Path,
) -> ReferringParaphraseExpansionSummary:
    """Aggregate data scale, balance, and augmentation provenance."""
    language_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    for sample in samples:
        language_counts[sample.language] += 1
        category_counts[sample.target_label] += 1
        method_counts[sample.augmentation_method] += 1
        dimension_counts.update(sample.dimensions)
    return ReferringParaphraseExpansionSummary(
        base_sample_count=base_sample_count,
        llm_paraphrase_sample_count=method_counts["llm_paraphrase"],
        output_sample_count=len(samples),
        minimum_sample_count=minimum_sample_count,
        minimum_sample_count_passed=len(samples) >= minimum_sample_count,
        reviewed_result_count=reviewed_result_count,
        unreviewed_result_count=unreviewed_result_count,
        skipped_duplicate_paraphrase_count=skipped_duplicate_paraphrase_count,
        language_counts=dict(sorted(language_counts.items())),
        dimension_counts=dict(sorted(dimension_counts.items())),
        category_counts=dict(sorted(category_counts.items())),
        augmentation_method_counts=dict(sorted(method_counts.items())),
        weak_part_counts={label: category_counts[label] for label in WEAK_PART_LABELS},
        base_index_path=str(base_index_path),
        result_path=str(result_path),
        output_path=str(output_path),
    )
