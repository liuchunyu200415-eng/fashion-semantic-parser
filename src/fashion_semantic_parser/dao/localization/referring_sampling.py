"""Deterministic balanced sampling for large referring-expression indexes."""

import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from fashion_semantic_parser.dao.localization.referring_training import (
    ReferringTrainingSample,
)

DEFAULT_BALANCE_SEED = "prd-312-balanced-v1"
TARGETED_PARAPHRASE_SEED = "prd-312-paraphrase-v1"
WEAK_PART_LABELS = ("zipper", "rivet", "neckline", "pocket")

Stratum = tuple[str, str, str]
RankedRow = tuple[int, int, int, str]


class ReferringBalancedSubsetSummary(BaseModel):
    """Counts and provenance for one deterministic balanced subset."""

    input_sample_count: int = Field(ge=1)
    requested_sample_count: int = Field(ge=1)
    output_sample_count: int = Field(ge=1)
    selected_image_count: int = Field(ge=1)
    target_reference_count: int = Field(ge=1)
    seed: str = Field(min_length=1)
    sampling_unit: str = "target_label_x_language_x_modifier_dimensions"
    language_counts: dict[str, int]
    dimension_counts: dict[str, int]
    category_counts: dict[str, int]
    stratum_counts: dict[str, int]
    weak_part_counts: dict[str, int]
    input_path: str
    output_path: str


def build_balanced_referring_subset(
    *,
    index_path: Path,
    output_path: Path,
    summary_output_path: Path,
    sample_count: int = 100_000,
    seed: str = DEFAULT_BALANCE_SEED,
) -> ReferringBalancedSubsetSummary:
    """Select a reproducible balanced subset without loading the full index."""
    if sample_count < 1:
        raise ValueError("sample_count must be at least one.")
    normalized_seed = " ".join(seed.strip().split())
    if not normalized_seed:
        raise ValueError("seed cannot be empty.")

    stratum_capacities, input_count = _count_strata(index_path)
    if sample_count > input_count:
        raise ValueError(
            f"Requested {sample_count} samples from an index with {input_count}."
        )
    quotas = _allocate_balanced_quotas(stratum_capacities, sample_count)
    selected_rows = _select_stable_rows(
        index_path=index_path,
        quotas=quotas,
        seed=normalized_seed,
    )
    selected_rows.sort(key=lambda row: row[0])
    samples = [row[1] for row in selected_rows]
    if len(samples) != sample_count:
        raise RuntimeError(
            f"Balanced selection produced {len(samples)} of {sample_count} samples."
        )

    _write_samples_atomic(output_path, samples)
    summary = _build_summary(
        samples=samples,
        input_sample_count=input_count,
        requested_sample_count=sample_count,
        seed=normalized_seed,
        input_path=index_path,
        output_path=output_path,
    )
    _write_summary_atomic(summary_output_path, summary)
    return summary


def select_targeted_paraphrase_samples(
    samples: list[ReferringTrainingSample],
    *,
    limit: int | None,
) -> list[ReferringTrainingSample]:
    """Prioritize weak parts and modifier-rich queries, balanced by stratum."""
    if limit is None or limit >= len(samples):
        return list(samples)
    if limit < 1:
        raise ValueError("limit must be at least one when provided.")

    tiered: dict[int, dict[Stratum, list[ReferringTrainingSample]]] = {}
    for sample in samples:
        tier = _paraphrase_priority_tier(sample)
        tiered.setdefault(tier, {}).setdefault(_sample_stratum(sample), []).append(
            sample
        )

    selected: list[ReferringTrainingSample] = []
    for tier in sorted(tiered):
        groups = tiered[tier]
        for group in groups.values():
            group.sort(
                key=lambda sample: _stable_rank(TARGETED_PARAPHRASE_SEED, sample.id)
            )
        positions = {stratum: 0 for stratum in groups}
        active = sorted(groups)
        while active and len(selected) < limit:
            next_active: list[Stratum] = []
            for stratum in active:
                position = positions[stratum]
                group = groups[stratum]
                if position < len(group):
                    selected.append(group[position])
                    positions[stratum] += 1
                if positions[stratum] < len(group):
                    next_active.append(stratum)
                if len(selected) == limit:
                    break
            active = next_active
        if len(selected) == limit:
            break
    return selected


def _count_strata(index_path: Path) -> tuple[Counter[Stratum], int]:
    """Validate the index and count each balancing stratum in one pass."""
    counts: Counter[Stratum] = Counter()
    sample_ids: set[str] = set()
    total = 0
    for _, sample, _ in _iter_samples(index_path):
        if sample.id in sample_ids:
            raise ValueError(f"Referring index contains duplicate ID {sample.id!r}.")
        sample_ids.add(sample.id)
        counts[_sample_stratum(sample)] += 1
        total += 1
    if total == 0:
        raise ValueError(f"Referring training index is empty: {index_path}")
    return counts, total


def _allocate_balanced_quotas(
    capacities: Counter[Stratum],
    sample_count: int,
) -> dict[Stratum, int]:
    """Water-fill available strata so scarce cells are retained first."""
    quotas = {stratum: 0 for stratum in capacities}
    active = sorted(capacities)
    remaining = sample_count
    while remaining:
        progressed = False
        next_active: list[Stratum] = []
        for stratum in active:
            if quotas[stratum] < capacities[stratum] and remaining:
                quotas[stratum] += 1
                remaining -= 1
                progressed = True
            if quotas[stratum] < capacities[stratum]:
                next_active.append(stratum)
        if not progressed:
            raise RuntimeError("Balanced quota allocation exhausted its capacities.")
        active = next_active
    return quotas


def _select_stable_rows(
    *,
    index_path: Path,
    quotas: dict[Stratum, int],
    seed: str,
) -> list[tuple[int, ReferringTrainingSample]]:
    """Keep the lowest stable hashes in each stratum with bounded memory."""
    heaps: dict[Stratum, list[RankedRow]] = {
        stratum: [] for stratum, quota in quotas.items() if quota
    }
    for ordinal, sample, serialized in _iter_samples(index_path):
        stratum = _sample_stratum(sample)
        quota = quotas.get(stratum, 0)
        if not quota:
            continue
        rank = _stable_rank(seed, sample.id)
        row = (-rank, -ordinal, ordinal, serialized)
        heap = heaps[stratum]
        if len(heap) < quota:
            heapq.heappush(heap, row)
        elif row > heap[0]:
            heapq.heapreplace(heap, row)

    selected: list[tuple[int, ReferringTrainingSample]] = []
    for heap in heaps.values():
        for _, _, ordinal, serialized in heap:
            selected.append(
                (ordinal, ReferringTrainingSample.model_validate_json(serialized))
            )
    return selected


def _iter_samples(
    index_path: Path,
) -> Iterable[tuple[int, ReferringTrainingSample, str]]:
    """Yield validated non-blank JSONL rows with their source ordinal."""
    with index_path.open("r", encoding="utf-8") as input_file:
        for ordinal, line in enumerate(input_file):
            if not line.strip():
                raise ValueError(
                    f"Blank JSONL record at line {ordinal + 1} in {index_path}."
                )
            serialized = line.rstrip("\n")
            yield (
                ordinal,
                ReferringTrainingSample.model_validate_json(serialized),
                serialized,
            )


def _sample_stratum(sample: ReferringTrainingSample) -> Stratum:
    """Return the label, language, and complete modifier signature."""
    modifiers = sorted(
        dimension for dimension in sample.dimensions if dimension != "basic"
    )
    dimension_key = "+".join(modifiers) if modifiers else "basic"
    return sample.target_label, sample.language, dimension_key


def _paraphrase_priority_tier(sample: ReferringTrainingSample) -> int:
    """Rank weak-part and modifier-rich sources before generic expressions."""
    weak_part = sample.target_label in WEAK_PART_LABELS
    modifier_rich = len(sample.dimensions) > 1
    if weak_part and modifier_rich:
        return 0
    if weak_part:
        return 1
    if modifier_rich:
        return 2
    return 3


def _stable_rank(seed: str, sample_id: str) -> int:
    """Produce a platform-independent pseudo-random ordering key."""
    digest = hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _write_samples_atomic(
    path: Path,
    samples: list[ReferringTrainingSample],
) -> None:
    """Write only fully selected and validated rows to the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            for sample in samples:
                output_file.write(sample.model_dump_json() + "\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_summary_atomic(
    path: Path,
    summary: ReferringBalancedSubsetSummary,
) -> None:
    """Publish summary metadata only after its complete serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        payload = json.dumps(
            summary.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        temporary.write_text(f"{payload}\n", encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


# Audit provenance is passed explicitly instead of hidden in mutable state.
# pylint: disable-next=too-many-arguments
def _build_summary(
    *,
    samples: list[ReferringTrainingSample],
    input_sample_count: int,
    requested_sample_count: int,
    seed: str,
    input_path: Path,
    output_path: Path,
) -> ReferringBalancedSubsetSummary:
    """Summarize every axis used to audit the selected training core."""
    languages: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    image_ids: set[int] = set()
    target_count = 0
    for sample in samples:
        languages[sample.language] += 1
        dimensions.update(sample.dimensions)
        categories[sample.target_label] += 1
        strata["|".join(_sample_stratum(sample))] += 1
        image_ids.add(sample.source_image_id)
        target_count += len(sample.targets)
    return ReferringBalancedSubsetSummary(
        input_sample_count=input_sample_count,
        requested_sample_count=requested_sample_count,
        output_sample_count=len(samples),
        selected_image_count=len(image_ids),
        target_reference_count=target_count,
        seed=seed,
        language_counts=dict(sorted(languages.items())),
        dimension_counts=dict(sorted(dimensions.items())),
        category_counts=dict(sorted(categories.items())),
        stratum_counts=dict(sorted(strata.items())),
        weak_part_counts={label: categories[label] for label in WEAK_PART_LABELS},
        input_path=str(input_path),
        output_path=str(output_path),
    )
