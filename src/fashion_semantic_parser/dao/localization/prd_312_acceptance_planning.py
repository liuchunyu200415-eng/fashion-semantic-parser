"""Deterministic quota planning for PRD 3.1.2 acceptance review."""

import hashlib
from collections.abc import Mapping, Sequence
from typing import TypeVar, cast

from fashion_semantic_parser.dao.localization.prd_312_acceptance import (
    AcceptanceLanguage,
    AcceptanceTargetCardinality,
    AcceptanceTargetRegion,
    Prd312AcceptanceContract,
    acceptance_contract_blockers,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_review import (
    Prd312AcceptanceReviewPlan,
    Prd312AcceptanceReviewRecord,
)
from fashion_semantic_parser.dao.localization.referring_smoke import (
    ReferringQueryDimension,
    ReferringQueryNovelty,
    ReferringReferenceFrame,
)

_EXACT_SOURCE_REGIONS = frozenset(("collar", "pocket", "decoration"))
_WEAK_LABEL_REGIONS = {
    "zipper": "decoration",
    "rivet": "decoration",
    "neckline": "collar",
    "pocket": "pocket",
}
PlanValue = TypeVar("PlanValue", bound=str)


def build_prd_312_acceptance_review_plan(
    contract: Prd312AcceptanceContract,
) -> Prd312AcceptanceReviewPlan:
    """Create deterministic, balanced slots without fabricating evidence.

    Args:
        contract: Locked PRD 3.1.2 acceptance contract.

    Returns:
        A contract-sized plan whose records all remain pending.

    Raises:
        ValueError: If the contract is incomplete or quotas cannot be assigned.
    """
    blockers = acceptance_contract_blockers(contract)
    if blockers:
        raise ValueError("Acceptance contract is not ready: " + "; ".join(blockers))
    count = cast(int, contract.required_case_count)
    primary_dimensions = _spread_values(
        "primary_dimension", contract.primary_dimension_case_counts
    )
    novelty = _spread_values("novelty", contract.novelty_case_counts)
    languages = _spread_values("language", contract.language_case_counts)
    cardinalities = _spread_values(
        "target_cardinality", contract.target_cardinality_case_counts
    )
    regions = _spread_values("target_region", contract.target_region_case_counts)
    composite_indexes = _composite_indexes(contract, novelty)
    target_labels = _target_labels(contract, regions)
    records: list[Prd312AcceptanceReviewRecord] = []
    for index in range(count):
        primary = cast(ReferringQueryDimension, primary_dimensions[index])
        dimensions = _slot_dimensions(index, primary, index in composite_indexes)
        region = cast(AcceptanceTargetRegion, regions[index])
        reference_frame = _reference_frame(index) if "spatial" in dimensions else None
        records.append(
            Prd312AcceptanceReviewRecord(
                id=f"prd312-acceptance-{index + 1:04d}",
                language=cast(AcceptanceLanguage, languages[index]),
                primary_dimension=primary,
                dimensions=dimensions,
                novelty=cast(ReferringQueryNovelty, novelty[index]),
                target_cardinality=cast(
                    AcceptanceTargetCardinality, cardinalities[index]
                ),
                target_region=region,
                target_label=target_labels[index],
                reference_frame=reference_frame,
                annotation_requirement=(
                    "independent_exact_mask_review"
                    if region in _EXACT_SOURCE_REGIONS
                    else "independent_manual_mask"
                ),
            )
        )
    return Prd312AcceptanceReviewPlan(
        name="prd_312_acceptance_review_v1",
        description=" ".join(
            (
                "Independent human-review plan. Pending slots contain no fabricated",
                "image, query, or Mask evidence.",
            )
        ),
        contract=contract,
        records=records,
    )


def _spread_values(axis: str, counts: Mapping[PlanValue, int]) -> list[PlanValue]:
    """Expand exact counts into a stable pseudo-random ordering."""
    ranked: list[tuple[str, PlanValue]] = []
    for value, count in sorted(counts.items()):
        for ordinal in range(count):
            digest = hashlib.sha256(
                f"prd312-acceptance-v1:{axis}:{value}:{ordinal}".encode()
            ).hexdigest()
            ranked.append((digest, value))
    return [value for _, value in sorted(ranked)]


def _composite_indexes(
    contract: Prd312AcceptanceContract,
    novelty: Sequence[str],
) -> set[int]:
    """Assign the composite minimum preferentially to novel compositions."""
    required = cast(int, contract.minimum_composite_case_count)
    novel_candidates = [
        index for index, value in enumerate(novelty) if value == "novel_composition"
    ]
    seen_candidates = [
        index for index, value in enumerate(novelty) if value != "novel_composition"
    ]
    ranked = sorted(
        novel_candidates,
        key=lambda index: _stable_rank("composite", str(index)),
    )
    ranked.extend(
        sorted(
            seen_candidates,
            key=lambda index: _stable_rank("composite", str(index)),
        )
    )
    return set(ranked[:required])


def _slot_dimensions(
    index: int,
    primary: ReferringQueryDimension,
    composite: bool,
) -> list[ReferringQueryDimension]:
    """Build one modifier set while preserving its primary quota group."""
    dimensions: list[ReferringQueryDimension] = ["basic"]
    if primary != "basic":
        dimensions.append(primary)
    if not composite:
        return dimensions
    modifiers: tuple[ReferringQueryDimension, ...] = (
        "spatial",
        "attribute",
        "relation",
    )
    ranked = sorted(
        modifiers,
        key=lambda value: hashlib.sha256(
            f"prd312-acceptance-v1:modifier:{index}:{value}".encode()
        ).hexdigest(),
    )
    for modifier in ranked:
        if modifier not in dimensions:
            dimensions.append(modifier)
        if len(set(dimensions) - {"basic"}) >= 2:
            break
    return dimensions


def _reference_frame(index: int) -> ReferringReferenceFrame:
    """Spread spatial queries over all supported reference frames."""
    frames: tuple[ReferringReferenceFrame, ...] = ("image", "person", "garment")
    return frames[index % len(frames)]


def _stable_rank(namespace: str, value: str) -> str:
    """Return a reproducible SHA-256 sort key for one planner choice."""
    return hashlib.sha256(
        f"prd312-acceptance-v1:{namespace}:{value}".encode()
    ).hexdigest()


def _target_labels(
    contract: Prd312AcceptanceContract,
    regions: Sequence[str],
) -> list[str]:
    """Assign weak-label minima only within semantically valid PRD regions."""
    labels = list(regions)
    used: set[int] = set()
    for label, minimum in sorted(contract.minimum_target_label_case_counts.items()):
        required_region = _WEAK_LABEL_REGIONS.get(label)
        if required_region is None:
            raise ValueError(
                f"No PRD target-region mapping exists for weak label: {label}"
            )
        eligible = [
            index
            for index, region in enumerate(regions)
            if region == required_region and index not in used
        ]
        if len(eligible) < minimum:
            raise ValueError(
                f"Weak-label quota exceeds {required_region} capacity: {label}"
            )
        ranked = sorted(
            eligible,
            key=lambda index: hashlib.sha256(
                f"prd312-acceptance-v1:weak:{label}:{index}".encode()
            ).hexdigest(),
        )
        for index in ranked[:minimum]:
            labels[index] = label
            used.add(index)
    return labels
