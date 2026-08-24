"""Human screening worklist for PRD 3.1.2 acceptance holdout images."""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, field_validator, model_validator

from fashion_semantic_parser.dao.localization.prd_312_acceptance import (
    AcceptanceTargetRegion,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_holdout import (
    Prd312AcceptanceHoldoutInventory,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_review import (
    Prd312AcceptanceReviewPlan,
)

ScreeningStatus = Literal["pending", "eligible", "rejected"]
WeakTargetLabel = Literal["zipper", "rivet", "neckline", "pocket"]

TARGET_REGIONS: tuple[AcceptanceTargetRegion, ...] = (
    "collar",
    "cuff",
    "hem",
    "pocket",
    "shoulder",
    "waist",
    "pattern",
    "decoration",
)
WEAK_TARGET_LABELS: tuple[WeakTargetLabel, ...] = (
    "zipper",
    "rivet",
    "neckline",
    "pocket",
)
_WEAK_LABEL_REGION: dict[WeakTargetLabel, AcceptanceTargetRegion] = {
    "zipper": "decoration",
    "rivet": "decoration",
    "neckline": "collar",
    "pocket": "pocket",
}


class Prd312AcceptanceScreeningRecord(BaseModel):
    """One immutable candidate image plus human screening evidence."""

    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    screening_status: ScreeningStatus = "pending"
    target_region_instance_counts: dict[AcceptanceTargetRegion, int] = Field(
        default_factory=dict
    )
    weak_target_label_instance_counts: dict[WeakTargetLabel, int] = Field(
        default_factory=dict
    )
    rejection_reason: str | None = None
    screened_by: str | None = None
    screened_at: datetime | None = None
    notes: str | None = None

    @field_validator("rejection_reason", "screened_by", "notes")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Normalize optional reviewer text.

        Args:
            value: Optional human-entered value.

        Returns:
            Normalized text or ``None``.

        Raises:
            ValueError: If a provided value contains only whitespace.
        """
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Screening text cannot be whitespace-only.")
        return normalized

    @field_validator(
        "target_region_instance_counts", "weak_target_label_instance_counts"
    )
    @classmethod
    def validate_positive_counts(cls, values: dict[str, int]) -> dict[str, int]:
        """Reject zero and negative instance counts.

        Args:
            values: Sparse human-entered instance counts.

        Returns:
            The validated sparse count mapping.

        Raises:
            ValueError: If any stored count is below one.
        """
        if any(value < 1 for value in values.values()):
            raise ValueError("Stored screening instance counts must be positive.")
        return values

    @model_validator(mode="after")
    def validate_screening_evidence(self) -> "Prd312AcceptanceScreeningRecord":
        """Keep pending, eligible, and rejected evidence mutually exclusive.

        Returns:
            The validated screening record.

        Raises:
            ValueError: If the status conflicts with its human evidence.
        """
        reviewer_missing = self.screened_by is None or self.screened_at is None
        if (self.screened_by is None) != (self.screened_at is None):
            raise ValueError("Screening reviewer and timestamp must be paired.")
        if self.screened_at is not None and self.screened_at.utcoffset() is None:
            raise ValueError("Screening timestamp must include a timezone.")
        if self.screening_status == "pending":
            if (
                self.target_region_instance_counts
                or self.weak_target_label_instance_counts
                or self.rejection_reason is not None
                or not reviewer_missing
                or self.notes is not None
            ):
                raise ValueError("Pending screening records cannot contain evidence.")
        elif reviewer_missing:
            raise ValueError("Completed screening requires reviewer evidence.")
        elif self.screening_status == "eligible":
            if not self.target_region_instance_counts:
                raise ValueError("Eligible images require at least one target region.")
            if self.rejection_reason is not None:
                raise ValueError("Eligible images cannot have a rejection reason.")
            self._validate_weak_label_regions()
        else:
            if self.rejection_reason is None:
                raise ValueError("Rejected images require a rejection reason.")
            if (
                self.target_region_instance_counts
                or self.weak_target_label_instance_counts
            ):
                raise ValueError("Rejected images cannot retain instance counts.")
        return self

    def _validate_weak_label_regions(self) -> None:
        """Require every weak-label count to have compatible region evidence."""
        weak_counts = dict(self.weak_target_label_instance_counts)
        region_counts = dict(self.target_region_instance_counts)
        for label, count in weak_counts.items():
            region = _WEAK_LABEL_REGION[label]
            region_count = region_counts.get(region, 0)
            if count > region_count:
                raise ValueError(f"Weak-label count exceeds {region} count: {label}")


class Prd312AcceptanceScreeningPlan(BaseModel):
    """Versioned human-screening worklist bound to frozen input artifacts."""

    schema_version: Literal[1] = 1
    name: Literal["prd_312_acceptance_screening_v1"] = "prd_312_acceptance_screening_v1"
    generated_at: datetime
    candidate_inventory_path: str = Field(min_length=1)
    candidate_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_plan_path: str = Field(min_length=1)
    review_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_list_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exclusion_list_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_region_case_counts: dict[AcceptanceTargetRegion, int]
    minimum_weak_label_case_counts: dict[WeakTargetLabel, int]
    required_multi_target_case_count: int = Field(ge=0)
    model_assisted_selection_prohibited: Literal[True] = True
    formal_holdout_ready: Literal[False] = False
    records: list[Prd312AcceptanceScreeningRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> "Prd312AcceptanceScreeningPlan":
        """Reject duplicate candidates and incomplete locked requirements.

        Returns:
            The validated screening worklist.

        Raises:
            ValueError: If timestamps, candidates, or requirements conflict.
        """
        if self.generated_at.utcoffset() is None:
            raise ValueError("Screening generation time must include a timezone.")
        paths = [record.image_path for record in self.records]
        hashes = [record.image_sha256 for record in self.records]
        if len(paths) != len(set(paths)):
            raise ValueError("Screening image paths must be unique.")
        if len(hashes) != len(set(hashes)):
            raise ValueError("Screening image hashes must be unique.")
        if set(self.required_region_case_counts) != set(TARGET_REGIONS):
            raise ValueError("Screening plan requires every locked target region.")
        if set(self.minimum_weak_label_case_counts) != set(WEAK_TARGET_LABELS):
            raise ValueError("Screening plan requires every locked weak label.")
        return self


class Prd312AcceptanceScreeningSources(BaseModel):
    """Immutable paths and hashes of both screening source artifacts."""

    candidate_inventory_path: str = Field(min_length=1)
    candidate_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_plan_path: str = Field(min_length=1)
    review_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Prd312AcceptanceScreeningSummary(BaseModel):
    """Human-screening progress and conservative quota-supply diagnostics."""

    candidate_image_count: int = Field(ge=0)
    screening_status_counts: dict[str, int]
    screened_image_count: int = Field(ge=0)
    pending_image_count: int = Field(ge=0)
    region_eligible_image_counts: dict[AcceptanceTargetRegion, int]
    weak_label_eligible_image_counts: dict[WeakTargetLabel, int]
    multi_target_region_case_capacity: int = Field(ge=0)
    region_supply_deficits: dict[AcceptanceTargetRegion, int]
    weak_label_supply_deficits: dict[WeakTargetLabel, int]
    multi_target_supply_deficit: int = Field(ge=0)
    minimum_supply_ready: bool
    model_assisted_selection_prohibited: Literal[True] = True
    formal_holdout_ready: Literal[False] = False


def build_prd_312_acceptance_screening_plan(
    *,
    inventory: Prd312AcceptanceHoldoutInventory,
    review_plan: Prd312AcceptanceReviewPlan,
    sources: Prd312AcceptanceScreeningSources,
    generated_at: datetime,
) -> Prd312AcceptanceScreeningPlan:
    """Build a pending worklist without assigning images through a model.

    Args:
        inventory: Frozen and mechanically audited candidate inventory.
        review_plan: Locked 1,000-slot human-review plan.
        sources: Immutable paths and hashes of both source artifacts.
        generated_at: Time at which the worklist was generated.

    Returns:
        Pending screening worklist containing every eligible candidate image.

    Raises:
        ValueError: If input artifacts already claim readiness or review progress.
    """
    if inventory.formal_holdout_ready or inventory.independence_attested:
        raise ValueError("Candidate inventory must remain mechanically pending.")
    if any(record.review_status != "pending" for record in review_plan.records):
        raise ValueError("Screening requires a pristine pending review plan.")
    if any(record.image_path is not None for record in review_plan.records):
        raise ValueError("Screening cannot replace preassigned review images.")
    candidates = []
    for image in inventory.images:
        if image.status != "candidate":
            continue
        if image.width is None or image.height is None:
            raise ValueError("Candidate inventory image is missing dimensions.")
        candidates.append(
            Prd312AcceptanceScreeningRecord(
                image_path=image.image_path,
                image_sha256=image.image_sha256,
                width=image.width,
                height=image.height,
            )
        )
    contract = review_plan.contract
    return Prd312AcceptanceScreeningPlan(
        generated_at=generated_at,
        candidate_inventory_path=sources.candidate_inventory_path,
        candidate_inventory_sha256=sources.candidate_inventory_sha256,
        review_plan_path=sources.review_plan_path,
        review_plan_sha256=sources.review_plan_sha256,
        source_list_sha256=inventory.source_list_sha256,
        exclusion_list_sha256=inventory.exclusion_list_sha256,
        required_region_case_counts=contract.target_region_case_counts,
        minimum_weak_label_case_counts=cast(
            dict[WeakTargetLabel, int], contract.minimum_target_label_case_counts
        ),
        required_multi_target_case_count=contract.target_cardinality_case_counts[
            "multi_target"
        ],
        records=candidates,
    )


def summarize_prd_312_acceptance_screening(
    plan: Prd312AcceptanceScreeningPlan,
) -> Prd312AcceptanceScreeningSummary:
    """Summarize reviewed image supply without claiming formal readiness.

    Args:
        plan: Screening plan to audit.

    Returns:
        Progress counters and deficits against locked minimums.
    """
    statuses = Counter(record.screening_status for record in plan.records)
    region_counts = {
        region: sum(
            record.target_region_instance_counts.get(region, 0) > 0
            for record in plan.records
        )
        for region in TARGET_REGIONS
    }
    weak_counts = {
        label: sum(
            record.weak_target_label_instance_counts.get(label, 0) > 0
            for record in plan.records
        )
        for label in WEAK_TARGET_LABELS
    }
    multi_target_capacity = sum(
        count >= 2
        for record in plan.records
        for count in record.target_region_instance_counts.values()
    )
    region_deficits = {
        region: max(required - region_counts[region], 0)
        for region, required in plan.required_region_case_counts.items()
    }
    weak_deficits = {
        label: max(required - weak_counts[label], 0)
        for label, required in plan.minimum_weak_label_case_counts.items()
    }
    multi_target_deficit = max(
        plan.required_multi_target_case_count - multi_target_capacity,
        0,
    )
    return Prd312AcceptanceScreeningSummary(
        candidate_image_count=len(plan.records),
        screening_status_counts=dict(sorted(statuses.items())),
        screened_image_count=statuses["eligible"] + statuses["rejected"],
        pending_image_count=statuses["pending"],
        region_eligible_image_counts=region_counts,
        weak_label_eligible_image_counts=weak_counts,
        multi_target_region_case_capacity=multi_target_capacity,
        region_supply_deficits=region_deficits,
        weak_label_supply_deficits=weak_deficits,
        multi_target_supply_deficit=multi_target_deficit,
        minimum_supply_ready=(
            not any(region_deficits.values())
            and not any(weak_deficits.values())
            and multi_target_deficit == 0
        ),
    )


def load_prd_312_acceptance_screening_plan(
    path: Path,
) -> Prd312AcceptanceScreeningPlan:
    """Read and validate one UTF-8 screening-plan JSON artifact.

    Args:
        path: Screening-plan JSON path.

    Returns:
        Validated screening plan.
    """
    with path.open("r", encoding="utf-8") as file:
        return Prd312AcceptanceScreeningPlan.model_validate(json.load(file))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of one artifact.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
