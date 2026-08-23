"""Locked acceptance contract for PRD 3.1.2 referring localization."""

import json
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, Field, field_validator, model_validator

from fashion_semantic_parser.dao.localization.referring_smoke import (
    ReferringQueryDimension,
    ReferringQueryNovelty,
    ReferringReferenceFrame,
    ReferringSmokeTarget,
)

AcceptanceLanguage = Literal["zh", "en"]
MultiTargetPolicy = Literal["union_mask", "exclude"]
AcceptanceApprovalBasis = Literal["owner_signoff", "user_directive"]
AcceptanceTargetCardinality = Literal["single_target", "multi_target"]
AcceptanceTargetRegion = Literal[
    "collar",
    "cuff",
    "hem",
    "pocket",
    "shoulder",
    "waist",
    "pattern",
    "decoration",
]
AcceptanceAnnotationProvenance = Literal[
    "independent_manual_mask",
    "independent_existing_mask_human_verified",
]
CountKey = TypeVar("CountKey", bound=str)
REQUIRED_PRIMARY_DIMENSIONS: tuple[ReferringQueryDimension, ...] = (
    "basic",
    "spatial",
    "attribute",
    "relation",
)
REQUIRED_TARGET_CARDINALITIES: tuple[AcceptanceTargetCardinality, ...] = (
    "single_target",
    "multi_target",
)
REQUIRED_TARGET_REGIONS: tuple[AcceptanceTargetRegion, ...] = (
    "collar",
    "cuff",
    "hem",
    "pocket",
    "shoulder",
    "waist",
    "pattern",
    "decoration",
)
REQUIRED_WEAK_TARGET_LABELS: tuple[str, ...] = (
    "zipper",
    "rivet",
    "neckline",
    "pocket",
)


class Prd312AcceptanceContract(BaseModel):
    """Recorded metric and benchmark composition decisions."""

    schema_version: Literal[1] = 1
    status: Literal["draft", "locked"] = "draft"
    result_policy: Literal["single_query_top1"] = "single_query_top1"
    primary_metric: Literal["mask_iou"] = "mask_iou"
    success_comparison: Literal["strictly_greater_than"] = "strictly_greater_than"
    mask_iou_threshold: float = 0.50
    required_accuracy: float = 0.92
    multi_target_policy: MultiTargetPolicy | None = None
    required_case_count: int | None = Field(default=None, ge=1)
    primary_dimension_case_counts: dict[ReferringQueryDimension, int] = Field(
        default_factory=dict
    )
    novelty_case_counts: dict[ReferringQueryNovelty, int] = Field(default_factory=dict)
    language_case_counts: dict[AcceptanceLanguage, int] = Field(default_factory=dict)
    target_cardinality_case_counts: dict[AcceptanceTargetCardinality, int] = Field(
        default_factory=dict
    )
    target_region_case_counts: dict[AcceptanceTargetRegion, int] = Field(
        default_factory=dict
    )
    minimum_composite_case_count: int | None = Field(default=None, ge=0)
    minimum_target_label_case_counts: dict[str, int] = Field(default_factory=dict)
    approval_basis: AcceptanceApprovalBasis | None = None
    product_owner_approval: str | None = None
    project_owner_approval: str | None = None
    approved_at: datetime | None = None

    @field_validator(
        "product_owner_approval",
        "project_owner_approval",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Normalize approval fields and reject whitespace-only values."""
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Acceptance approval fields cannot be whitespace-only.")
        return normalized

    @model_validator(mode="after")
    def validate_fixed_metric(self) -> "Prd312AcceptanceContract":
        """Keep the mentor-confirmed metric immutable across benchmark versions."""
        if self.mask_iou_threshold != 0.50:
            raise ValueError("PRD 3.1.2 Mask IoU threshold must remain 0.50.")
        if self.required_accuracy != 0.92:
            raise ValueError("PRD 3.1.2 required accuracy must remain 0.92.")
        for counts in (
            self.primary_dimension_case_counts,
            self.novelty_case_counts,
            self.language_case_counts,
            self.target_cardinality_case_counts,
            self.target_region_case_counts,
            self.minimum_target_label_case_counts,
        ):
            if any(count < 0 for count in counts.values()):
                raise ValueError("Acceptance case counts cannot be negative.")
        if self.status == "locked":
            blockers = acceptance_contract_blockers(self)
            if blockers:
                raise ValueError(
                    "Locked acceptance contract is incomplete: " + "; ".join(blockers)
                )
        return self


class Prd312AcceptanceCase(BaseModel):
    """One reviewed positive query with query-level target Masks."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_dataset: str = Field(min_length=1)
    source_partition: Literal["acceptance_holdout"] = "acceptance_holdout"
    source_record_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query: str = Field(min_length=1)
    language: AcceptanceLanguage
    primary_dimension: ReferringQueryDimension
    dimensions: list[ReferringQueryDimension] = Field(min_length=1)
    novelty: ReferringQueryNovelty
    target_region: AcceptanceTargetRegion
    reference_frame: ReferringReferenceFrame | None = None
    target_label: str = Field(min_length=1)
    targets: list[ReferringSmokeTarget] = Field(min_length=1)
    annotation_provenance: AcceptanceAnnotationProvenance
    reviewed_by: str = Field(min_length=1)
    reviewed_at: datetime

    @field_validator(
        "source_dataset",
        "source_record_id",
        "image_path",
        "query",
        "target_label",
        "reviewed_by",
    )
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        """Store stable non-empty text in the acceptance manifest."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Acceptance case text fields cannot be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_case_contract(self) -> "Prd312AcceptanceCase":
        """Require Mask GT and an explicit mutually exclusive primary group."""
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("Acceptance dimensions cannot contain duplicates.")
        if "basic" not in self.dimensions:
            raise ValueError("Every acceptance query requires the basic target tag.")
        if self.primary_dimension not in self.dimensions:
            raise ValueError("primary_dimension must also appear in dimensions.")
        if any(target.segmentation is None for target in self.targets):
            raise ValueError("Every acceptance target requires a Mask segmentation.")
        if "spatial" in self.dimensions and self.reference_frame is None:
            raise ValueError("Spatial acceptance cases require a reference_frame.")
        if self.reviewed_at.utcoffset() is None:
            raise ValueError("Acceptance review timestamps must include a timezone.")
        return self


class Prd312AcceptanceManifest(BaseModel):
    """Self-contained locked benchmark for final PRD accuracy evaluation."""

    schema_version: Literal[2] = 2
    name: str = Field(min_length=1)
    description: str | None = None
    acceptance_partition: Literal["acceptance_holdout"] = "acceptance_holdout"
    excluded_from_training: Literal[True] = True
    excluded_from_model_selection: Literal[True] = True
    excluded_from_threshold_tuning: Literal[True] = True
    independence_attested_by: str = Field(min_length=1)
    independence_attested_at: datetime
    contract: Prd312AcceptanceContract
    cases: list[Prd312AcceptanceCase] = Field(min_length=1)

    @field_validator("name", "independence_attested_by")
    @classmethod
    def normalize_manifest_text(cls, value: str) -> str:
        """Normalize required manifest-level text."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Acceptance manifest text cannot be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_manifest_contract(self) -> "Prd312AcceptanceManifest":
        """Reject duplicate IDs and benchmark composition drift."""
        if self.independence_attested_at.utcoffset() is None:
            raise ValueError("Independence attestation must include a timezone.")
        case_ids = [case.id for case in self.cases]
        duplicates = sorted(
            case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate acceptance case IDs: {duplicates}")
        if self.contract.status != "locked":
            raise ValueError("Acceptance manifest requires a locked contract.")
        if len(self.cases) != self.contract.required_case_count:
            raise ValueError("Acceptance case count does not match the contract.")
        _require_counts(
            expected=self.contract.primary_dimension_case_counts,
            actual=Counter(case.primary_dimension for case in self.cases),
            name="primary_dimension",
        )
        _require_counts(
            expected=self.contract.novelty_case_counts,
            actual=Counter(case.novelty for case in self.cases),
            name="novelty",
        )
        _require_counts(
            expected=self.contract.language_case_counts,
            actual=Counter(case.language for case in self.cases),
            name="language",
        )
        _require_counts(
            expected=self.contract.target_cardinality_case_counts,
            actual=Counter(
                "single_target" if len(case.targets) == 1 else "multi_target"
                for case in self.cases
            ),
            name="target_cardinality",
        )
        _require_counts(
            expected=self.contract.target_region_case_counts,
            actual=Counter(case.target_region for case in self.cases),
            name="target_region",
        )
        composite_count = sum(
            len(set(case.dimensions) - {"basic"}) >= 2 for case in self.cases
        )
        if composite_count < cast(int, self.contract.minimum_composite_case_count):
            raise ValueError(
                "Acceptance composite case count is below the locked minimum: "
                f"minimum={self.contract.minimum_composite_case_count}, "
                f"actual={composite_count}"
            )
        label_counts = Counter(case.target_label for case in self.cases)
        for label, minimum in self.contract.minimum_target_label_case_counts.items():
            if label_counts[label] < minimum:
                raise ValueError(
                    "Acceptance target-label count is below the locked minimum: "
                    f"label={label}, minimum={minimum}, actual={label_counts[label]}"
                )
        if self.contract.multi_target_policy == "exclude" and any(
            len(case.targets) != 1 for case in self.cases
        ):
            raise ValueError("Multi-target cases are excluded by the locked contract.")
        return self


def acceptance_contract_blockers(contract: Prd312AcceptanceContract) -> list[str]:
    """Return every unresolved decision that prevents final acceptance."""
    blockers: list[str] = []
    if contract.status != "locked":
        blockers.append("status is not locked")
    if contract.multi_target_policy is None:
        blockers.append("multi_target_policy is not confirmed")
    if contract.required_case_count is None:
        blockers.append("required_case_count is not confirmed")
    blockers.extend(_composition_blockers(contract))
    if contract.approval_basis is None:
        blockers.append("approval basis is missing")
    elif contract.approval_basis == "owner_signoff":
        if contract.product_owner_approval is None:
            blockers.append("product owner approval is missing")
        if contract.project_owner_approval is None:
            blockers.append("project owner approval is missing")
    if contract.approved_at is None:
        blockers.append("approval timestamp is missing")
    return blockers


def _composition_blockers(contract: Prd312AcceptanceContract) -> list[str]:
    """Return unresolved benchmark-composition decisions."""
    blockers: list[str] = []
    required_dimensions = set(REQUIRED_PRIMARY_DIMENSIONS)
    if set(contract.primary_dimension_case_counts) != required_dimensions:
        blockers.append("all four primary query-type counts are not confirmed")
    elif any(
        contract.primary_dimension_case_counts[dimension] == 0
        for dimension in REQUIRED_PRIMARY_DIMENSIONS
    ):
        blockers.append("all four primary query types require non-zero coverage")
    for name, counts in (
        ("primary_dimension", contract.primary_dimension_case_counts),
        ("novelty", contract.novelty_case_counts),
        ("language", contract.language_case_counts),
        ("target_cardinality", contract.target_cardinality_case_counts),
        ("target_region", contract.target_region_case_counts),
    ):
        if not counts:
            blockers.append(f"{name} counts are not confirmed")
        elif contract.required_case_count is not None and sum(counts.values()) != (
            contract.required_case_count
        ):
            blockers.append(f"{name} counts do not sum to required_case_count")
    if set(contract.target_cardinality_case_counts) != set(
        REQUIRED_TARGET_CARDINALITIES
    ):
        blockers.append("single-target and multi-target counts are not confirmed")
    if set(contract.target_region_case_counts) != set(REQUIRED_TARGET_REGIONS):
        blockers.append("all eight PRD target-region counts are not confirmed")
    if contract.minimum_composite_case_count is None:
        blockers.append("minimum composite-query count is not confirmed")
    elif (
        contract.required_case_count is not None
        and contract.minimum_composite_case_count > contract.required_case_count
    ):
        blockers.append("minimum composite-query count exceeds required_case_count")
    if set(contract.minimum_target_label_case_counts) != set(
        REQUIRED_WEAK_TARGET_LABELS
    ):
        blockers.append("all four weak-part minimum counts are not confirmed")
    return blockers


def load_prd_312_acceptance_contract(path: Path) -> Prd312AcceptanceContract:
    """Read one UTF-8 PRD acceptance contract."""
    with path.open("r", encoding="utf-8") as file:
        return cast(
            Prd312AcceptanceContract,
            Prd312AcceptanceContract.model_validate(json.load(file)),
        )


def load_prd_312_acceptance_manifest(path: Path) -> Prd312AcceptanceManifest:
    """Read and fully validate one locked acceptance manifest."""
    with path.open("r", encoding="utf-8") as file:
        return cast(
            Prd312AcceptanceManifest,
            Prd312AcceptanceManifest.model_validate(json.load(file)),
        )


def _require_counts(
    *, expected: Mapping[CountKey, int], actual: Counter[CountKey], name: str
) -> None:
    """Require exact benchmark composition for one mutually exclusive axis."""
    expected_nonzero = {key: value for key, value in expected.items() if value}
    actual_nonzero = {key: value for key, value in actual.items() if value}
    if dict(sorted(actual_nonzero.items())) != dict(sorted(expected_nonzero.items())):
        raise ValueError(
            f"Acceptance {name} counts differ from the locked contract: "
            f"expected={dict(sorted(expected_nonzero.items()))}, "
            f"actual={dict(sorted(actual_nonzero.items()))}"
        )
