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
CountKey = TypeVar("CountKey", bound=str)
REQUIRED_PRIMARY_DIMENSIONS: tuple[ReferringQueryDimension, ...] = (
    "basic",
    "spatial",
    "attribute",
    "relation",
)


class Prd312AcceptanceContract(BaseModel):
    """Product-approved metric and benchmark composition decisions."""

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
    image_path: str = Field(min_length=1)
    query: str = Field(min_length=1)
    language: AcceptanceLanguage
    primary_dimension: ReferringQueryDimension
    dimensions: list[ReferringQueryDimension] = Field(min_length=1)
    novelty: ReferringQueryNovelty
    reference_frame: ReferringReferenceFrame | None = None
    target_label: str = Field(min_length=1)
    targets: list[ReferringSmokeTarget] = Field(min_length=1)

    @field_validator("image_path", "query", "target_label")
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
        return self


class Prd312AcceptanceManifest(BaseModel):
    """Self-contained locked benchmark for final PRD accuracy evaluation."""

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str | None = None
    contract: Prd312AcceptanceContract
    cases: list[Prd312AcceptanceCase] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Normalize the versioned acceptance benchmark name."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Acceptance manifest name cannot be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_manifest_contract(self) -> "Prd312AcceptanceManifest":
        """Reject duplicate IDs and benchmark composition drift."""
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
    ):
        if not counts:
            blockers.append(f"{name} counts are not confirmed")
        elif contract.required_case_count is not None and sum(counts.values()) != (
            contract.required_case_count
        ):
            blockers.append(f"{name} counts do not sum to required_case_count")
    if contract.product_owner_approval is None:
        blockers.append("product owner approval is missing")
    if contract.project_owner_approval is None:
        blockers.append("project owner approval is missing")
    if contract.approved_at is None:
        blockers.append("approval timestamp is missing")
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
    if dict(sorted(actual.items())) != dict(sorted(expected.items())):
        raise ValueError(
            f"Acceptance {name} counts differ from the locked contract: "
            f"expected={dict(sorted(expected.items()))}, "
            f"actual={dict(sorted(actual.items()))}"
        )
