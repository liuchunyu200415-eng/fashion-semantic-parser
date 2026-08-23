"""Human-review workflow for the independent PRD 3.1.2 acceptance set."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, field_validator, model_validator

from fashion_semantic_parser.dao.localization.prd_312_acceptance import (
    AcceptanceAnnotationProvenance,
    AcceptanceLanguage,
    AcceptanceTargetCardinality,
    AcceptanceTargetRegion,
    Prd312AcceptanceCase,
    Prd312AcceptanceContract,
    Prd312AcceptanceManifest,
    acceptance_contract_blockers,
)
from fashion_semantic_parser.dao.localization.referring_smoke import (
    ReferringQueryDimension,
    ReferringQueryNovelty,
    ReferringReferenceFrame,
    ReferringSmokeTarget,
)

ReviewStatus = Literal["pending", "reviewed", "rejected"]
AnnotationRequirement = Literal[
    "independent_exact_mask_review",
    "independent_manual_mask",
]


class Prd312AcceptanceReviewRecord(BaseModel):
    """One quota slot plus the evidence required to approve it."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    language: AcceptanceLanguage
    primary_dimension: ReferringQueryDimension
    dimensions: list[ReferringQueryDimension] = Field(min_length=1)
    novelty: ReferringQueryNovelty
    target_cardinality: AcceptanceTargetCardinality
    target_region: AcceptanceTargetRegion
    target_label: str = Field(min_length=1)
    reference_frame: ReferringReferenceFrame | None = None
    annotation_requirement: AnnotationRequirement
    review_status: ReviewStatus = "pending"
    rejection_reason: str | None = None
    source_dataset: str | None = None
    source_record_id: str | None = None
    image_path: str | None = None
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    query: str | None = None
    annotation_provenance: AcceptanceAnnotationProvenance | None = None
    targets: list[ReferringSmokeTarget] = Field(default_factory=list)
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    notes: str | None = None

    @field_validator(
        "target_label",
        "rejection_reason",
        "source_dataset",
        "source_record_id",
        "image_path",
        "query",
        "reviewed_by",
        "notes",
    )
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        """Normalize optional review text and reject whitespace-only values.

        Args:
            value: Optional source text.

        Returns:
            Normalized text or ``None``.

        Raises:
            ValueError: If a provided value contains only whitespace.
        """
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Acceptance review text cannot be whitespace-only.")
        return normalized

    @model_validator(mode="after")
    def validate_review_record(self) -> "Prd312AcceptanceReviewRecord":
        """Keep planned semantics and reviewed evidence mutually consistent.

        Returns:
            The validated review record.

        Raises:
            ValueError: If slot semantics or reviewed evidence are inconsistent.
        """
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("Acceptance review dimensions cannot contain duplicates.")
        if "basic" not in self.dimensions:
            raise ValueError("Every review slot requires the basic target tag.")
        if self.primary_dimension not in self.dimensions:
            raise ValueError("primary_dimension must also appear in dimensions.")
        if ("spatial" in self.dimensions) != (self.reference_frame is not None):
            raise ValueError(
                "Spatial review slots require exactly one explicit reference frame."
            )
        if self.review_status == "rejected" and self.rejection_reason is None:
            raise ValueError("Rejected review records require a rejection_reason.")
        if self.review_status == "reviewed":
            self._validate_reviewed_evidence()
        return self

    def _validate_reviewed_evidence(self) -> None:
        """Require complete independent Mask evidence for one approved record."""
        required = {
            "source_dataset": self.source_dataset,
            "source_record_id": self.source_record_id,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "query": self.query,
            "annotation_provenance": self.annotation_provenance,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                "Reviewed acceptance record is missing evidence: " + ", ".join(missing)
            )
        if self.reviewed_at is not None and self.reviewed_at.utcoffset() is None:
            raise ValueError("Acceptance review timestamps must include a timezone.")
        if self.target_cardinality == "single_target" and len(self.targets) != 1:
            raise ValueError("A single-target review record requires exactly one Mask.")
        if self.target_cardinality == "multi_target" and len(self.targets) < 2:
            raise ValueError(
                "A multi-target review record requires at least two Masks."
            )
        if any(target.segmentation is None for target in self.targets):
            raise ValueError("Reviewed acceptance targets require Mask segmentations.")
        if any(
            target.label is not None and target.label != self.target_label
            for target in self.targets
        ):
            raise ValueError("Reviewed target labels must match the planned label.")

    def to_acceptance_case(self) -> Prd312AcceptanceCase:
        """Convert one approved record into the immutable final-case schema.

        Returns:
            A fully validated final acceptance case.

        Raises:
            ValueError: If the record has not passed human review.
        """
        if self.review_status != "reviewed":
            raise ValueError(f"Acceptance record is not reviewed: {self.id}")
        return Prd312AcceptanceCase.model_validate(
            {
                "id": self.id,
                "source_dataset": self.source_dataset,
                "source_partition": "acceptance_holdout",
                "source_record_id": self.source_record_id,
                "image_path": self.image_path,
                "image_sha256": self.image_sha256,
                "query": self.query,
                "language": self.language,
                "primary_dimension": self.primary_dimension,
                "dimensions": self.dimensions,
                "novelty": self.novelty,
                "target_region": self.target_region,
                "reference_frame": self.reference_frame,
                "target_label": self.target_label,
                "targets": [target.model_dump(mode="json") for target in self.targets],
                "annotation_provenance": self.annotation_provenance,
                "reviewed_by": self.reviewed_by,
                "reviewed_at": self.reviewed_at,
            }
        )


class Prd312AcceptanceReviewPlan(BaseModel):
    """Contract-sized annotation plan kept separate from training data."""

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str | None = None
    acceptance_partition: Literal["acceptance_holdout"] = "acceptance_holdout"
    excluded_from_training: Literal[True] = True
    excluded_from_model_selection: Literal[True] = True
    excluded_from_threshold_tuning: Literal[True] = True
    independence_attested_by: str | None = None
    independence_attested_at: datetime | None = None
    contract: Prd312AcceptanceContract
    records: list[Prd312AcceptanceReviewRecord] = Field(min_length=1)

    @field_validator("independence_attested_by")
    @classmethod
    def normalize_optional_attestation(cls, value: str | None) -> str | None:
        """Normalize an optional independence reviewer.

        Args:
            value: Optional reviewer identity.

        Returns:
            Normalized reviewer identity or ``None``.

        Raises:
            ValueError: If a provided identity contains only whitespace.
        """
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Independence reviewer cannot be whitespace-only.")
        return normalized

    @model_validator(mode="after")
    def validate_plan(self) -> "Prd312AcceptanceReviewPlan":
        """Reject duplicate slots and any drift from the locked composition.

        Returns:
            The validated review plan.

        Raises:
            ValueError: If attestation, IDs, or locked quotas are inconsistent.
        """
        if acceptance_contract_blockers(self.contract):
            raise ValueError("Acceptance review plan requires a locked contract.")
        if (self.independence_attested_by is None) != (
            self.independence_attested_at is None
        ):
            raise ValueError("Independence attestation name and time must be paired.")
        if (
            self.independence_attested_at is not None
            and self.independence_attested_at.utcoffset() is None
        ):
            raise ValueError("Independence attestation must include a timezone.")
        ids = [record.id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Acceptance review record IDs must be unique.")
        blockers = review_plan_composition_blockers(self)
        if blockers:
            raise ValueError(
                "Acceptance review composition differs: " + "; ".join(blockers)
            )
        return self


def review_plan_composition_blockers(
    plan: Prd312AcceptanceReviewPlan,
) -> list[str]:
    """Return quota drift without treating pending review as an error.

    Args:
        plan: Acceptance review plan to inspect.

    Returns:
        Human-readable composition blockers.
    """
    contract = plan.contract
    blockers: list[str] = []
    if len(plan.records) != contract.required_case_count:
        blockers.append("record count")
    axes = (
        (
            "primary_dimension",
            contract.primary_dimension_case_counts,
            Counter(record.primary_dimension for record in plan.records),
        ),
        (
            "novelty",
            contract.novelty_case_counts,
            Counter(r.novelty for r in plan.records),
        ),
        (
            "language",
            contract.language_case_counts,
            Counter(record.language for record in plan.records),
        ),
        (
            "target_cardinality",
            contract.target_cardinality_case_counts,
            Counter(record.target_cardinality for record in plan.records),
        ),
        (
            "target_region",
            contract.target_region_case_counts,
            Counter(record.target_region for record in plan.records),
        ),
    )
    for name, expected, actual in axes:
        if dict(expected) != dict(actual):
            blockers.append(name)
    composite_count = sum(
        len(set(record.dimensions) - {"basic"}) >= 2 for record in plan.records
    )
    if composite_count < cast(int, contract.minimum_composite_case_count):
        blockers.append("composite minimum")
    label_counts = Counter(record.target_label for record in plan.records)
    for label, minimum in contract.minimum_target_label_case_counts.items():
        if label_counts[label] < minimum:
            blockers.append(f"weak label {label}")
    return blockers


def acceptance_review_blockers(plan: Prd312AcceptanceReviewPlan) -> list[str]:
    """Return every item that prevents final manifest publication.

    Args:
        plan: Acceptance review plan to inspect.

    Returns:
        Human-readable review and attestation blockers.
    """
    blockers = review_plan_composition_blockers(plan)
    statuses = Counter(record.review_status for record in plan.records)
    if statuses["pending"]:
        blockers.append(f"pending records={statuses['pending']}")
    if statuses["rejected"]:
        blockers.append(f"rejected records={statuses['rejected']}")
    if plan.independence_attested_by is None:
        blockers.append("independence attestation is missing")
    return blockers


def finalize_prd_312_acceptance_manifest(
    plan: Prd312AcceptanceReviewPlan,
) -> Prd312AcceptanceManifest:
    """Publish a final manifest after review and independence gates pass.

    Args:
        plan: Fully reviewed and independently attested plan.

    Returns:
        Immutable final acceptance manifest.

    Raises:
        ValueError: If any review or independence blocker remains.
    """
    blockers = acceptance_review_blockers(plan)
    if blockers:
        raise ValueError("Acceptance review is incomplete: " + "; ".join(blockers))
    return Prd312AcceptanceManifest(
        name="prd_312_acceptance_v1",
        description="Independent manually reviewed PRD 3.1.2 acceptance benchmark.",
        independence_attested_by=cast(str, plan.independence_attested_by),
        independence_attested_at=cast(datetime, plan.independence_attested_at),
        contract=plan.contract,
        cases=[record.to_acceptance_case() for record in plan.records],
    )


def load_prd_312_acceptance_review_plan(path: Path) -> Prd312AcceptanceReviewPlan:
    """Read and fully validate one UTF-8 acceptance review plan.

    Args:
        path: Review-plan JSON path.

    Returns:
        Validated acceptance review plan.
    """
    with path.open("r", encoding="utf-8") as file:
        return Prd312AcceptanceReviewPlan.model_validate(json.load(file))


def write_model_json_atomic(path: Path, model: BaseModel) -> None:
    """Atomically publish one pretty UTF-8 Pydantic JSON document.

    Args:
        path: Destination JSON path.
        model: Pydantic model to serialize.

    Raises:
        OSError: If the temporary file cannot be written or published.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    payload = json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2)
    try:
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
