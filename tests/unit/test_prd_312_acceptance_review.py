"""Tests for independent PRD 3.1.2 acceptance-set human review."""

from collections import Counter

import pytest
from pydantic import ValidationError

from fashion_semantic_parser.dao.localization.prd_312_acceptance import (
    Prd312AcceptanceContract,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_planning import (
    build_prd_312_acceptance_review_plan,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_review import (
    Prd312AcceptanceReviewPlan,
    Prd312AcceptanceReviewRecord,
    acceptance_review_blockers,
    finalize_prd_312_acceptance_manifest,
)


def _full_contract() -> Prd312AcceptanceContract:
    """Return the repository's locked 1,000-case contract shape."""
    return Prd312AcceptanceContract.model_validate(
        {
            "status": "locked",
            "multi_target_policy": "union_mask",
            "required_case_count": 1000,
            "primary_dimension_case_counts": {
                "basic": 250,
                "spatial": 250,
                "attribute": 250,
                "relation": 250,
            },
            "novelty_case_counts": {"seen": 700, "novel_composition": 300},
            "language_case_counts": {"zh": 500, "en": 500},
            "target_cardinality_case_counts": {
                "single_target": 850,
                "multi_target": 150,
            },
            "target_region_case_counts": {
                "collar": 125,
                "cuff": 125,
                "hem": 125,
                "pocket": 125,
                "shoulder": 125,
                "waist": 125,
                "pattern": 125,
                "decoration": 125,
            },
            "minimum_composite_case_count": 200,
            "minimum_target_label_case_counts": {
                "zipper": 50,
                "rivet": 50,
                "neckline": 50,
                "pocket": 50,
            },
            "approval_basis": "user_directive",
            "approved_at": "2026-08-23T02:30:58+08:00",
        }
    )


def test_review_plan_matches_every_locked_quota_without_fake_evidence() -> None:
    """The generated plan is exact while every image and Mask remains pending."""
    plan = build_prd_312_acceptance_review_plan(_full_contract())

    assert len(plan.records) == 1000
    assert Counter(record.primary_dimension for record in plan.records) == {
        "basic": 250,
        "spatial": 250,
        "attribute": 250,
        "relation": 250,
    }
    assert Counter(record.target_region for record in plan.records) == {
        "collar": 125,
        "cuff": 125,
        "hem": 125,
        "pocket": 125,
        "shoulder": 125,
        "waist": 125,
        "pattern": 125,
        "decoration": 125,
    }
    assert (
        sum(len(set(record.dimensions) - {"basic"}) >= 2 for record in plan.records)
        == 200
    )
    assert all(
        record.novelty == "novel_composition"
        for record in plan.records
        if len(set(record.dimensions) - {"basic"}) >= 2
    )
    assert Counter(record.target_label for record in plan.records)["zipper"] == 50
    assert Counter(record.target_label for record in plan.records)["rivet"] == 50
    assert Counter(record.target_label for record in plan.records)["neckline"] == 50
    assert all(record.review_status == "pending" for record in plan.records)
    assert all(record.image_path is None for record in plan.records)
    assert all(not record.targets for record in plan.records)


def test_missing_fashionpedia_regions_require_independent_manual_masks() -> None:
    """Sleeves, belts, and epaulettes cannot silently stand in for PRD regions."""
    plan = build_prd_312_acceptance_review_plan(_full_contract())
    requirements = {
        record.target_region: record.annotation_requirement for record in plan.records
    }

    assert requirements["collar"] == "independent_exact_mask_review"
    assert requirements["pocket"] == "independent_exact_mask_review"
    assert requirements["decoration"] == "independent_exact_mask_review"
    for region in ("cuff", "hem", "shoulder", "waist", "pattern"):
        assert requirements[region] == "independent_manual_mask"


def test_reviewed_record_requires_traceable_mask_evidence() -> None:
    """Changing a pending slot to reviewed without evidence fails closed."""
    record = build_prd_312_acceptance_review_plan(_full_contract()).records[0]
    payload = record.model_dump(mode="json")
    payload["review_status"] = "reviewed"

    with pytest.raises(ValidationError, match="missing evidence"):
        Prd312AcceptanceReviewRecord.model_validate(payload)


def test_pending_plan_cannot_be_finalized() -> None:
    """Quota completion alone does not create formal acceptance evidence."""
    plan = build_prd_312_acceptance_review_plan(_full_contract())

    blockers = acceptance_review_blockers(plan)

    assert "pending records=1000" in blockers
    assert "independence attestation is missing" in blockers
    with pytest.raises(ValueError, match="Acceptance review is incomplete"):
        finalize_prd_312_acceptance_manifest(plan)


def test_complete_independent_review_finalizes_schema_version_two() -> None:
    """Only complete reviewed Masks and an attestation publish the benchmark."""
    payload = build_prd_312_acceptance_review_plan(_full_contract()).model_dump(
        mode="json"
    )
    payload["independence_attested_by"] = "benchmark-owner"
    payload["independence_attested_at"] = "2026-08-24T10:00:00+08:00"
    for index, record in enumerate(payload["records"]):
        record.update(
            {
                "review_status": "reviewed",
                "source_dataset": "independent-test-set",
                "source_record_id": f"source-{index}",
                "image_path": f"data/acceptance/{index}.jpg",
                "image_sha256": f"{index:064x}",
                "query": f"acceptance query {index}",
                "annotation_provenance": "independent_manual_mask",
                "reviewed_by": "reviewer",
                "reviewed_at": "2026-08-24T09:00:00+08:00",
            }
        )
        target_count = 1 if record["target_cardinality"] == "single_target" else 2
        record["targets"] = [
            {
                "label": record["target_label"],
                "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
            }
            for _ in range(target_count)
        ]
    plan = Prd312AcceptanceReviewPlan.model_validate(payload)

    manifest = finalize_prd_312_acceptance_manifest(plan)

    assert manifest.schema_version == 2
    assert len(manifest.cases) == 1000
    assert manifest.excluded_from_training is True
    assert manifest.cases[0].source_partition == "acceptance_holdout"


def test_plan_rejects_composition_drift() -> None:
    """Editing a slot away from the locked distribution invalidates the file."""
    plan = build_prd_312_acceptance_review_plan(_full_contract())
    payload = plan.model_dump(mode="json")
    payload["records"][0]["language"] = (
        "en" if payload["records"][0]["language"] == "zh" else "zh"
    )

    with pytest.raises(ValidationError, match="composition differs"):
        Prd312AcceptanceReviewPlan.model_validate(payload)
