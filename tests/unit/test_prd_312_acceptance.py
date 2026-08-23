"""Tests for the locked PRD 3.1.2 query-level acceptance contract."""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from fashion_semantic_parser.dao.localization.prd_312_acceptance import (
    Prd312AcceptanceContract,
    Prd312AcceptanceManifest,
    acceptance_contract_blockers,
)
from scripts.evaluate_prd_312_acceptance import (
    _binary_mask_iou,
    _passes_mask_iou,
    build_acceptance_report,
    evaluate_acceptance_case,
)


def _locked_contract() -> dict[str, object]:
    """Return one minimal complete product-approved contract payload."""
    return {
        "schema_version": 1,
        "status": "locked",
        "result_policy": "single_query_top1",
        "primary_metric": "mask_iou",
        "success_comparison": "strictly_greater_than",
        "mask_iou_threshold": 0.5,
        "required_accuracy": 0.92,
        "multi_target_policy": "union_mask",
        "required_case_count": 4,
        "primary_dimension_case_counts": {
            "basic": 1,
            "spatial": 1,
            "attribute": 1,
            "relation": 1,
        },
        "novelty_case_counts": {"seen": 2, "novel_composition": 2},
        "language_case_counts": {"zh": 2, "en": 2},
        "target_cardinality_case_counts": {
            "single_target": 4,
            "multi_target": 0,
        },
        "target_region_case_counts": {
            "collar": 1,
            "cuff": 1,
            "hem": 1,
            "pocket": 1,
            "shoulder": 0,
            "waist": 0,
            "pattern": 0,
            "decoration": 0,
        },
        "minimum_composite_case_count": 0,
        "minimum_target_label_case_counts": {
            "zipper": 0,
            "rivet": 0,
            "neckline": 0,
            "pocket": 0,
        },
        "approval_basis": "owner_signoff",
        "product_owner_approval": "product-owner",
        "project_owner_approval": "project-owner",
        "approved_at": "2026-08-14T12:00:00+08:00",
    }


def _manifest() -> Prd312AcceptanceManifest:
    """Build a four-axis locked manifest with deterministic square Masks."""
    cases = []
    dimensions = ("basic", "spatial", "attribute", "relation")
    target_regions = ("collar", "cuff", "hem", "pocket")
    for index, primary_dimension in enumerate(dimensions):
        case_dimensions = ["basic"]
        if primary_dimension != "basic":
            case_dimensions.append(primary_dimension)
        reference_frame = None
        if primary_dimension == "spatial":
            reference_frame = "image"
        cases.append(
            {
                "id": f"case_{index}",
                "source_dataset": "independent-test-set",
                "source_partition": "acceptance_holdout",
                "source_record_id": f"source-{index}",
                "image_path": f"data/image_{index}.jpg",
                "image_sha256": f"{index:064x}",
                "query": f"query {index}",
                "language": "zh" if index < 2 else "en",
                "primary_dimension": primary_dimension,
                "dimensions": case_dimensions,
                "novelty": "seen" if index % 2 == 0 else "novel_composition",
                "target_region": target_regions[index],
                "reference_frame": reference_frame,
                "target_label": "sleeve",
                "targets": [{"segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]]}],
                "annotation_provenance": "independent_manual_mask",
                "reviewed_by": "reviewer",
                "reviewed_at": "2026-08-24T09:00:00+08:00",
            }
        )
    return Prd312AcceptanceManifest.model_validate(
        {
            "schema_version": 2,
            "name": "prd_312_acceptance_v1",
            "independence_attested_by": "benchmark-owner",
            "independence_attested_at": "2026-08-24T10:00:00+08:00",
            "contract": _locked_contract(),
            "cases": cases,
        }
    )


def test_draft_contract_reports_every_external_decision_blocker() -> None:
    """A code-ready draft must not be mistaken for a locked PRD benchmark."""
    contract = Prd312AcceptanceContract()

    blockers = acceptance_contract_blockers(contract)

    assert "status is not locked" in blockers
    assert "multi_target_policy is not confirmed" in blockers
    assert "all four primary query-type counts are not confirmed" in blockers
    assert "approval basis is missing" in blockers


def test_user_directive_can_lock_contract_without_owner_names() -> None:
    """A direct decision is recorded without fabricating external signatories."""
    payload = _locked_contract()
    payload["approval_basis"] = "user_directive"
    payload["product_owner_approval"] = None
    payload["project_owner_approval"] = None

    contract = Prd312AcceptanceContract.model_validate(payload)

    assert not acceptance_contract_blockers(contract)


def test_locked_contract_requires_every_primary_query_type() -> None:
    """A locked mix cannot silently omit a mentor-required query dimension."""
    payload = _locked_contract()
    payload["primary_dimension_case_counts"] = {
        "basic": 2,
        "spatial": 1,
        "attribute": 1,
        "relation": 0,
    }

    with pytest.raises(ValidationError, match="non-zero coverage"):
        Prd312AcceptanceContract.model_validate(payload)


def test_manifest_enforces_locked_composition_counts() -> None:
    """The actual cases must exactly match the approved mutually exclusive mix."""
    manifest = _manifest()
    assert len(manifest.cases) == 4

    payload = manifest.model_dump(mode="json")
    payload["cases"][0]["language"] = "en"
    with pytest.raises(ValidationError, match="language counts differ"):
        Prd312AcceptanceManifest.model_validate(payload)


def test_manifest_enforces_target_mix_and_minimums() -> None:
    """Target cardinality, PRD region, composite, and weak labels are guarded."""
    manifest = _manifest()

    region_payload = manifest.model_dump(mode="json")
    region_payload["cases"][0]["target_region"] = "cuff"
    with pytest.raises(ValidationError, match="target_region counts differ"):
        Prd312AcceptanceManifest.model_validate(region_payload)

    composite_payload = manifest.model_dump(mode="json")
    composite_payload["contract"]["minimum_composite_case_count"] = 1
    with pytest.raises(ValidationError, match="composite case count"):
        Prd312AcceptanceManifest.model_validate(composite_payload)

    label_payload = manifest.model_dump(mode="json")
    label_payload["contract"]["minimum_target_label_case_counts"]["zipper"] = 1
    with pytest.raises(ValidationError, match="target-label count"):
        Prd312AcceptanceManifest.model_validate(label_payload)


def test_iou_boundary_is_strictly_greater_than_half() -> None:
    """Exactly 0.50 fails while any finite value above it succeeds."""
    target = np.asarray([[True, True], [False, False]])
    prediction = np.asarray([[True, False], [False, False]])
    iou = _binary_mask_iou(prediction, target)

    assert iou == 0.5
    assert _passes_mask_iou(iou, 0.5) is False
    assert _passes_mask_iou(0.500001, 0.5) is True


def test_acceptance_uses_only_top1_and_unions_multi_target_gt() -> None:
    """A correct second candidate cannot rescue Top-1; GT instances form one Mask."""
    case = SimpleNamespace(
        id="two_sleeves",
        query="这件衣服的袖子",
        language="zh",
        primary_dimension="basic",
        dimensions=["basic"],
        novelty="seen",
        target_region="cuff",
        reference_frame=None,
        target_label="sleeve",
        targets=[
            SimpleNamespace(segmentation=[[1, 1, 3, 1, 3, 3, 1, 3]]),
            SimpleNamespace(segmentation=[[6, 1, 8, 1, 8, 3, 6, 3]]),
        ],
    )
    correct_union = [
        [1, 1, 3, 1, 3, 3, 1, 3],
        [6, 1, 8, 1, 8, 3, 6, 3],
    ]
    row = evaluate_acceptance_case(
        case=case,
        response={
            "case_id": case.id,
            "query": case.query,
            "regions": [
                {"mask": [[1, 6, 3, 6, 3, 8, 1, 8]]},
                {"mask": correct_union},
            ],
        },
        image_shape=(10, 10),
    )
    exact = evaluate_acceptance_case(
        case=case,
        response={
            "case_id": case.id,
            "query": case.query,
            "regions": [{"mask": correct_union}],
        },
        image_shape=(10, 10),
    )

    assert row["prediction_count"] == 2
    assert row["query_passed"] is False
    assert exact["top1_mask_iou_percent"] == 100.0
    assert exact["query_passed"] is True


def test_inference_error_remains_a_failed_query_in_accuracy_denominator() -> None:
    """Runtime failures are scored as misses instead of disappearing from accuracy."""
    case = _manifest().cases[0]
    row = evaluate_acceptance_case(
        case=case,
        response={
            "case_id": case.id,
            "query": case.query,
            "regions": [],
            "error": {"type": "RuntimeError", "message": "failed"},
        },
        image_shape=(10, 10),
    )

    report = build_acceptance_report(
        manifest_path="data/acceptance.json",
        manifest=_manifest(),
        responses_dir="outputs/responses",
        rows=[row],
    )

    assert row["query_passed"] is False
    assert report["overall"]["query_count"] == 1
    assert report["overall"]["query_accuracy_percent"] == 0.0
    assert report["prd_accuracy_passed"] is False
