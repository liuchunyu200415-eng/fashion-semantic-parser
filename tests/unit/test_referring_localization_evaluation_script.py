"""Tests for query-level open-language localization evaluation."""

import json

import pytest

from fashion_semantic_parser.dao.localization.referring_smoke import (
    ReferringSmokeManifest,
)
from scripts.evaluate_referring_localization import (
    _box_iou,
    build_referring_report,
    evaluate_referring_case,
    summarize_referring_rows,
)


def _case(
    *,
    case_id: str = "left_cuff",
    annotation_status: str = "box",
    expected_count: int | None = 1,
    targets: list[dict[str, object]] | None = None,
    dimensions: list[str] | None = None,
    novelty: str = "novel_composition",
    contrast_set_id: str | None = "cuff_variants",
):
    if targets is None:
        targets = [{"box": {"x_min": 1, "y_min": 1, "x_max": 5, "y_max": 5}}]
    payload = {
        "schema_version": 1,
        "name": "test smoke",
        "cases": [
            {
                "id": case_id,
                "image_path": "data/image.jpg",
                "query": "衣服左边的袖口",
                "grounding_prompt": "the cuff on the left side",
                "dimensions": dimensions or ["basic", "spatial"],
                "novelty": novelty,
                "reference_frame": "image",
                "annotation_status": annotation_status,
                "expected_count": expected_count,
                "targets": targets,
                "contrast_set_id": contrast_set_id,
            }
        ],
    }
    return ReferringSmokeManifest.model_validate(payload).cases[0]


def _region(
    x_min: float = 1,
    y_min: float = 1,
    x_max: float = 5,
    y_max: float = 5,
    *,
    mask: list[list[float]] | None = None,
) -> dict[str, object]:
    return {
        "box": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        "mask": mask or [],
    }


def _response(case_id: str, regions: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_id": case_id,
        "regions": regions,
        "elapsed_seconds": 0.2,
    }


def test_box_iou_handles_exact_and_disjoint_boxes() -> None:
    """Box-only annotations need a deterministic independent metric."""
    assert _box_iou((1, 1, 5, 5), (1, 1, 5, 5)) == pytest.approx(1.0)
    assert _box_iou((1, 1, 2, 2), (5, 5, 6, 6)) == 0.0


def test_exact_box_target_passes_but_extra_candidate_fails_exact_set() -> None:
    """Target recall and clean query-level selection must remain separate."""
    case = _case()
    exact = evaluate_referring_case(
        case=case,
        response=_response(case.id, [_region()]),
        image_shape=(10, 10),
    )
    extra = evaluate_referring_case(
        case=case,
        response=_response(case.id, [_region(), _region(6, 6, 9, 9)]),
        image_shape=(10, 10),
    )

    assert exact["target_recall_passed"] is True
    assert exact["query_passed"] is True
    assert extra["target_recall_passed"] is True
    assert extra["expected_count_passed"] is False
    assert extra["query_passed"] is False
    assert extra["precision50_percent"] == 50.0


def test_one_prediction_cannot_match_two_targets() -> None:
    """Multi-target expressions require one-to-one instance matching."""
    case = _case(
        expected_count=2,
        targets=[
            {"box": {"x_min": 1, "y_min": 1, "x_max": 5, "y_max": 5}},
            {"box": {"x_min": 6, "y_min": 1, "x_max": 9, "y_max": 5}},
        ],
    )

    row = evaluate_referring_case(
        case=case,
        response=_response(case.id, [_region()]),
        image_shape=(10, 10),
    )

    assert row["matched_count"] == 1
    assert row["recall50_percent"] == 50.0
    assert row["query_passed"] is False


def test_mask_label_uses_mask_iou_and_retains_empty_prediction_as_miss() -> None:
    """Mask scope must score pixels and keep zero-prediction cases in recall."""
    polygon = [[1, 1, 5, 1, 5, 5, 1, 5]]
    case = _case(
        annotation_status="mask",
        targets=[{"segmentation": polygon}],
    )

    matched = evaluate_referring_case(
        case=case,
        response=_response(case.id, [_region(mask=polygon)]),
        image_shape=(10, 10),
    )
    missed = evaluate_referring_case(
        case=case,
        response=_response(case.id, []),
        image_shape=(10, 10),
    )

    assert matched["metric"] == "mask_iou"
    assert matched["matched_ious_percent"] == [100.0]
    assert missed["scored"] is True
    assert missed["matched_count"] == 0
    assert missed["recall50_percent"] == 0.0
    assert missed["precision50_percent"] is None
    assert missed["query_passed"] is False


def test_negative_and_unlabelled_cases_use_separate_denominators() -> None:
    """A reviewed absence is scored; missing annotation is not accuracy evidence."""
    negative = _case(
        case_id="negative",
        annotation_status="negative",
        expected_count=None,
        targets=[],
        dimensions=["attribute"],
        contrast_set_id=None,
    )
    unlabelled = _case(
        case_id="unlabelled",
        annotation_status="unlabelled",
        expected_count=None,
        targets=[],
        dimensions=["relation"],
        contrast_set_id=None,
    )
    negative_row = evaluate_referring_case(
        case=negative,
        response=_response(negative.id, []),
        image_shape=(10, 10),
    )
    unlabelled_row = evaluate_referring_case(
        case=unlabelled,
        response=_response(unlabelled.id, [_region()]),
        image_shape=(10, 10),
    )
    summary = summarize_referring_rows([negative_row, unlabelled_row])

    assert negative_row["query_passed"] is True
    assert unlabelled_row["scored"] is False
    assert unlabelled_row["query_passed"] is None
    assert summary["scored_case_count"] == 1
    assert summary["unlabelled_case_count"] == 1
    assert summary["query_success_rate_percent"] == 100.0
    assert summary["positive_match_counts"]["recall50_percent"] is None


def test_report_groups_combined_dimensions_and_keeps_iou_scopes_separate() -> None:
    """One combined phrase contributes to both modifiers without mixing IoUs."""
    mask_polygon = [[1, 1, 5, 1, 5, 5, 1, 5]]
    mask_case = _case(
        case_id="floral_left_cuff",
        annotation_status="mask",
        targets=[{"segmentation": mask_polygon}],
        dimensions=["spatial", "attribute"],
    )
    box_case = _case(
        case_id="inner_garment",
        dimensions=["relation"],
        contrast_set_id=None,
    )
    rows = [
        evaluate_referring_case(
            case=mask_case,
            response=_response(mask_case.id, [_region(mask=mask_polygon)]),
            image_shape=(10, 10),
        ),
        evaluate_referring_case(
            case=box_case,
            response=_response(box_case.id, [_region()]),
            image_shape=(10, 10),
        ),
    ]

    report = build_referring_report(
        manifest_path="data/benchmark.json",
        manifest_name="test",
        responses_dir="outputs/responses",
        rows=rows,
        min_iou=0.5,
    )

    assert report["by_dimension"]["spatial"]["case_count"] == 1
    assert report["by_dimension"]["attribute"]["case_count"] == 1
    assert report["by_dimension"]["relation"]["case_count"] == 1
    assert report["overall"]["mask_instance_metrics"]["eligible_case_count"] == 1
    assert report["overall"]["box_instance_metrics"]["eligible_case_count"] == 1
    assert report["accuracy_boundary"]["prd_accuracy_passed"] is None
    json.dumps(report, allow_nan=False)
