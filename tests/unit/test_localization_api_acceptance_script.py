"""Tests for repeatable PRD 3.1.2 query API acceptance."""

from scripts.accept_localization_api import (
    _best_direct_mask_iou,
    build_acceptance_cases,
    build_acceptance_report,
    summarize_query_response,
)


def test_direct_mask_iou_rejects_a_large_wrong_region() -> None:
    """A label match must not hide a mask that mostly covers the wrong area."""
    ground_truth = {
        "image_width": 100,
        "image_height": 100,
        "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
    }
    matching_regions = [
        {"mask": [[0, 0, 99, 0, 99, 99, 0, 99]]},
    ]

    iou = _best_direct_mask_iou(matching_regions, ground_truth)

    assert iou is not None
    assert iou < 0.10


def test_direct_mask_iou_accepts_matching_polygon() -> None:
    """A direct prediction matching the selected GT should clear IoU 0.50."""
    polygon = [10, 10, 30, 10, 30, 30, 10, 30]
    ground_truth = {
        "image_width": 100,
        "image_height": 100,
        "segmentation": [polygon],
    }

    iou = _best_direct_mask_iou([{"mask": [polygon]}], ground_truth)

    assert iou == 1.0


def test_acceptance_builds_four_direct_and_four_derived_cases() -> None:
    """The manifest should cover every required PRD localization region."""
    validation = {
        "images": [
            {"id": 1, "file_name": "data/one.jpg", "width": 100, "height": 120},
            {"id": 2, "file_name": "data/two.jpg", "width": 100, "height": 120},
        ],
        "categories": [
            {"id": 1, "name": "collar"},
            {"id": 2, "name": "pocket"},
            {"id": 3, "name": "epaulette"},
            {"id": 4, "name": "ruffle"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "area": 10},
            {"id": 2, "image_id": 2, "category_id": 1, "area": 20},
            {"id": 3, "image_id": 1, "category_id": 2, "area": 30},
            {"id": 4, "image_id": 1, "category_id": 3, "area": 40},
            {"id": 5, "image_id": 1, "category_id": 4, "area": 50},
        ],
    }

    cases = build_acceptance_cases(validation, "data/derived.jpg")

    assert [case["target_region"] for case in cases] == [
        "collar",
        "pocket",
        "shoulder",
        "decoration",
        "cuff",
        "hem",
        "waist",
        "pattern",
    ]
    assert cases[0]["image_id"] == 2
    assert cases[0]["ground_truth"]["annotation_id"] == 2
    assert cases[0]["ground_truth_role"] == "exact"
    assert cases[2]["expected_labels"] == ["epaulette", "shoulder"]
    assert cases[2]["ground_truth_role"] == "partial_reference"
    assert cases[-1]["expected_source_contains"] == "derived from top appearance"


def test_acceptance_response_checks_source_roi_masks_boxes_and_segmentation() -> None:
    """One valid derived result should satisfy all per-request checks."""
    case = {
        "target_region": "hem",
        "query": "下摆在哪里？",
        "image_id": None,
        "image_path": "data/example.jpg",
        "expected_labels": ["hem"],
        "expected_source_contains": "derived from top mask",
        "case_source": "derived_region_image",
    }
    response = {
        "segmentation": {"instances": []},
        "localization": {
            "subject_roi_source": "detected",
            "subject_roi": {"x_min": 1, "y_min": 2, "x_max": 90, "y_max": 100},
            "regions": [
                {
                    "region_label": "hem",
                    "matched_text": "下摆 derived from top mask",
                    "mask": [[1, 2, 10, 2, 10, 5]],
                    "box": {"x_min": 1, "y_min": 2, "x_max": 10, "y_max": 5},
                }
            ],
        },
    }

    row = summarize_query_response(case, response, elapsed_seconds=1.25)

    assert row["expected_detected"] is True
    assert row["source_matched"] is True
    assert row["all_masks_present"] is True
    assert row["all_boxes_valid"] is True
    assert row["segmentation_present"] is True
    assert row["elapsed_seconds"] == 1.25


def test_direct_quality_failure_rejects_the_aggregate_report() -> None:
    """A direct label hit with a spatially wrong mask must fail acceptance."""
    case = {
        "target_region": "decoration",
        "query": "荷叶边在哪里？",
        "image_id": 1,
        "image_path": "data/example.jpg",
        "expected_labels": ["ruffle"],
        "expected_source_contains": None,
        "case_source": "largest_ruffle_annotation",
        "ground_truth_role": "exact",
        "ground_truth": {
            "annotation_id": 1,
            "category_label": "ruffle",
            "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
            "bbox": [10, 10, 20, 20],
            "image_width": 100,
            "image_height": 100,
        },
    }
    response = {
        "segmentation": {"instances": []},
        "localization": {
            "subject_roi_source": "detected",
            "subject_roi": {"x_min": 0, "y_min": 0, "x_max": 99, "y_max": 99},
            "regions": [
                {
                    "region_label": "ruffle",
                    "matched_text": "荷叶边",
                    "mask": [[0, 0, 99, 0, 99, 99, 0, 99]],
                    "box": {"x_min": 0, "y_min": 0, "x_max": 99, "y_max": 99},
                }
            ],
        },
    }

    row = summarize_query_response(case, response, elapsed_seconds=1.0)
    report = build_acceptance_report(
        base_url="http://127.0.0.1:8002",
        validation_json="validation.json",
        derived_image="data/derived.jpg",
        rows=[row],
    )

    assert row["expected_detected"] is True
    assert row["quality_checked"] is True
    assert row["quality_passed"] is False
    assert row["best_mask_iou"] < 10.0
    assert report["all_direct_mask_iou_passed"] is False
    assert report["accepted"] is False


def test_partial_epaulette_reference_does_not_score_full_shoulder_iou() -> None:
    """An epaulette mask is a visual reference, not full-shoulder ground truth."""
    case = {
        "target_region": "shoulder",
        "query": "肩部在哪里？",
        "image_id": 1,
        "image_path": "data/example.jpg",
        "expected_labels": ["epaulette", "shoulder"],
        "expected_source_contains": None,
        "case_source": "largest_epaulette_annotation",
        "ground_truth_role": "partial_reference",
        "ground_truth": {
            "annotation_id": 1,
            "category_label": "epaulette",
            "segmentation": [[10, 10, 20, 10, 20, 15, 10, 15]],
            "bbox": [10, 10, 10, 5],
            "image_width": 100,
            "image_height": 100,
        },
    }
    response = {
        "segmentation": {"instances": []},
        "localization": {
            "subject_roi_source": "detected",
            "subject_roi": {"x_min": 0, "y_min": 0, "x_max": 99, "y_max": 99},
            "regions": [
                {
                    "region_label": "shoulder",
                    "matched_text": "肩部 derived from outerwear mask",
                    "mask": [[5, 5, 30, 5, 30, 25, 5, 25]],
                    "box": {"x_min": 5, "y_min": 5, "x_max": 30, "y_max": 25},
                }
            ],
        },
    }

    row = summarize_query_response(case, response, elapsed_seconds=1.0)

    assert row["expected_detected"] is True
    assert row["quality_checked"] is False
    assert row["quality_passed"] is True
    assert row["best_mask_iou"] is None


def test_acceptance_report_requires_every_functional_check() -> None:
    """A source mismatch should fail the aggregate acceptance decision."""
    valid_row = {
        "expected_detected": True,
        "source_matched": True,
        "subject_roi_source": "detected",
        "subject_roi_present": True,
        "segmentation_present": True,
        "all_masks_present": True,
        "all_boxes_valid": True,
        "elapsed_seconds": 1.0,
    }
    mismatched_row = {**valid_row, "source_matched": False}

    report = build_acceptance_report(
        base_url="http://127.0.0.1:8002",
        validation_json="validation.json",
        derived_image="data/derived.jpg",
        rows=[valid_row, mismatched_row],
    )

    assert report["accepted"] is False
    assert report["all_expected_detected"] is True
    assert report["all_sources_matched"] is False


def test_acceptance_allows_explicit_full_image_roi_fallback() -> None:
    """Missing person detections remain valid when fallback is explicit."""
    row = {
        "expected_detected": True,
        "source_matched": True,
        "subject_roi_source": "full_image_fallback",
        "subject_roi_present": False,
        "segmentation_present": True,
        "all_masks_present": True,
        "all_boxes_valid": True,
        "elapsed_seconds": 1.0,
    }

    report = build_acceptance_report(
        base_url="http://127.0.0.1:8002",
        validation_json="validation.json",
        derived_image="data/derived.jpg",
        rows=[row],
    )

    assert report["accepted"] is True
    assert report["all_roi_modes_valid"] is True
    assert report["all_subject_rois_detected"] is False
    assert report["roi_source_counts"] == {"full_image_fallback": 1}


def test_acceptance_rejects_inconsistent_detected_roi_state() -> None:
    """A detected source without a person box should fail functional acceptance."""
    row = {
        "expected_detected": True,
        "source_matched": True,
        "subject_roi_source": "detected",
        "subject_roi_present": False,
        "segmentation_present": True,
        "all_masks_present": True,
        "all_boxes_valid": True,
        "elapsed_seconds": 1.0,
    }

    report = build_acceptance_report(
        base_url="http://127.0.0.1:8002",
        validation_json="validation.json",
        derived_image="data/derived.jpg",
        rows=[row],
    )

    assert report["accepted"] is False
    assert report["all_roi_modes_valid"] is False
