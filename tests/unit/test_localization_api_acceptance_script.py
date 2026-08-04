"""Tests for repeatable PRD 3.1.2 query API acceptance."""

from scripts.accept_localization_api import (
    build_acceptance_cases,
    build_acceptance_report,
    summarize_query_response,
)


def test_acceptance_builds_four_direct_and_four_derived_cases() -> None:
    """The manifest should cover every required PRD localization region."""
    validation = {
        "images": [
            {"id": 1, "file_name": "data/one.jpg"},
            {"id": 2, "file_name": "data/two.jpg"},
        ],
        "categories": [
            {"id": 1, "name": "collar"},
            {"id": 2, "name": "pocket"},
            {"id": 3, "name": "epaulette"},
            {"id": 4, "name": "ruffle"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 1, "area": 10},
            {"image_id": 2, "category_id": 1, "area": 20},
            {"image_id": 1, "category_id": 2, "area": 30},
            {"image_id": 1, "category_id": 3, "area": 40},
            {"image_id": 1, "category_id": 4, "area": 50},
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
    assert cases[2]["expected_labels"] == ["epaulette", "shoulder"]
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
