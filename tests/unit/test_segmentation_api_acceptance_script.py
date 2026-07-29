"""Tests for repeatable segmentation query API acceptance."""

import pytest

from scripts.accept_segmentation_api import (
    build_acceptance_report,
    select_high_confidence_cases,
    summarize_query_response,
)


def test_acceptance_selects_highest_score_for_every_category() -> None:
    """Saved predictions should provide one deterministic case per category."""
    validation = {
        "images": [
            {"id": 1, "file_name": "data/one.jpg"},
            {"id": 2, "file_name": "data/two.jpg"},
        ],
        "categories": [
            {"id": 1, "name": "top"},
            {"id": 7, "name": "bag"},
        ],
    }
    predictions = [
        {"image_id": 1, "category_id": 1, "score": 0.8},
        {"image_id": 2, "category_id": 1, "score": 0.9},
        {"image_id": 1, "category_id": 7, "score": 0.7},
    ]

    cases = select_high_confidence_cases(
        validation,
        predictions,
        score_threshold=0.6,
    )

    assert [(case["expected_category"], case["image_id"]) for case in cases] == [
        ("top", 2),
        ("bag", 1),
    ]


def test_acceptance_rejects_categories_without_saved_predictions() -> None:
    """Every validation category needs a deployment-threshold API case."""
    validation = {
        "images": [{"id": 1, "file_name": "data/one.jpg"}],
        "categories": [
            {"id": 1, "name": "top"},
            {"id": 7, "name": "bag"},
        ],
    }

    with pytest.raises(ValueError, match="bag"):
        select_high_confidence_cases(
            validation,
            [{"image_id": 1, "category_id": 1, "score": 0.9}],
            score_threshold=0.6,
        )


def test_acceptance_response_checks_roi_masks_boxes_and_expected_class() -> None:
    """One valid API response should satisfy every per-request check."""
    case = {
        "category_id": 7,
        "expected_category": "bag",
        "image_id": 10,
        "image_path": "data/example.jpg",
        "selected_score": 0.91,
    }
    response = {
        "segmentation": {
            "subject_roi_source": "detected",
            "subject_roi": {
                "x_min": 1,
                "y_min": 2,
                "x_max": 100,
                "y_max": 200,
            },
            "instances": [
                {
                    "category_label": "bag",
                    "mask": [[1, 2, 10, 2, 10, 20]],
                    "box": {
                        "x_min": 1,
                        "y_min": 2,
                        "x_max": 10,
                        "y_max": 20,
                    },
                }
            ],
        }
    }

    row = summarize_query_response(case, response)

    assert row["expected_detected"] is True
    assert row["subject_roi_source"] == "detected"
    assert row["subject_roi_present"] is True
    assert row["all_masks_present"] is True
    assert row["all_boxes_valid"] is True


def test_acceptance_report_requires_every_check() -> None:
    """A category miss should fail the aggregate acceptance decision."""
    valid_row = {
        "expected_detected": True,
        "subject_roi_source": "detected",
        "subject_roi_present": True,
        "all_masks_present": True,
        "all_boxes_valid": True,
        "detected_categories": ["top"],
    }
    missed_row = {**valid_row, "expected_detected": False}

    report = build_acceptance_report(
        base_url="http://127.0.0.1:8000",
        validation_json="validation.json",
        predictions_json="predictions.json",
        score_threshold=0.6,
        rows=[valid_row, missed_row],
    )

    assert report["accepted"] is False
    assert report["all_expected_detected"] is False
    assert report["all_subject_rois_detected"] is True
