"""Tests for manifest-driven open-language localization prediction."""

from types import SimpleNamespace

import pytest

from scripts.predict_referring_localization import (
    _build_settings_overrides,
    _latency_summary,
    build_case_response,
)


class _Prediction:
    def model_dump(self, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "image_path": "data/image.jpg",
            "query": "衣服左边的袖口",
            "regions": [],
        }


def test_case_response_preserves_full_query_and_grounding_prompt() -> None:
    """Saved evidence must show the user text and actual model text separately."""
    case = SimpleNamespace(
        id="left_cuff_001",
        grounding_prompt="the cuff on the left side of the garment",
        dimensions=["basic", "spatial"],
        novelty="novel_composition",
        reference_frame="image",
        annotation_status="box",
        expected_count=1,
        contrast_set_id="image_001_cuffs",
    )

    response = build_case_response(
        case=case,
        prediction=_Prediction(),
        elapsed_seconds=0.25,
        includes_model_load=True,
    )

    assert response["query"] == "衣服左边的袖口"
    assert response["grounding_prompt"] == ("the cuff on the left side of the garment")
    assert response["dimensions"] == ["basic", "spatial"]
    assert response["contrast_set_id"] == "image_001_cuffs"
    assert response["includes_model_load"] is True


def test_referring_settings_only_include_explicit_valid_overrides() -> None:
    """Smoke experiments must not mutate the committed deployment profile."""
    assert _build_settings_overrides(
        roi_mode="auto",
        box_threshold=0.15,
        text_threshold=None,
        max_regions=10,
        subject_roi_margin=0.35,
    ) == {
        "box_threshold": 0.15,
        "max_regions": 10,
        "subject_roi_margin": 0.35,
    }

    with pytest.raises(ValueError, match="full ROI mode"):
        _build_settings_overrides(
            roi_mode="full",
            box_threshold=None,
            text_threshold=None,
            max_regions=None,
            subject_roi_margin=0.35,
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        _build_settings_overrides(
            roi_mode="auto",
            box_threshold=1.1,
            text_threshold=None,
            max_regions=None,
            subject_roi_margin=None,
        )


def test_latency_summary_separates_cold_first_case_from_warm_cases() -> None:
    """Cold model loading must not be compared with the warm latency target."""
    summary = _latency_summary([7.0, 0.2, 0.4])

    assert summary["first_case_including_model_load"] == 7.0
    assert summary["warm_case_count"] == 2
    assert summary["warm_mean"] == pytest.approx(0.3)
    assert summary["all_wall_clock_mean"] == pytest.approx(7.6 / 3)
