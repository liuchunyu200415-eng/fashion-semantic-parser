"""Tests for offline localization candidate filtering and evaluation scope."""

import json
from pathlib import Path

from scripts.evaluate_localization_predictions import (
    _evaluation_image_ids,
    _filter_predictions_per_image,
    _json_safe,
    _summary_path,
)


class _FakeCOCO:
    def getImgIds(self, *, catIds: list[int]) -> list[int]:
        assert catIds == [7]
        return [10, 20, 30]


def test_filter_predictions_applies_score_then_top_k_per_image() -> None:
    """Top-K must be independent per image and ordered by confidence."""
    predictions = [
        {"image_id": 20, "score": 0.7},
        {"image_id": 10, "score": 0.6},
        {"image_id": 10, "score": 0.9},
        {"image_id": 20, "score": 0.4},
    ]

    filtered = _filter_predictions_per_image(
        predictions,
        score_threshold=0.5,
        top_k=1,
    )

    assert filtered == [
        {"image_id": 10, "score": 0.9},
        {"image_id": 20, "score": 0.7},
    ]


def test_evaluation_uses_run_image_ids_including_misses(tmp_path: Path) -> None:
    """A run summary must preserve evaluated images that had no predictions."""
    summary_path = tmp_path / "collar_summary.json"
    summary_path.write_text(
        json.dumps({"image_ids": [10, 30]}),
        encoding="utf-8",
    )

    image_ids = _evaluation_image_ids(
        _FakeCOCO(),
        category_id=7,
        summary_path=summary_path,
    )

    assert image_ids == [10, 30]


def test_evaluation_defaults_to_all_category_images(tmp_path: Path) -> None:
    """Without a run summary, full category ground truth remains evaluable."""
    image_ids = _evaluation_image_ids(
        _FakeCOCO(),
        category_id=7,
        summary_path=tmp_path / "missing.json",
    )

    assert image_ids == [10, 20, 30]


def test_summary_path_and_json_safety() -> None:
    """Output helpers should be deterministic and standards-compliant."""
    prediction_path = Path("outputs/collar.json")

    assert _summary_path(prediction_path) == Path("outputs/collar_summary.json")
    assert _json_safe({"value": float("nan")}) == {"value": None}
