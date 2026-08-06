"""Tests for PRD 3.1.2 warm service latency helpers."""

import json
from pathlib import Path

import pytest

from scripts.benchmark_localization_latency import (
    _latency_summary,
    _resolve_benchmark_images,
)


def test_latency_summary_reports_p95_and_request_rate() -> None:
    """Latency output must expose the target's tail statistic."""
    summary = _latency_summary([10.0, 20.0, 30.0, 40.0])

    assert summary["mean"] == 25.0
    assert summary["median"] == 25.0
    assert summary["p95"] == pytest.approx(38.5)
    assert summary["requests_per_second_from_mean"] == 40.0


def test_validation_images_are_selected_deterministically(tmp_path: Path) -> None:
    """Repeated benchmarks should time the same validation image order."""
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 20, "file_name": "second.jpg"},
                    {"id": 10, "file_name": "first.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = _resolve_benchmark_images(
        explicit_images=[],
        validation_path=validation_path,
        image_limit=1,
    )

    assert paths == ["first.jpg"]
