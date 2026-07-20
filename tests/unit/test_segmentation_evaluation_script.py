"""Tests for segmentation evaluation CLI output helpers."""

import json
from pathlib import Path

from scripts.evaluate_segmentation_baseline import _write_metrics_output


def test_write_metrics_output_creates_clean_json_file(tmp_path: Path) -> None:
    """Metrics should remain machine-readable when framework logs are redirected."""
    output_path = tmp_path / "nested" / "metrics.json"
    metrics = {"segm": {"AP": 12.5}}

    _write_metrics_output(
        output_path,
        json.dumps(metrics, ensure_ascii=False, indent=2),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == metrics
