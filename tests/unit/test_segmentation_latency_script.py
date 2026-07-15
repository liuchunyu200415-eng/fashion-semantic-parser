"""Tests for the PRD 3.1.1 latency benchmark CLI helpers."""

import json
from pathlib import Path

from scripts.benchmark_segmentation_latency import _resolve_benchmark_image_paths


def test_benchmark_images_are_selected_deterministically(tmp_path: Path) -> None:
    """Validation samples should be ordered by COCO image id before limiting."""
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 2, "file_name": "images/second.jpg"},
                    {"id": 1, "file_name": "images/first.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = _resolve_benchmark_image_paths(
        explicit_images=[],
        val_json=str(validation_path),
        image_root=".",
        image_limit=1,
        resolve_path=lambda path: _resolve_test_path(tmp_path, path),
    )

    assert paths == [tmp_path / "images/first.jpg"]


def test_explicit_benchmark_images_override_validation_json(tmp_path: Path) -> None:
    """Explicit image paths should not require opening a validation file."""
    paths = _resolve_benchmark_image_paths(
        explicit_images=["images/example.jpg"],
        val_json="missing.json",
        image_root=".",
        image_limit=0,
        resolve_path=lambda path: _resolve_test_path(tmp_path, path),
    )

    assert paths == [tmp_path / "images/example.jpg"]


def _resolve_test_path(root: Path, path: str | Path) -> Path:
    """Resolve test paths with the same absolute-path behavior as the project."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate
