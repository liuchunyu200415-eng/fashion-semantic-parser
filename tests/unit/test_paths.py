"""Tests for project-relative path helpers."""

from pathlib import Path

import pytest

from fashion_semantic_parser.common.paths import PROJECT_ROOT, resolve_project_path


def test_resolve_project_path_returns_path_under_project_root() -> None:
    """Project-relative paths should resolve under the current project root."""
    resolved_path = resolve_project_path("configs/app.yaml")

    assert resolved_path == PROJECT_ROOT / "configs/app.yaml"


def test_resolve_project_path_rejects_absolute_path() -> None:
    """Absolute paths should be rejected to keep project configuration portable."""
    with pytest.raises(ValueError):
        resolve_project_path(Path("/tmp/example.jpg"))

