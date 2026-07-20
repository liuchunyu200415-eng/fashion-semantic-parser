"""Tests for project-relative path helpers."""

from pathlib import Path
from typing import Any

import pytest

import fashion_semantic_parser.common.paths as paths_module
from fashion_semantic_parser.common.paths import PROJECT_ROOT, resolve_project_path


def test_resolve_project_path_returns_path_under_project_root() -> None:
    """Project-relative paths should resolve under the current project root."""
    resolved_path = resolve_project_path("configs/app.yaml")

    assert resolved_path == PROJECT_ROOT / "configs/app.yaml"


def test_resolve_project_path_rejects_absolute_path() -> None:
    """Absolute paths should be rejected to keep project configuration portable."""
    with pytest.raises(ValueError):
        resolve_project_path(Path("/tmp/example.jpg"))


def test_resolve_project_path_rejects_parent_traversal() -> None:
    """API-facing project paths must not escape the repository root."""
    with pytest.raises(ValueError, match="cannot leave"):
        resolve_project_path("../private-image.jpg")


def test_resolve_project_path_allows_project_data_symlink(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Trusted project symlinks may point to mounted AutoDL data volumes."""
    project_root = tmp_path / "project"
    external_data = tmp_path / "autodl-data"
    project_root.mkdir()
    external_data.mkdir()
    image_path = external_data / "example.jpg"
    image_path.write_bytes(b"test-image")
    (project_root / "data").symlink_to(external_data, target_is_directory=True)
    monkeypatch.setattr(paths_module, "PROJECT_ROOT", project_root)

    resolved_path = paths_module.resolve_project_path("data/example.jpg")

    assert resolved_path == project_root / "data/example.jpg"
    assert resolved_path.is_file()
