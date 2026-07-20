"""Relative path helpers for project resources."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(relative_path: str | Path) -> Path:
    """Resolve a project-relative path to an absolute filesystem path.

    Args:
        relative_path: Path relative to the project root.

    Returns:
        Absolute path under the current project root.

    Raises:
        ValueError: If an absolute path is provided.
    """
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError("Only project-relative paths are allowed.")

    project_root = Path(os.path.abspath(PROJECT_ROOT))
    resolved_path = Path(os.path.abspath(project_root / path))
    try:
        resolved_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            "Project-relative paths cannot leave the project root."
        ) from error
    return resolved_path


def to_project_relative_path(path: str | Path) -> str:
    """Convert an absolute project path to a project-relative POSIX string.

    Args:
        path: Absolute or relative filesystem path.

    Returns:
        Project-relative path string when possible.
    """
    project_path = Path(path)
    if not project_path.is_absolute():
        return project_path.as_posix()

    try:
        return project_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return project_path.as_posix()
