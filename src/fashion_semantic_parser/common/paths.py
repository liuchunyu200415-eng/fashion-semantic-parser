"""Relative path helpers for project resources."""

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
    return PROJECT_ROOT / path

