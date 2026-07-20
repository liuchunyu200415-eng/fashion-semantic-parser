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

    resolved_path = (PROJECT_ROOT / path).resolve()
    try:
        resolved_path.relative_to(PROJECT_ROOT.resolve())
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
