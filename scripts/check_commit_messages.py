"""Validate Git commit subjects against the Conventional Commits convention."""

import argparse
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVENTIONAL_SUBJECT = re.compile(
    r"^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(\([a-z0-9][a-z0-9._/-]*\))?(!)?: .+"
)


def parse_args() -> argparse.Namespace:
    """Parse the Git revision range to inspect.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Require Conventional Commits subjects in a Git range."
    )
    parser.add_argument(
        "--range",
        dest="commit_range",
        required=True,
        help="Git revision range such as origin/main..HEAD or HEAD^..HEAD.",
    )
    return parser.parse_args()


def read_commit_subjects(commit_range: str) -> list[str]:
    """Read commit subjects from one Git revision range.

    Args:
        commit_range: Revision range accepted by ``git log``.

    Returns:
        Commit subjects ordered according to ``git log``.

    Raises:
        RuntimeError: If Git cannot resolve or inspect the range.
    """
    result = subprocess.run(
        ["git", "log", "--format=%s", commit_range],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or "unknown git error"
        raise RuntimeError(f"Cannot inspect commit range {commit_range}: {detail}")
    return [subject for subject in result.stdout.splitlines() if subject]


def invalid_commit_subjects(subjects: list[str]) -> list[str]:
    """Return subjects that do not follow Conventional Commits.

    Args:
        subjects: Commit subjects to validate.

    Returns:
        Invalid subjects in their original order.
    """
    return [
        subject
        for subject in subjects
        if CONVENTIONAL_SUBJECT.fullmatch(subject) is None
    ]


def main() -> None:
    """Validate all subjects in the requested Git range.

    Raises:
        SystemExit: If the range is empty or contains an invalid subject.
    """
    args = parse_args()
    subjects = read_commit_subjects(args.commit_range)
    if not subjects:
        raise SystemExit(f"No commits found in range: {args.commit_range}")
    invalid = invalid_commit_subjects(subjects)
    if invalid:
        details = "\n".join(f"- {subject}" for subject in invalid)
        raise SystemExit(
            f"Non-conventional commit subjects in {args.commit_range}:\n{details}"
        )
    print(f"Validated {len(subjects)} Conventional Commit subject(s).")


if __name__ == "__main__":
    main()
