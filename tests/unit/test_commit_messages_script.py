"""Tests for Conventional Commits validation."""

from scripts.check_commit_messages import invalid_commit_subjects


def test_conventional_commit_subjects_pass() -> None:
    """Supported types, scopes, and breaking markers should pass."""
    subjects = [
        "feat(localization): add spatial reranking",
        "fix!: change localization response contract",
        "ci: enforce repository quality gates",
    ]

    assert invalid_commit_subjects(subjects) == []


def test_non_conventional_commit_subjects_fail() -> None:
    """Unstructured or malformed subjects should be rejected."""
    subjects = ["update files", "Fix: wrong capitalization", "feat missing colon"]

    assert invalid_commit_subjects(subjects) == subjects
