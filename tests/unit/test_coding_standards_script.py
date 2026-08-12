"""Tests for repository-specific coding-standard enforcement."""

from pathlib import Path

from scripts.check_coding_standards import (
    CodingStandardMetrics,
    analyze_python_files,
    validate_against_baseline,
)


def test_audit_detects_strict_and_gradual_violations(tmp_path: Path) -> None:
    """One source fixture should exercise strict and debt-budget metrics."""
    source = tmp_path / "source.py"
    function_signature = (
        "def public(a: int, b: int, c: int, " + "d: int, e: int, f: int = []) -> str:"
    )
    source.write_text(
        "\n".join(
            [
                '"""Fixture module."""',
                function_signature,
                '    """Short documentation."""',
                '    return ("a" "b")',
                "",
            ]
        ),
        encoding="utf-8",
    )
    test = tmp_path / "test_source.py"
    test.write_text(
        "\n".join(
            [
                '"""Fixture tests."""',
                "def test_public() -> None:",
                '    """Exercise one scenario."""',
                "    print('forbidden')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    metrics = analyze_python_files([source], [test])

    assert metrics.functions_over_five_parameters == 1
    assert metrics.mutable_default_arguments == 1
    assert metrics.implicit_string_concatenations == 1
    assert metrics.google_docstring_section_gaps == 2
    assert metrics.test_print_calls == 1


def test_baseline_rejects_only_regressions() -> None:
    """Debt equal to its budget passes while an increase fails explicitly."""
    metrics = CodingStandardMetrics(oversized_modules=2)
    baseline = {name: value for name, value in CodingStandardMetrics().__dict__.items()}
    baseline["oversized_modules"] = 2

    assert validate_against_baseline(metrics, baseline) == []

    baseline["oversized_modules"] = 1
    assert validate_against_baseline(metrics, baseline) == [
        "oversized_modules: current=2 allowed=1"
    ]
