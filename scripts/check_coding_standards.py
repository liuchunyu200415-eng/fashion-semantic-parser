"""Enforce project coding rules that are not covered by standard linters."""

import argparse
import ast
import json
import tokenize
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "configs" / "coding_standard_baseline.json"
SOURCE_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts")
TEST_ROOT = PROJECT_ROOT / "tests"


@dataclass(frozen=True)
class CodingStandardMetrics:
    """Counts of enforceable violations and gradual-refactoring debt."""

    missing_module_docstrings: int = 0
    missing_public_docstrings: int = 0
    google_docstring_section_gaps: int = 0
    oversized_modules: int = 0
    functions_over_five_parameters: int = 0
    complex_any_dict_annotations: int = 0
    implicit_string_concatenations: int = 0
    mutable_default_arguments: int = 0
    bare_except_handlers: int = 0
    relative_imports: int = 0
    wildcard_imports: int = 0
    test_print_calls: int = 0
    tab_characters: int = 0
    invalid_eof_newlines: int = 0


def parse_args() -> argparse.Namespace:
    """Parse the optional debt-baseline path.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Check company-specific Python coding-standard rules."
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    return parser.parse_args()


def analyze_python_files(
    source_files: list[Path],
    test_files: list[Path],
) -> CodingStandardMetrics:
    """Measure strict violations and bounded legacy debt.

    Args:
        source_files: Application and script Python files.
        test_files: Unit and integration test Python files.

    Returns:
        Aggregate coding-standard metrics.

    Raises:
        ValueError: If one inspected file is not valid Python.
    """
    values = {name: 0 for name in CodingStandardMetrics.__dataclass_fields__}
    test_set = set(test_files)
    for path in [*source_files, *test_files]:
        data = path.read_bytes()
        text = data.decode("utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as error:
            raise ValueError(f"Cannot audit invalid Python: {path}") from error
        values["tab_characters"] += text.count("\t")
        values["invalid_eof_newlines"] += int(
            not data.endswith(b"\n") or data.endswith(b"\n\n")
        )
        values["missing_module_docstrings"] += int(ast.get_docstring(tree) is None)
        values[
            "implicit_string_concatenations"
        ] += _implicit_string_concatenation_count(data)
        if path in test_set:
            values["test_print_calls"] += _test_print_call_count(tree)
        else:
            values["oversized_modules"] += int(len(text.splitlines()) > 500)
            values["complex_any_dict_annotations"] += text.count("dict[str, Any]")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = _function_parameters(node)
                values["functions_over_five_parameters"] += int(len(parameters) > 5)
                values["mutable_default_arguments"] += int(_has_mutable_default(node))
            if isinstance(node, ast.ExceptHandler):
                values["bare_except_handlers"] += int(node.type is None)
            if isinstance(node, ast.ImportFrom):
                values["relative_imports"] += int(node.level > 0)
                values["wildcard_imports"] += sum(
                    alias.name == "*" for alias in node.names
                )
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _add_public_docstring_metrics(node, values)
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_") and ast.get_docstring(node) is None:
                    values["missing_public_docstrings"] += 1
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _add_public_docstring_metrics(child, values)
    return CodingStandardMetrics(**values)


def validate_against_baseline(
    metrics: CodingStandardMetrics,
    baseline: dict[str, int],
) -> list[str]:
    """Return every metric that exceeds its allowed debt budget.

    Args:
        metrics: Current repository metrics.
        baseline: Maximum allowed value for every metric.

    Returns:
        Human-readable violations; an empty list means the audit passed.
    """
    violations = []
    for name, current in asdict(metrics).items():
        allowed = baseline.get(name)
        if allowed is None:
            violations.append(f"Baseline is missing metric: {name}")
        elif current > allowed:
            violations.append(f"{name}: current={current} allowed={allowed}")
    return violations


def main() -> None:
    """Audit the repository and fail when strict rules or debt budgets regress.

    Raises:
        SystemExit: If the baseline is invalid or any metric regresses.
    """
    args = parse_args()
    baseline_path = args.baseline.resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or not all(
        isinstance(name, str) and isinstance(value, int)
        for name, value in baseline.items()
    ):
        raise SystemExit(f"Invalid coding-standard baseline: {baseline_path}")
    source_files = sorted(path for root in SOURCE_ROOTS for path in root.rglob("*.py"))
    test_files = sorted(TEST_ROOT.rglob("*.py"))
    metrics = analyze_python_files(source_files, test_files)
    violations = validate_against_baseline(metrics, baseline)
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit("Coding-standard regression:\n" + "\n".join(violations))


def _add_public_docstring_metrics(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    values: dict[str, int],
) -> None:
    """Record missing public docs and applicable Google-style sections."""
    if node.name.startswith("_"):
        return
    docstring = ast.get_docstring(node, clean=False)
    if docstring is None:
        values["missing_public_docstrings"] += 1
        return
    parameters = _function_parameters(node)
    if parameters and "Args:" not in docstring:
        values["google_docstring_section_gaps"] += 1
    returns_none = isinstance(node.returns, ast.Constant) and node.returns.value is None
    if node.returns is not None and not returns_none and "Returns:" not in docstring:
        values["google_docstring_section_gaps"] += 1
    if any(isinstance(child, ast.Raise) for child in ast.walk(node)):
        if "Raises:" not in docstring:
            values["google_docstring_section_gaps"] += 1


def _function_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.arg]:
    """Return user-visible parameters excluding conventional receivers."""
    parameters = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    return [
        parameter for parameter in parameters if parameter.arg not in {"self", "cls"}
    ]


def _has_mutable_default(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function uses a mutable literal default argument."""
    defaults = [
        *node.args.defaults,
        *(value for value in node.args.kw_defaults if value is not None),
    ]
    return any(isinstance(value, (ast.List, ast.Dict, ast.Set)) for value in defaults)


def _test_print_call_count(tree: ast.AST) -> int:
    """Count direct print calls forbidden in automated tests."""
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        for node in ast.walk(tree)
    )


def _implicit_string_concatenation_count(data: bytes) -> int:
    """Count adjacent string tokens joined implicitly inside one statement."""
    previous_was_string = False
    count = 0
    ignored = {
        tokenize.ENCODING,
        tokenize.NL,
        tokenize.COMMENT,
        tokenize.INDENT,
        tokenize.DEDENT,
    }
    for token in tokenize.tokenize(BytesIO(data).readline):
        if token.type == tokenize.STRING:
            count += int(previous_was_string)
            previous_was_string = True
        elif token.type == tokenize.NEWLINE:
            previous_was_string = False
        elif token.type not in ignored:
            previous_was_string = False
    return count


if __name__ == "__main__":
    main()
