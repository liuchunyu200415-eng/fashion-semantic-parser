"""Validate human screening CSV decisions against their frozen worklist."""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_PLAN = "data/benchmarks/localization/prd_312_acceptance_screening_v1.json"
DEFAULT_CSV = "outputs/localization/prd_312_acceptance_screening_v1.csv"
DEFAULT_OUTPUT = (
    "data/benchmarks/localization/prd_312_acceptance_screening_reviewed_v1.json"
)
DEFAULT_SUMMARY = (
    "outputs/localization/prd_312_acceptance_screening_reviewed_summary.json"
)


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse the frozen screening plan, edited CSV, and outputs.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Import and audit human PRD 3.1.2 image-screening decisions."
    )
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    """Import the edited CSV and publish validated progress artifacts."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization import (
        prd_312_acceptance_screening_csv as screening_csv,
    )
    from fashion_semantic_parser.dao.localization.prd_312_acceptance_review import (
        write_model_json_atomic,
    )
    from fashion_semantic_parser.dao.localization.prd_312_acceptance_screening import (
        load_prd_312_acceptance_screening_plan,
        summarize_prd_312_acceptance_screening,
    )

    plan_path = resolve_project_path(args.plan)
    csv_path = resolve_project_path(args.csv)
    output_path = resolve_project_path(args.output)
    summary_path = resolve_project_path(args.summary_output)
    plan = load_prd_312_acceptance_screening_plan(plan_path)
    reviewed_plan = screening_csv.import_prd_312_acceptance_screening_csv(
        plan,
        csv_path,
    )
    summary = summarize_prd_312_acceptance_screening(reviewed_plan)
    write_model_json_atomic(output_path, reviewed_plan)
    write_model_json_atomic(summary_path, summary)
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print(f"reviewed_screening_plan_path: {output_path}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
