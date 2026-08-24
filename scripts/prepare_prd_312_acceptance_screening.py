"""Create the PRD 3.1.2 acceptance image-screening work package."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_INVENTORY = (
    "data/benchmarks/localization/prd_312_acceptance_holdout_candidates_v1.json"
)
DEFAULT_REVIEW_PLAN = "data/benchmarks/localization/prd_312_acceptance_review_v1.json"
DEFAULT_OUTPUT = "data/benchmarks/localization/prd_312_acceptance_screening_v1.json"
DEFAULT_CSV = "outputs/localization/prd_312_acceptance_screening_v1.csv"
DEFAULT_SUMMARY = "outputs/localization/prd_312_acceptance_screening_summary.json"


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse frozen inputs and screening artifact destinations.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create a model-independent human image-screening work package."
    )
    parser.add_argument("--inventory", default=DEFAULT_INVENTORY)
    parser.add_argument("--review-plan", default=DEFAULT_REVIEW_PLAN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", default=DEFAULT_CSV)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing pending template; never use after human editing.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and publish a pending JSON plan and editable CSV template.

    Raises:
        FileExistsError: If an output exists without explicit overwrite consent.
    """
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization import (
        prd_312_acceptance_screening_csv as screening_csv,
    )
    from fashion_semantic_parser.dao.localization.prd_312_acceptance_holdout import (
        Prd312AcceptanceHoldoutInventory,
    )
    from fashion_semantic_parser.dao.localization.prd_312_acceptance_review import (
        load_prd_312_acceptance_review_plan,
        write_model_json_atomic,
    )
    from fashion_semantic_parser.dao.localization.prd_312_acceptance_screening import (
        Prd312AcceptanceScreeningSources,
        build_prd_312_acceptance_screening_plan,
        sha256_file,
        summarize_prd_312_acceptance_screening,
    )

    inventory_path = resolve_project_path(args.inventory)
    review_path = resolve_project_path(args.review_plan)
    output_path = resolve_project_path(args.output)
    csv_path = resolve_project_path(args.csv_output)
    summary_path = resolve_project_path(args.summary_output)
    if not args.overwrite:
        existing = [path for path in (output_path, csv_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite screening work: "
                + ", ".join(str(path) for path in existing)
            )
    inventory = Prd312AcceptanceHoldoutInventory.model_validate_json(
        inventory_path.read_text(encoding="utf-8")
    )
    review_plan = load_prd_312_acceptance_review_plan(review_path)
    project_root = Path(__file__).resolve().parents[1]
    plan = build_prd_312_acceptance_screening_plan(
        inventory=inventory,
        review_plan=review_plan,
        sources=Prd312AcceptanceScreeningSources(
            candidate_inventory_path=inventory_path.relative_to(
                project_root
            ).as_posix(),
            candidate_inventory_sha256=sha256_file(inventory_path),
            review_plan_path=review_path.relative_to(project_root).as_posix(),
            review_plan_sha256=sha256_file(review_path),
        ),
        generated_at=datetime.now(timezone.utc),
    )
    summary = summarize_prd_312_acceptance_screening(plan)
    write_model_json_atomic(output_path, plan)
    screening_csv.write_prd_312_acceptance_screening_csv(csv_path, plan)
    write_model_json_atomic(summary_path, summary)
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print(f"screening_plan_path: {output_path}")
    print(f"screening_csv_path: {csv_path}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
