"""Finalize a reviewed PRD 3.1.2 plan into the immutable benchmark."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_REVIEW_PLAN = "data/benchmarks/localization/prd_312_acceptance_review_v1.json"
DEFAULT_OUTPUT = "data/benchmarks/localization/prd_312_acceptance_v1.json"


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse reviewed-plan and final-manifest paths.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=" ".join(
            (
                "Fail closed unless all acceptance slots are reviewed and the",
                "independent holdout is attested.",
            )
        )
    )
    parser.add_argument("--review-plan", default=DEFAULT_REVIEW_PLAN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Audit and publish the final manifest.

    Raises:
        SystemExit: If any review or independence blocker remains.
    """
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.prd_312_acceptance_review import (
        acceptance_review_blockers,
        finalize_prd_312_acceptance_manifest,
        load_prd_312_acceptance_review_plan,
        write_model_json_atomic,
    )

    review_path = resolve_project_path(args.review_plan)
    plan = load_prd_312_acceptance_review_plan(review_path)
    blockers = acceptance_review_blockers(plan)
    statuses = Counter(record.review_status for record in plan.records)
    report = {
        "review_plan_path": str(review_path),
        "record_count": len(plan.records),
        "review_status_counts": dict(sorted(statuses.items())),
        "independence_attested": plan.independence_attested_by is not None,
        "blockers": blockers,
        "formal_manifest_ready": not blockers,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if blockers:
        raise SystemExit("Formal acceptance manifest is not ready.")
    manifest = finalize_prd_312_acceptance_manifest(plan)
    output_path = resolve_project_path(args.output)
    write_model_json_atomic(output_path, manifest)
    print(f"formal_manifest_path: {output_path}")


if __name__ == "__main__":
    main()
