"""Create the contract-sized PRD 3.1.2 human-review plan."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_CONTRACT = "configs/prd_312_acceptance_contract.json"
DEFAULT_OUTPUT = "data/benchmarks/localization/prd_312_acceptance_review_v1.json"


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse acceptance-review plan paths.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=" ".join(
            (
                "Create 1,000 deterministic quota slots for independent manual",
                "PRD 3.1.2 acceptance review.",
            )
        )
    )
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Build, validate, and atomically write the pending review plan."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.prd_312_acceptance import (
        load_prd_312_acceptance_contract,
    )
    from fashion_semantic_parser.dao.localization.prd_312_acceptance_planning import (
        build_prd_312_acceptance_review_plan,
    )
    from fashion_semantic_parser.dao.localization.prd_312_acceptance_review import (
        write_model_json_atomic,
    )

    contract = load_prd_312_acceptance_contract(resolve_project_path(args.contract))
    plan = build_prd_312_acceptance_review_plan(contract)
    output_path = resolve_project_path(args.output)
    write_model_json_atomic(output_path, plan)
    summary = {
        "review_plan_path": str(output_path),
        "record_count": len(plan.records),
        "review_status_counts": dict(
            sorted(Counter(record.review_status for record in plan.records).items())
        ),
        "annotation_requirement_counts": dict(
            sorted(
                Counter(
                    record.annotation_requirement for record in plan.records
                ).items()
            )
        ),
        "target_region_counts": dict(
            sorted(Counter(record.target_region for record in plan.records).items())
        ),
        "target_label_counts": dict(
            sorted(Counter(record.target_label for record in plan.records).items())
        ),
        "independence_attested": False,
        "formal_manifest_ready": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
