"""Report unresolved product decisions in the PRD 3.1.2 accuracy contract."""

# Direct execution adds ``src`` before importing the local package.
# pylint: disable=import-outside-toplevel

import argparse
import json
import sys
from pathlib import Path


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse the acceptance-contract audit arguments."""
    parser = argparse.ArgumentParser(
        description="Block PRD accuracy claims until the acceptance contract is locked."
    )
    parser.add_argument(
        "--contract",
        default="configs/prd_312_acceptance_contract.json",
    )
    return parser.parse_args()


def main() -> None:
    """Print a bounded JSON audit and fail while decisions remain unresolved."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.prd_312_acceptance import (
        acceptance_contract_blockers,
        load_prd_312_acceptance_contract,
    )

    contract = load_prd_312_acceptance_contract(resolve_project_path(args.contract))
    blockers = acceptance_contract_blockers(contract)
    report = {
        "acceptance_contract_locked": not blockers,
        "metric": ("single-query Top-1 Mask IoU strictly greater than 0.50"),
        "required_accuracy_percent": contract.required_accuracy * 100.0,
        "blockers": blockers,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
