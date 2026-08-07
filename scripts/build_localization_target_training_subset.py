"""Build a replay subset for weak PRD 3.1.2 localization classes."""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TARGET_CATEGORIES = ("buckle", "bow", "ribbon", "rivet", "tassel")


def add_src_to_python_path() -> None:
    """Add the local package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse targeted localization subset arguments."""
    parser = argparse.ArgumentParser(
        description="Select complete COCO images containing weak classes."
    )
    parser.add_argument(
        "--source",
        default=("data/processed/autodl/localization/fashionpedia_parts_train.json"),
    )
    parser.add_argument(
        "--output",
        default=(
            "data/processed/autodl/localization/"
            "fashionpedia_parts_train_critical_long_tail.json"
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_TARGET_CATEGORIES),
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build or audit the targeted class-replay subset."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.training_subset import (
        build_localization_target_coco_subset,
    )

    summary = build_localization_target_coco_subset(
        source_path=resolve_project_path(args.source),
        output_path=None if args.audit_only else resolve_project_path(args.output),
        target_categories=args.categories,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
