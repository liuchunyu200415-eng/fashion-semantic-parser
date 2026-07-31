"""Build a COCO subset for shoes, bag, and accessory small-object training."""

import argparse
import json
import sys
from pathlib import Path


def add_src_to_python_path() -> None:
    """Add the local src directory when the package is not installed yet."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse small-object subset arguments."""
    parser = argparse.ArgumentParser(
        description="Select COCO images containing target small objects."
    )
    parser.add_argument(
        "--source",
        default="data/processed/autodl/segmentation/fashionpedia_train.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/processed/autodl/segmentation/"
            "fashionpedia_train_small_objects.json"
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["shoes", "bag", "accessory"],
    )
    parser.add_argument("--maximum-area", type=float, default=float(32**2))
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Print counts without writing the subset JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Build or audit the targeted small-object subset."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.segmentation.small_objects import (
        build_small_object_coco_subset,
    )

    summary = build_small_object_coco_subset(
        source_path=resolve_project_path(args.source),
        output_path=None if args.audit_only else resolve_project_path(args.output),
        target_categories=args.categories,
        maximum_area=args.maximum_area,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
