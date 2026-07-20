"""Audit or convert Fashionpedia annotations to the PRD COCO taxonomy."""

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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit Fashionpedia labels or convert them to the PRD eight-class "
            "COCO schema."
        )
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="Fashionpedia split to inspect or convert.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Project-relative output path. Defaults to processed COCO path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional deterministic image limit for smoke tests.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Report mapping and exclusion counts without requiring images.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Fashionpedia annotation audit or conversion."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.segmentation.fashionpedia import (
        audit_fashionpedia_annotations,
        convert_fashionpedia_to_coco,
    )

    settings = load_settings()
    root = resolve_project_path(settings.datasets.fashionpedia_root)
    if args.audit_only:
        summary = audit_fashionpedia_annotations(
            root=root,
            split=args.split,
            limit=args.limit,
        )
    else:
        output_path = args.output or (
            f"data/processed/autodl/segmentation/fashionpedia_{args.split}.json"
        )
        summary = convert_fashionpedia_to_coco(
            root=root,
            split=args.split,
            output_path=resolve_project_path(output_path),
            limit=args.limit,
        )

    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
