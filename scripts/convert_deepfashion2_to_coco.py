"""Convert DeepFashion2 annotations to COCO instance segmentation JSON."""

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
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert DeepFashion2 annotations to COCO format."
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="DeepFashion2 split to convert.",
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
        help="Optional maximum number of images for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    """Run DeepFashion2 to COCO conversion."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.segmentation.coco import (
        convert_deepfashion2_to_coco,
    )

    settings = load_settings()
    output_path = args.output
    if output_path is None:
        output_path = (
            f"data/processed/autodl/segmentation/deepfashion2_{args.split}.json"
        )

    summary = convert_deepfashion2_to_coco(
        root=resolve_project_path(settings.datasets.deepfashion2_root),
        split=args.split,
        output_path=resolve_project_path(output_path),
        limit=args.limit,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
