"""Build lightweight dataset indexes under the processed data directory."""

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
        description="Build JSONL indexes for configured fashion datasets."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/autodl/indexes",
        help="Project-relative output directory for generated indexes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum records per index for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    """Build dataset indexes and print the generated manifest."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.datasets.indexes import build_dataset_indexes

    settings = load_settings()
    manifest = build_dataset_indexes(
        fashionai_root=resolve_project_path(settings.datasets.fashionai_root),
        deepfashion2_root=resolve_project_path(settings.datasets.deepfashion2_root),
        output_dir=resolve_project_path(args.output_dir),
        limit=args.limit,
    )
    print(json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
