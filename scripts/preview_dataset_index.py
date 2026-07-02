"""Preview records from generated dataset indexes."""

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
        description="Preview generated dataset JSONL indexes."
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/autodl/indexes/manifest.json",
        help="Project-relative path to the generated manifest.",
    )
    parser.add_argument(
        "--index-name",
        default="deepfashion2_train",
        help="Index name to preview.",
    )
    parser.add_argument(
        "--category-name",
        default=None,
        help="Optional category filter for records with item annotations.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of records to preview.",
    )
    return parser.parse_args()


def main() -> None:
    """Print index metadata and a small record preview."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.datasets.index_reader import (
        DatasetIndexReader,
    )

    reader = DatasetIndexReader(resolve_project_path(args.manifest))
    preview = {
        "indexes": reader.list_indexes(),
        "record_counts": reader.record_counts(),
        "records": list(
            reader.iter_records(
                args.index_name,
                category_name=args.category_name,
                limit=args.limit,
            )
        ),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
