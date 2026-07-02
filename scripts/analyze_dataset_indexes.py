"""Analyze generated dataset indexes and print summary statistics."""

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
        description="Compute statistics from generated dataset JSONL indexes."
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/autodl/indexes/manifest.json",
        help="Project-relative path to the generated manifest.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional project-relative path for writing statistics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Print dataset index statistics as JSON."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.datasets.index_reader import (
        DatasetIndexReader,
    )
    from fashion_semantic_parser.dao.datasets.statistics import (
        compute_dataset_statistics,
    )

    reader = DatasetIndexReader(resolve_project_path(args.manifest))
    statistics = compute_dataset_statistics(reader)
    statistics_json = json.dumps(
        statistics.model_dump(),
        ensure_ascii=False,
        indent=2,
    )
    if args.output is not None:
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(statistics_json, encoding="utf-8")

    print(statistics_json)


if __name__ == "__main__":
    main()
