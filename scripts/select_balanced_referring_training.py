"""Select a deterministic balanced core from a large referring index."""

# Direct execution adds ``src`` before importing the local package.
# pylint: disable=import-outside-toplevel,duplicate-code

import argparse
import sys
from pathlib import Path


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one balanced core selection."""
    parser = argparse.ArgumentParser(
        description=(
            "Balance Fashionpedia referring records by label, language, and "
            "modifier dimensions."
        )
    )
    parser.add_argument(
        "--index",
        default=(
            "data/processed/autodl/localization/" "fashionpedia_referring_train.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "data/processed/autodl/localization/"
            "fashionpedia_referring_train_balanced_100k.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "outputs/localization/referring_training/"
            "fashionpedia_train_balanced_100k_summary.json"
        ),
    )
    parser.add_argument("--sample-count", type=int, default=100_000)
    parser.add_argument("--seed", default="prd-312-balanced-v1")
    return parser.parse_args()


def main() -> None:
    """Select, validate, atomically write, and summarize the balanced core."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.referring_sampling import (
        build_balanced_referring_subset,
    )

    summary = build_balanced_referring_subset(
        index_path=resolve_project_path(args.index),
        output_path=resolve_project_path(args.output),
        summary_output_path=resolve_project_path(args.summary_output),
        sample_count=args.sample_count,
        seed=args.seed,
    )
    print(f"input_sample_count: {summary.input_sample_count}")
    print(f"output_sample_count: {summary.output_sample_count}")
    print(f"selected_image_count: {summary.selected_image_count}")
    print(f"target_reference_count: {summary.target_reference_count}")
    print(f"weak_part_counts: {summary.weak_part_counts}")
    print(f"output_path: {summary.output_path}")
    print(f"summary_output_path: {resolve_project_path(args.summary_output)}")


if __name__ == "__main__":
    main()
