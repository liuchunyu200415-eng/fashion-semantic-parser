"""Merge reviewed LLM rewrites into the Fashionpedia referring index."""

# Direct execution adds ``src`` before importing the local package.
# The bootstrap mirrors other standalone repository scripts.
# pylint: disable=import-outside-toplevel,duplicate-code

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
    """Parse one strict paraphrase merge and scale audit."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge LLM paraphrases without changing any source target or Mask."
        )
    )
    parser.add_argument(
        "--base-index",
        default=(
            "data/processed/autodl/localization/" "fashionpedia_referring_train.jsonl"
        ),
    )
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--output",
        default=(
            "data/processed/autodl/localization/"
            "fashionpedia_referring_train_expanded.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "outputs/localization/referring_training/"
            "fashionpedia_train_expanded_summary.json"
        ),
    )
    parser.add_argument("--minimum-sample-count", type=int, default=100_000)
    parser.add_argument("--allow-unreviewed", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate, merge, audit, and print one bounded summary."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.referring_paraphrase import (
        merge_referring_paraphrases,
    )

    summary = merge_referring_paraphrases(
        base_index_path=resolve_project_path(args.base_index),
        result_path=resolve_project_path(args.results),
        output_path=resolve_project_path(args.output),
        summary_output_path=resolve_project_path(args.summary_output),
        minimum_sample_count=args.minimum_sample_count,
        allow_unreviewed=args.allow_unreviewed,
    )
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
