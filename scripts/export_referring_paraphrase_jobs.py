"""Export vendor-neutral LLM rewrite jobs from a referring JSONL index."""

# Direct execution adds ``src`` before importing the local package.
# pylint: disable=import-outside-toplevel

import argparse
import sys
from pathlib import Path


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one bounded LLM rewrite-job export."""
    parser = argparse.ArgumentParser(
        description=(
            "Export paraphrase requests without sending Fashionpedia data "
            "to any external model automatically."
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
            "outputs/localization/referring_training/"
            "fashionpedia_train_paraphrase_jobs.jsonl"
        ),
    )
    parser.add_argument("--paraphrases-per-sample", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Write source-fingerprinted jobs and report only the bounded count."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.referring_paraphrase import (
        export_referring_paraphrase_jobs,
    )

    count = export_referring_paraphrase_jobs(
        index_path=resolve_project_path(args.index),
        output_path=resolve_project_path(args.output),
        paraphrases_per_sample=args.paraphrases_per_sample,
        limit=args.limit,
    )
    print(f"paraphrase_job_count: {count}")
    print(f"output_path: {resolve_project_path(args.output)}")


if __name__ == "__main__":
    main()
