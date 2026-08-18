"""Generate unreviewed referring rewrites with the PRD-listed Qwen-VL model."""

# Direct execution adds ``src`` before importing the local package.
# pylint: disable=import-outside-toplevel

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
    """Parse one bounded, resumable offline generation run."""
    parser = argparse.ArgumentParser(
        description=("Generate unreviewed paraphrases with pinned Qwen-VL-Chat-Int4.")
    )
    parser.add_argument(
        "--jobs",
        default=(
            "outputs/localization/referring_training/"
            "targeted_paraphrase_jobs_20k.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/localization/referring_training/"
            "qwen_vl_targeted_paraphrases_20k_unreviewed.jsonl"
        ),
    )
    parser.add_argument(
        "--failures",
        default=(
            "outputs/localization/referring_training/"
            "qwen_vl_targeted_paraphrases_20k_failures.jsonl"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/localization_qwen_vl_paraphrase.yaml",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Load local Qwen-VL, generate missing jobs, and report audit counts."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization import (
        referring_paraphrase_generation,
    )
    from fashion_semantic_parser.service.qwen_vl_paraphraser import (
        QwenVlParaphraser,
        load_qwen_vl_paraphrase_settings,
    )

    settings = load_qwen_vl_paraphrase_settings(args.config)
    if args.model_path is not None:
        settings = settings.model_copy(update={"model_path": args.model_path})
    generator = QwenVlParaphraser(settings)
    summary = referring_paraphrase_generation.run_referring_paraphrase_jobs(
        job_path=resolve_project_path(args.jobs),
        output_path=resolve_project_path(args.output),
        failure_path=resolve_project_path(args.failures),
        generator=generator,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
    )
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if summary.failed_result_count:
        raise RuntimeError(
            f"Qwen-VL paraphrase generation retained "
            f"{summary.failed_result_count} failed jobs."
        )


if __name__ == "__main__":
    main()
