"""Prepare compact Fashionpedia language-region training records."""

import argparse
import json
import sys
from pathlib import Path


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one Fashionpedia referring-data preparation run."""
    parser = argparse.ArgumentParser(
        description=(
            "Build bilingual basic/spatial/relation and official-attribute "
            "query records that reference Fashionpedia source Masks."
        )
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-spatial-separation", type=float, default=0.05)
    parser.add_argument("--max-attributes-per-annotation", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Generate the JSONL index and print its bounded audit summary."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.localization.referring_training import (
        prepare_fashionpedia_referring_training_data,
    )

    settings = load_settings()
    output = args.output or (
        "data/processed/autodl/localization/"
        f"fashionpedia_referring_{args.split}.jsonl"
    )
    summary_output = args.summary_output or (
        "outputs/localization/referring_training/"
        f"fashionpedia_{args.split}_summary.json"
    )
    summary = prepare_fashionpedia_referring_training_data(
        root=resolve_project_path(settings.datasets.fashionpedia_root),
        split=args.split,
        output_path=resolve_project_path(output),
        summary_output_path=resolve_project_path(summary_output),
        limit=args.limit,
        min_spatial_separation=args.min_spatial_separation,
        max_attributes_per_annotation=args.max_attributes_per_annotation,
    )
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
