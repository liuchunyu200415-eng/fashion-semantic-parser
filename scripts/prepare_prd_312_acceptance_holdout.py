"""Freeze an auditable PRD 3.1.2 acceptance holdout candidate pool."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_IMAGES = "outputs/localization/prd_312_acceptance_holdout_images.txt"
DEFAULT_EXCLUSIONS = "configs/prd_312_acceptance_excluded_images.txt"
DEFAULT_OUTPUT = (
    "data/benchmarks/localization/prd_312_acceptance_holdout_candidates_v1.json"
)
DEFAULT_SUMMARY = "outputs/localization/prd_312_acceptance_holdout_summary.json"


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse candidate-list, exclusion-list, and output paths.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Hash, decode, deduplicate, and freeze holdout candidates."
    )
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--excluded-images", default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    """Build and atomically publish the holdout candidate inventory.

    Raises:
        ValueError: If the progress interval is invalid.
    """
    args = parse_args()
    if args.progress_every < 1:
        raise ValueError("progress-every must be at least one")
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.prd_312_acceptance_holdout import (
        prepare_prd_312_acceptance_holdout_inventory,
        read_path_list,
    )
    from fashion_semantic_parser.dao.localization.prd_312_acceptance_review import (
        write_model_json_atomic,
    )

    project_root = Path(__file__).resolve().parents[1]
    image_list_path = resolve_project_path(args.images)
    exclusion_list_path = resolve_project_path(args.excluded_images)
    inventory, summary = prepare_prd_312_acceptance_holdout_inventory(
        project_root=project_root,
        image_paths=read_path_list(image_list_path),
        excluded_paths=read_path_list(exclusion_list_path),
        generated_at=datetime.now(timezone.utc),
        progress_callback=lambda current, total: _print_progress(
            current, total, args.progress_every
        ),
    )
    output_path = resolve_project_path(args.output)
    summary_path = resolve_project_path(args.summary_output)
    write_model_json_atomic(output_path, inventory)
    write_model_json_atomic(summary_path, summary)
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print(f"inventory_path: {output_path}")
    print(f"summary_path: {summary_path}")


def _print_progress(current: int, total: int, every: int) -> None:
    """Print bounded progress for long image inventories."""
    if current == 1 or current == total or current % every == 0:
        print(f"[{current}/{total}]", flush=True)


if __name__ == "__main__":
    main()
