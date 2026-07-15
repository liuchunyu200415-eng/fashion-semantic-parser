"""Train a PRD 3.1.1 garment instance segmentation model."""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def add_src_to_python_path() -> None:
    """Add the local src directory when the package is not installed yet."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PRD 3.1.1 garment instance segmentation."
    )
    parser.add_argument(
        "--config",
        default="configs/segmentation_mask_rcnn.yaml",
        help="Project-relative segmentation YAML config path.",
    )
    parser.add_argument("--train-json", default=None)
    parser.add_argument("--val-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--ims-per-batch", type=int, default=None)
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--checkpoint-period", type=int, default=None)
    parser.add_argument("--eval-period", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume optimizer, scheduler, and iteration state from last_checkpoint.",
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help=(
            "Skip validation after training; evaluate the saved checkpoint "
            "separately."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run Detectron2-family segmentation training."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.service.segmentation_baseline import (
        Detectron2SegmentationBaseline,
        SegmentationBaselineSettings,
    )

    raw_config = _read_yaml(resolve_project_path(args.config))
    overrides = {
        "train_json": args.train_json,
        "val_json": args.val_json,
        "output_dir": args.output_dir,
        "weights": args.weights,
        "ims_per_batch": args.ims_per_batch,
        "base_lr": args.base_lr,
        "max_iter": args.max_iter,
        "checkpoint_period": args.checkpoint_period,
        "eval_period": args.eval_period,
        "device": args.device,
        "score_threshold": args.score_threshold,
        "resume": args.resume,
    }
    raw_config.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    if args.skip_final_eval:
        raw_config["evaluate_after_training"] = False
    settings = SegmentationBaselineSettings.model_validate(raw_config)
    Detectron2SegmentationBaseline(settings).train()


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


if __name__ == "__main__":
    main()
