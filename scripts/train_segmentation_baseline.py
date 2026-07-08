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
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--device", default=None)
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
        "max_iter": args.max_iter,
        "device": args.device,
    }
    raw_config.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
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
