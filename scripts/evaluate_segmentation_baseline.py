"""Evaluate a trained PRD 3.1.1 garment instance segmentation model."""

import argparse
import json
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
        description="Evaluate PRD 3.1.1 garment instance segmentation."
    )
    parser.add_argument(
        "--config",
        default="configs/segmentation_mask2former.yaml",
        help="Project-relative segmentation YAML config path.",
    )
    parser.add_argument("--val-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--weights",
        required=True,
        help="Project-relative or absolute trained Detectron2 weights path.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--min-size-test", type=int, default=None)
    parser.add_argument("--max-size-test", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run Detectron2-family segmentation evaluation."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.service.segmentation_baseline import (
        Detectron2SegmentationBaseline,
        SegmentationBaselineSettings,
    )

    raw_config = _read_yaml(resolve_project_path(args.config))
    overrides = {
        "val_json": args.val_json,
        "output_dir": args.output_dir,
        "weights": args.weights,
        "device": args.device,
        "score_threshold": args.score_threshold,
        "min_size_test": args.min_size_test,
        "max_size_test": args.max_size_test,
    }
    raw_config.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    settings = SegmentationBaselineSettings.model_validate(raw_config)
    results = Detectron2SegmentationBaseline(settings).evaluate()
    print(json.dumps(results, ensure_ascii=False, indent=2))


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


if __name__ == "__main__":
    main()
