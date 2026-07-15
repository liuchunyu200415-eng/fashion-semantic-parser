"""Run PRD 3.1.1 garment instance segmentation on one image."""

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
        description="Predict PRD 3.1.1 garment instances for one image."
    )
    parser.add_argument("image", help="Project-relative RGB product image path.")
    parser.add_argument(
        "--config",
        default="configs/segmentation_mask_rcnn.yaml",
        help="Project-relative segmentation YAML config path.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Project-relative or absolute trained Detectron2 weights path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional project-relative output JSON path.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--min-size-test", type=int, default=None)
    parser.add_argument("--max-size-test", type=int, default=None)
    parser.add_argument(
        "--subject-roi",
        default=None,
        help="Optional subject ROI as x_min,y_min,x_max,y_max.",
    )
    return parser.parse_args()


def main() -> None:
    """Run single-image instance segmentation and print JSON."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.service.segmentation_baseline import (
        Detectron2SegmentationBaseline,
        SegmentationBaselineSettings,
        filter_prediction_by_subject_roi,
    )
    from fashion_semantic_parser.models.segmentation import SegmentationSubjectROI

    raw_config = _read_yaml(resolve_project_path(args.config))
    overrides = {
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
    prediction = Detectron2SegmentationBaseline(settings).predict_image(
        resolve_project_path(args.image)
    )
    if args.subject_roi:
        prediction = filter_prediction_by_subject_roi(
            prediction,
            _parse_subject_roi(args.subject_roi, SegmentationSubjectROI),
        )
    output_json = json.dumps(
        prediction.model_dump(),
        ensure_ascii=False,
        indent=2,
    )

    if args.output:
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json + "\n", encoding="utf-8")
    else:
        print(output_json)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


def _parse_subject_roi(raw_value: str, roi_class: type[Any]) -> Any:
    """Parse a subject ROI from x_min,y_min,x_max,y_max text."""
    values = [float(value.strip()) for value in raw_value.split(",")]
    if len(values) != 4:
        raise ValueError("--subject-roi must use x_min,y_min,x_max,y_max")
    return roi_class(
        x_min=values[0],
        y_min=values[1],
        x_max=values[2],
        y_max=values[3],
    )


if __name__ == "__main__":
    main()
