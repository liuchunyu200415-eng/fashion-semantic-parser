"""Visualize PRD 3.1.1 garment segmentation predictions on one image."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

COLORS_BY_CATEGORY = {
    "top": (46, 204, 113),
    "pants": (52, 152, 219),
    "skirt": (155, 89, 182),
    "outerwear": (241, 196, 15),
    "dress": (231, 76, 60),
    "shoes": (230, 126, 34),
    "bag": (26, 188, 156),
    "accessory": (149, 165, 166),
}


def add_src_to_python_path() -> None:
    """Add the local src directory when the package is not installed yet."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize PRD 3.1.1 garment segmentation for one image."
    )
    parser.add_argument("image", help="Project-relative RGB product image path.")
    parser.add_argument(
        "--config",
        default="configs/segmentation_mask2former.yaml",
        help="Project-relative segmentation YAML config path.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Project-relative or absolute trained Detectron2 weights path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Project-relative visualization PNG/JPG output path.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional project-relative JSON prediction output path.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--min-size-test", type=int, default=None)
    parser.add_argument("--max-size-test", type=int, default=None)
    parser.add_argument("--detections-per-image", type=int, default=None)
    parser.add_argument(
        "--subject-roi",
        default=None,
        help="Optional subject crop as x_min,y_min,x_max,y_max.",
    )
    parser.add_argument(
        "--subject-roi-margin",
        type=float,
        default=None,
        help="Fractional context added around each side of the subject ROI.",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    """Run prediction and save an image overlay."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.service.segmentation_baseline import (
        Detectron2SegmentationBaseline,
        SegmentationBaselineSettings,
    )
    from fashion_semantic_parser.models.segmentation import SegmentationSubjectROI

    raw_config = _read_yaml(resolve_project_path(args.config))
    overrides = {
        "weights": args.weights,
        "device": args.device,
        "precision": args.precision,
        "score_threshold": args.score_threshold,
        "min_size_test": args.min_size_test,
        "max_size_test": args.max_size_test,
        "detections_per_image": args.detections_per_image,
        "subject_roi_margin": args.subject_roi_margin,
    }
    raw_config.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    settings = SegmentationBaselineSettings.model_validate(raw_config)
    image_path = resolve_project_path(args.image)
    subject_roi = (
        _parse_subject_roi(args.subject_roi, SegmentationSubjectROI)
        if args.subject_roi
        else None
    )
    prediction = Detectron2SegmentationBaseline(settings).predict_image(
        image_path,
        subject_roi=subject_roi,
    )

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    visualization = draw_prediction(
        image,
        prediction.model_dump(),
        args.alpha,
        subject_roi.model_dump() if subject_roi else None,
    )
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualization)

    if args.json_output:
        json_output_path = resolve_project_path(args.json_output)
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(
            json.dumps(prediction.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {"output": output_path.as_posix(), "instances": len(prediction.instances)}
        )
    )


def draw_prediction(
    image: np.ndarray,
    prediction: dict[str, Any],
    alpha: float,
    subject_roi: dict[str, float] | None = None,
) -> np.ndarray:
    """Draw translucent mask polygons, boxes, and labels on a BGR image."""
    result = image.copy()
    overlay = image.copy()

    for instance in prediction["instances"]:
        color = COLORS_BY_CATEGORY.get(instance["category_label"], (255, 255, 255))
        _draw_mask_polygons(overlay, instance["mask"], color)

    result = cv2.addWeighted(overlay, alpha, result, 1.0 - alpha, 0)

    for instance in prediction["instances"]:
        color = COLORS_BY_CATEGORY.get(instance["category_label"], (255, 255, 255))
        _draw_box_and_label(result, instance, color)

    if subject_roi:
        _draw_subject_roi(result, subject_roi)

    return result


def _draw_mask_polygons(
    image: np.ndarray, polygons: list[list[float]], color: tuple[int, int, int]
) -> None:
    """Fill all polygons belonging to one instance."""
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(image, [points], color)


def _draw_box_and_label(
    image: np.ndarray,
    instance: dict[str, Any],
    color: tuple[int, int, int],
) -> None:
    """Draw one xyxy bounding box and compact label."""
    box = instance["box"]
    x_min = int(round(box["x_min"]))
    y_min = int(round(box["y_min"]))
    x_max = int(round(box["x_max"]))
    y_max = int(round(box["y_max"]))
    if x_max <= x_min or y_max <= y_min:
        return
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    label = f'{instance["category_label"]} {instance["confidence"]:.2f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (label_width, label_height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    label_y = max(y_min, label_height + baseline + 4)
    cv2.rectangle(
        image,
        (x_min, label_y - label_height - baseline - 4),
        (x_min + label_width + 6, label_y),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x_min + 3, label_y - baseline - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _roi_to_int_bounds(roi: dict[str, float]) -> tuple[int, int, int, int]:
    """Convert ROI coordinates to integer pixel bounds."""
    return (
        int(round(roi["x_min"])),
        int(round(roi["y_min"])),
        int(round(roi["x_max"])),
        int(round(roi["y_max"])),
    )


def _draw_subject_roi(image: np.ndarray, roi: dict[str, float]) -> None:
    """Draw the unexpanded subject/person ROI used to create the crop."""
    x_min, y_min, x_max, y_max = _roi_to_int_bounds(roi)
    color = (255, 255, 255)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(
        image,
        "subject ROI",
        (x_min + 4, max(18, y_min - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


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
