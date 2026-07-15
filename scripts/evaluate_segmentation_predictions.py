"""Evaluate direct mask IoU from saved Detectron2 COCO predictions."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def add_src_to_python_path() -> None:
    """Add the local src directory when the package is not installed yet."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate direct mask IoU from saved COCO predictions."
    )
    parser.add_argument(
        "--val-json",
        required=True,
        help="Project-relative or absolute COCO ground-truth JSON path.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="COCO predictions JSON written by Detectron2 COCOEvaluator.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional project-relative output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run COCO mask matching without loading model weights or a GPU."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.service.segmentation_metrics import (
        _coco_matched_mask_iou_metrics,
    )

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError(
            "pycocotools is required for saved-prediction mask IoU evaluation."
        ) from error

    validation_path = _resolve_path(args.val_json, resolve_project_path)
    prediction_path = _resolve_path(args.predictions, resolve_project_path)
    predictions = _read_prediction_list(prediction_path)
    filtered_predictions = _filter_predictions(
        predictions,
        score_threshold=args.score_threshold,
    )
    if not filtered_predictions:
        raise ValueError(
            "No predictions remain after applying --score-threshold "
            f"{args.score_threshold}."
        )

    coco_ground_truth = COCO(str(validation_path))
    coco_predictions = coco_ground_truth.loadRes(filtered_predictions)
    coco_eval = COCOeval(coco_ground_truth, coco_predictions, "segm")
    coco_eval.evaluate()
    class_names = _coco_class_names(coco_ground_truth, coco_eval.params.catIds)
    mask_iou_results = _coco_matched_mask_iou_metrics(
        coco_eval,
        class_names=class_names,
    )
    results = {
        "validation_json": str(validation_path),
        "predictions_json": str(prediction_path),
        "score_threshold": args.score_threshold,
        "prediction_count_before_filter": len(predictions),
        "prediction_count_after_filter": len(filtered_predictions),
        "segm_direct_iou": mask_iou_results,
    }
    output_json = json.dumps(results, ensure_ascii=False, indent=2)
    print(output_json)

    if args.output:
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json + "\n", encoding="utf-8")


def _read_prediction_list(path: Path) -> list[dict[str, Any]]:
    """Read the flat COCO result list written by Detectron2."""
    with path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected a COCO prediction list: {path}")
    if not all(isinstance(prediction, dict) for prediction in predictions):
        raise ValueError(f"COCO predictions must be JSON objects: {path}")
    return predictions


def _filter_predictions(
    predictions: list[dict[str, Any]],
    score_threshold: float,
) -> list[dict[str, Any]]:
    """Keep predictions at or above the selected deployment score threshold."""
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    return [
        prediction
        for prediction in predictions
        if float(prediction.get("score", 0.0)) >= score_threshold
    ]


def _coco_class_names(coco_ground_truth: Any, category_ids: list[int]) -> list[str]:
    """Return class names in the same category order used by COCOeval."""
    categories = coco_ground_truth.loadCats(category_ids)
    category_names = {
        int(category["id"]): str(category["name"]) for category in categories
    }
    return [category_names[category_id] for category_id in category_ids]


def _resolve_path(path: str, resolver: Any) -> Path:
    """Use absolute paths directly and resolve project-relative paths."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolver(candidate)


if __name__ == "__main__":
    main()
