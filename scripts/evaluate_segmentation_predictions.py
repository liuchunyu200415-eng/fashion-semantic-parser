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
        "--category-score-threshold",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeat for category-specific deployment thresholds.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional project-relative or absolute output JSON path.",
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
    coco_ground_truth = COCO(str(validation_path))
    category_thresholds = _parse_category_thresholds(args.category_score_threshold)
    category_thresholds_by_id = _category_thresholds_by_id(
        coco_ground_truth,
        category_thresholds,
    )
    filtered_predictions = _filter_predictions(
        predictions,
        score_threshold=args.score_threshold,
        category_score_thresholds=category_thresholds_by_id,
    )
    if not filtered_predictions:
        raise ValueError(
            "No predictions remain after applying --score-threshold "
            f"{args.score_threshold}."
        )

    coco_predictions = coco_ground_truth.loadRes(filtered_predictions)
    coco_eval = COCOeval(coco_ground_truth, coco_predictions, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    class_names = _coco_class_names(coco_ground_truth, coco_eval.params.catIds)
    mask_iou_results = _coco_matched_mask_iou_metrics(
        coco_eval,
        class_names=class_names,
    )
    results = {
        "validation_json": str(validation_path),
        "predictions_json": str(prediction_path),
        "score_threshold": args.score_threshold,
        "category_score_thresholds": category_thresholds,
        "prediction_count_before_filter": len(predictions),
        "prediction_count_after_filter": len(filtered_predictions),
        "segm_coco": _coco_ap_summary(coco_eval.stats),
        "segm_direct_iou": mask_iou_results,
    }
    output_json = json.dumps(results, ensure_ascii=False, indent=2)
    print(output_json)

    if args.output:
        output_path = _resolve_path(args.output, resolve_project_path)
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
    category_score_thresholds: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Keep predictions at or above the selected deployment score threshold."""
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    category_score_thresholds = category_score_thresholds or {}
    return [
        prediction
        for prediction in predictions
        if float(prediction.get("score", 0.0))
        >= category_score_thresholds.get(
            int(prediction.get("category_id", -1)),
            score_threshold,
        )
    ]


def _parse_category_thresholds(values: list[str]) -> dict[str, float]:
    """Parse repeated NAME=VALUE category threshold arguments."""
    thresholds: dict[str, float] = {}
    for value in values:
        name, separator, raw_threshold = value.partition("=")
        if not separator or not name.strip():
            raise ValueError("--category-score-threshold must use NAME=VALUE syntax.")
        try:
            threshold = float(raw_threshold)
        except ValueError as error:
            raise ValueError(f"Invalid category threshold value: {value}") from error
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Category score thresholds must be between 0 and 1.")
        thresholds[name.strip()] = threshold
    return thresholds


def _category_thresholds_by_id(
    coco_ground_truth: Any,
    thresholds: dict[str, float],
) -> dict[int, float]:
    """Resolve category-name thresholds against a COCO taxonomy."""
    categories = coco_ground_truth.loadCats(coco_ground_truth.getCatIds())
    category_ids_by_name = {
        str(category["name"]): int(category["id"]) for category in categories
    }
    unknown_names = set(thresholds) - set(category_ids_by_name)
    if unknown_names:
        raise ValueError(
            "Unknown COCO categories in score thresholds: "
            + ", ".join(sorted(unknown_names))
        )
    return {
        category_ids_by_name[name]: threshold for name, threshold in thresholds.items()
    }


def _coco_class_names(coco_ground_truth: Any, category_ids: list[int]) -> list[str]:
    """Return class names in the same category order used by COCOeval."""
    categories = coco_ground_truth.loadCats(category_ids)
    category_names = {
        int(category["id"]): str(category["name"]) for category in categories
    }
    return [category_names[category_id] for category_id in category_ids]


def _coco_ap_summary(stats: Any) -> dict[str, float]:
    """Return standard COCO mask AP values as percentages."""
    values = list(stats)
    if len(values) < 6:
        raise ValueError("COCOeval stats must contain at least six AP values.")
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    return {
        key: float(value) * 100.0 if float(value) >= 0.0 else float("nan")
        for key, value in zip(keys, values[:6], strict=True)
    }


def _resolve_path(path: str, resolver: Any) -> Path:
    """Use absolute paths directly and resolve project-relative paths."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolver(candidate)


if __name__ == "__main__":
    main()
