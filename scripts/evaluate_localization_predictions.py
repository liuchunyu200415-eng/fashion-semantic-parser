"""Evaluate saved PRD 3.1.2 candidates with offline Top-K and score filters."""

import argparse
import io
import json
import math
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, cast

DEFAULT_VALIDATION_JSON = (
    "data/processed/autodl/localization/fashionpedia_parts_validation.json"
)


def add_src_to_python_path() -> None:
    """Add the local package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse saved-prediction evaluation arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one Fashionpedia part category after per-image Top-K "
            "and score filtering, without rerunning either model."
        )
    )
    parser.add_argument(
        "--val-json",
        default=DEFAULT_VALIDATION_JSON,
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--run-summary",
        default=None,
        help=(
            "Optional prediction-run summary containing image_ids. By default "
            "the sibling *_summary.json file is used when present."
        ),
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    """Apply candidate filters and evaluate exact-category masks with COCO."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path

    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("--top-k must be at least one.")

    validation_path = _resolve_path(args.val_json, resolve_project_path)
    prediction_path = _resolve_path(args.predictions, resolve_project_path)
    default_summary_path = _summary_path(prediction_path)
    summary_path = (
        _resolve_path(args.run_summary, resolve_project_path)
        if args.run_summary
        else default_summary_path
    )
    result = evaluate_localization_categories(
        validation_path=validation_path,
        prediction_path=prediction_path,
        category_names=[args.category],
        score_threshold=args.score_threshold,
        top_k=args.top_k,
        summary_path=summary_path,
    )
    result["category"] = args.category
    result["category_id"] = result["category_ids"][0]
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(serialized)
    if args.output:
        output_path = _resolve_path(args.output, resolve_project_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")


def evaluate_localization_categories(
    *,
    validation_path: Path,
    prediction_path: Path,
    category_names: list[str],
    score_threshold: float = 0.0,
    top_k: int | None = None,
    summary_path: Path | None = None,
    image_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate one or more exact Fashionpedia categories on an explicit scope."""
    add_src_to_python_path()
    from fashion_semantic_parser.service.segmentation_metrics import (
        _coco_ap_at_iou,
        _coco_matched_mask_iou_metrics,
        _summarize_mask_iou_matches,
    )

    try:
        from pycocotools.coco import COCO  # type: ignore[import-untyped]
        from pycocotools.cocoeval import COCOeval  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError(
            "pycocotools is required for localization evaluation."
        ) from error

    if not category_names:
        raise ValueError("At least one localization category is required.")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least one.")

    with redirect_stdout(io.StringIO()):
        coco_ground_truth = COCO(str(validation_path))
    category_ids = [
        _category_id(coco_ground_truth, category_name)
        for category_name in category_names
    ]
    if image_ids is None:
        if summary_path is not None:
            image_ids = _evaluation_image_ids_for_categories(
                coco_ground_truth,
                category_ids=category_ids,
                summary_path=summary_path,
            )
        else:
            image_ids = sorted(
                {
                    int(image_id)
                    for category_id in category_ids
                    for image_id in coco_ground_truth.getImgIds(catIds=[category_id])
                }
            )
    image_ids = sorted({int(image_id) for image_id in image_ids})
    if not image_ids:
        raise ValueError("No evaluation images were selected.")

    predictions = _read_prediction_list(prediction_path)
    evaluation_image_ids = set(image_ids)
    evaluation_category_ids = set(category_ids)
    relevant_predictions = [
        prediction
        for prediction in predictions
        if int(prediction.get("category_id", -1)) in evaluation_category_ids
        and int(prediction.get("image_id", -1)) in evaluation_image_ids
    ]
    filtered_predictions = _filter_predictions_per_image_category(
        relevant_predictions,
        score_threshold=score_threshold,
        top_k=top_k,
    )
    ground_truth_count = len(
        coco_ground_truth.getAnnIds(
            imgIds=image_ids,
            catIds=category_ids,
            iscrowd=False,
        )
    )

    if filtered_predictions:
        with redirect_stdout(io.StringIO()):
            coco_predictions = coco_ground_truth.loadRes(filtered_predictions)
            coco_eval = COCOeval(coco_ground_truth, coco_predictions, "segm")
            coco_eval.params.imgIds = image_ids
            coco_eval.params.catIds = category_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        coco_metrics = _coco_ap_summary(coco_eval.stats)
        for threshold in (0.50, 0.85, 0.90):
            metric_name = f"AP{int(threshold * 100)}"
            coco_metrics[metric_name] = _coco_ap_at_iou(coco_eval, threshold)
            for category_index, category_name in enumerate(category_names):
                coco_metrics[f"{metric_name}-{category_name}"] = _coco_ap_at_iou(
                    coco_eval,
                    threshold,
                    category_index=category_index,
                )
        direct_metrics = _coco_matched_mask_iou_metrics(
            coco_eval,
            class_names=category_names,
        )
    else:
        coco_metrics = {
            key: None
            for key in ("AP", "AP50", "AP75", "AP85", "AP90", "APs", "APm", "APl")
        }
        direct_metrics = _summarize_mask_iou_matches(
            matched_ious=[],
            ground_truth_count=ground_truth_count,
            prediction_count=0,
        )

    precision = _finite_float(direct_metrics.get("Precision50"))
    recall = _finite_float(direct_metrics.get("Recall50"))
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )
    return cast(
        dict[str, Any],
        _json_safe(
            {
                "validation_json": str(validation_path),
                "predictions_json": str(prediction_path),
                "run_summary_json": (
                    str(summary_path)
                    if summary_path is not None and summary_path.is_file()
                    else None
                ),
                "categories": category_names,
                "category_ids": category_ids,
                "image_count": len(image_ids),
                "ground_truth_count": ground_truth_count,
                "candidate_count_before_filter": len(relevant_predictions),
                "candidate_count_after_filter": len(filtered_predictions),
                "score_threshold": score_threshold,
                "top_k": top_k,
                "segm_coco": coco_metrics,
                "segm_direct_iou": {
                    **direct_metrics,
                    "F1_50": f1,
                },
            },
        ),
    )


def _read_prediction_list(path: Path) -> list[dict[str, Any]]:
    """Read a flat COCO result list."""
    with path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)
    if not isinstance(predictions, list) or not all(
        isinstance(prediction, dict) for prediction in predictions
    ):
        raise ValueError(f"Expected a list of prediction objects: {path}")
    return predictions


def _filter_predictions_per_image(
    predictions: list[dict[str, Any]],
    *,
    score_threshold: float,
    top_k: int | None,
) -> list[dict[str, Any]]:
    """Apply score filtering, confidence sorting, and Top-K independently."""
    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) >= score_threshold:
            grouped[int(prediction["image_id"])].append(prediction)

    filtered: list[dict[str, Any]] = []
    for image_id in sorted(grouped):
        ranked = sorted(
            grouped[image_id],
            key=lambda prediction: float(prediction.get("score", 0.0)),
            reverse=True,
        )
        filtered.extend(ranked if top_k is None else ranked[:top_k])
    return filtered


def _filter_predictions_per_image_category(
    predictions: list[dict[str, Any]],
    *,
    score_threshold: float,
    top_k: int | None,
) -> list[dict[str, Any]]:
    """Apply Top-K independently to each image and semantic category."""
    grouped: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) >= score_threshold:
            key = (int(prediction["image_id"]), int(prediction["category_id"]))
            grouped[key].append(prediction)

    filtered: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda prediction: float(prediction.get("score", 0.0)),
            reverse=True,
        )
        filtered.extend(ranked if top_k is None else ranked[:top_k])
    return filtered


def _evaluation_image_ids(
    coco_ground_truth: Any,
    *,
    category_id: int,
    summary_path: Path,
) -> list[int]:
    """Use the run's explicit image set so zero-prediction images remain misses."""
    valid_image_ids = set(coco_ground_truth.getImgIds(catIds=[category_id]))
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        raw_image_ids = summary.get("image_ids") if isinstance(summary, dict) else None
        if not isinstance(raw_image_ids, list):
            raise ValueError(f"Run summary has no image_ids list: {summary_path}")
        image_ids = [int(image_id) for image_id in raw_image_ids]
        invalid_image_ids = set(image_ids) - valid_image_ids
        if invalid_image_ids:
            raise ValueError(
                "Run summary contains images without target ground truth: "
                f"{sorted(invalid_image_ids)[:5]}"
            )
        return image_ids
    return sorted(valid_image_ids)


def _evaluation_image_ids_for_categories(
    coco_ground_truth: Any,
    *,
    category_ids: list[int],
    summary_path: Path,
) -> list[int]:
    """Read one run scope and ensure every image has relevant ground truth."""
    valid_image_ids = {
        int(image_id)
        for category_id in category_ids
        for image_id in coco_ground_truth.getImgIds(catIds=[category_id])
    }
    if not summary_path.is_file():
        return sorted(valid_image_ids)
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    raw_image_ids = summary.get("image_ids") if isinstance(summary, dict) else None
    if not isinstance(raw_image_ids, list):
        raise ValueError(f"Run summary has no image_ids list: {summary_path}")
    image_ids = [int(image_id) for image_id in raw_image_ids]
    invalid_image_ids = set(image_ids) - valid_image_ids
    if invalid_image_ids:
        raise ValueError(
            "Run summary contains images without target ground truth: "
            f"{sorted(invalid_image_ids)[:5]}"
        )
    return image_ids


def _category_id(coco_ground_truth: Any, category_name: str) -> int:
    """Resolve one exact COCO category name."""
    matches = [
        int(category_id)
        for category_id, category in coco_ground_truth.cats.items()
        if category.get("name") == category_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one category named {category_name!r}, found {len(matches)}."
        )
    return matches[0]


def _coco_ap_summary(stats: Any) -> dict[str, float | None]:
    """Return standard COCO segmentation AP values as percentages."""
    values = list(stats)
    if len(values) < 6:
        raise ValueError("COCOeval stats must contain at least six AP values.")
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    return {
        key: float(value) * 100.0 if float(value) >= 0.0 else None
        for key, value in zip(keys, values[:6], strict=True)
    }


def _finite_float(value: Any) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats before strict JSON serialization."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summary_path(prediction_path: Path) -> Path:
    """Return the default sibling summary path written by dataset prediction."""
    return prediction_path.with_name(f"{prediction_path.stem}_summary.json")


def _resolve_path(path: str, resolver: Any) -> Path:
    """Use absolute paths directly and resolve project-relative paths."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolver(candidate)


if __name__ == "__main__":
    main()
