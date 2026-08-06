"""Benchmark deployed PRD 3.1.2 masks on categories with exact ground truth."""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_VALIDATION_JSON = (
    "data/processed/autodl/localization/fashionpedia_parts_validation.json"
)
DEFAULT_CONFIG = "configs/localization_mask2former_parts_deployment.yaml"
DEFAULT_OUTPUT_DIR = "outputs/localization/performance/exact_gt"
ACCURACY_TARGET_PERCENT = 92.0


def add_src_to_python_path() -> None:
    """Add the local package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse exact-ground-truth benchmark arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the deployed supervised part model once per validation image "
            "and evaluate only PRD regions with equivalent Fashionpedia masks."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--val-json", default=DEFAULT_VALIDATION_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--roi-mode",
        choices=["full", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--image-limit-per-category",
        type=int,
        default=None,
        help="Optional deterministic smoke-test limit for each exact category.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--inference-score-threshold",
        type=float,
        default=None,
        help=(
            "Optional model-output threshold override. Use 0.0 when saving "
            "candidates for offline score/Top-K calibration."
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional extra offline filter. The deployment config threshold "
            "has already been applied during inference."
        ),
    )
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Generate deployment predictions and report exact-GT accuracy metrics."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.taxonomy import (
        FASHIONPEDIA_PART_CATEGORIES,
        PRD_LOCALIZATION_REGION_COVERAGE,
    )
    from fashion_semantic_parser.service.region_localization import (
        Mask2FormerPartLocalizationService,
    )
    from fashion_semantic_parser.service.segmentation_runtime import (
        GarmentSegmentationService,
    )
    from scripts.evaluate_localization_predictions import (
        evaluate_localization_categories,
    )

    if args.image_limit_per_category is not None and args.image_limit_per_category < 1:
        raise ValueError("--image-limit-per-category must be at least one.")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least one.")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    if (
        args.inference_score_threshold is not None
        and not 0.0 <= args.inference_score_threshold <= 1.0
    ):
        raise ValueError("--inference-score-threshold must be between 0 and 1.")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("--top-k must be at least one.")

    validation_path = _resolve_path(args.val_json, resolve_project_path)
    output_dir = _resolve_path(args.output_dir, resolve_project_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _read_coco(validation_path)
    exact_categories = _exact_prd_source_categories(PRD_LOCALIZATION_REGION_COVERAGE)
    category_metadata = {
        category.english_name: category for category in FASHIONPEDIA_PART_CATEGORIES
    }
    missing_metadata = set(exact_categories) - set(category_metadata)
    if missing_metadata:
        raise ValueError(f"Missing taxonomy metadata: {sorted(missing_metadata)}")

    category_ids = _category_ids(source, exact_categories)
    selected_images = _select_exact_gt_images(
        source,
        category_ids=set(category_ids.values()),
        image_limit_per_category=args.image_limit_per_category,
    )
    segmentation_service = None
    if args.inference_score_threshold is not None:
        segmentation_service = GarmentSegmentationService(
            args.config,
            settings_overrides={
                "score_threshold": args.inference_score_threshold,
            },
        )
    service = Mask2FormerPartLocalizationService(
        args.config,
        segmentation_service=segmentation_service,
    )
    predictions: list[dict[str, Any]] = []
    roi_sources: Counter[str] = Counter()
    started_at = time.perf_counter()
    for index, image in enumerate(selected_images, start=1):
        prediction = service.segmentation_service.segment(
            str(image["file_name"]),
            auto_subject_roi=args.roi_mode == "auto",
        )
        predictions.extend(
            _segmentation_prediction_to_coco(
                prediction,
                image_id=int(image["id"]),
                category_ids=category_ids,
            )
        )
        roi_sources[prediction.subject_roi_source or "full_image"] += 1
        if index % args.progress_every == 0 or index == len(selected_images):
            _print_progress(
                completed=index,
                total=len(selected_images),
                prediction_count=len(predictions),
                started_at=started_at,
            )

    elapsed_seconds = time.perf_counter() - started_at
    prediction_path = output_dir / "predictions.json"
    prediction_path.write_text(
        json.dumps(predictions, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "config": args.config,
        "validation_json": str(validation_path),
        "predictions_json": str(prediction_path),
        "roi_mode": args.roi_mode,
        "inference_score_threshold_override": args.inference_score_threshold,
        "offline_score_threshold": args.score_threshold,
        "offline_top_k": args.top_k,
        "exact_prd_regions": [
            coverage.english_name
            for coverage in PRD_LOCALIZATION_REGION_COVERAGE
            if coverage.status == "exact"
        ],
        "exact_source_categories": exact_categories,
        "image_ids": [int(image["id"]) for image in selected_images],
        "image_count": len(selected_images),
        "prediction_count": len(predictions),
        "roi_source_counts": dict(sorted(roi_sources.items())),
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": (
            len(selected_images) / elapsed_seconds if elapsed_seconds > 0.0 else None
        ),
    }
    summary_path = output_dir / "predictions_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    evaluation = evaluate_localization_categories(
        validation_path=validation_path,
        prediction_path=prediction_path,
        category_names=exact_categories,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
        summary_path=summary_path,
    )
    result = _build_acceptance_result(
        evaluation=evaluation,
        summary=summary,
        coverage=PRD_LOCALIZATION_REGION_COVERAGE,
    )
    result_path = output_dir / "metrics.json"
    serialized = json.dumps(
        _json_safe(result),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    result_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized, flush=True)


def _exact_prd_source_categories(coverage: list[Any]) -> list[str]:
    """Flatten exact PRD coverage into stable, unique source categories."""
    categories: list[str] = []
    for region in coverage:
        if region.status != "exact":
            continue
        for category_name in region.source_categories:
            if category_name not in categories:
                categories.append(category_name)
    return categories


def _category_ids(
    source: dict[str, list[dict[str, Any]]],
    category_names: list[str],
) -> dict[str, int]:
    """Resolve exact COCO category IDs without relying on list position."""
    by_name = {
        str(category["name"]): int(category["id"]) for category in source["categories"]
    }
    missing = set(category_names) - set(by_name)
    if missing:
        raise ValueError(f"Validation COCO categories are missing: {sorted(missing)}")
    return {category_name: by_name[category_name] for category_name in category_names}


def _select_exact_gt_images(
    source: dict[str, list[dict[str, Any]]],
    *,
    category_ids: set[int],
    image_limit_per_category: int | None,
) -> list[dict[str, Any]]:
    """Select the union of exact-GT images, optionally limiting each category."""
    image_ids_by_category: dict[int, list[int]] = {}
    for category_id in sorted(category_ids):
        image_ids_by_category[category_id] = sorted(
            {
                int(annotation["image_id"])
                for annotation in source["annotations"]
                if int(annotation["category_id"]) == category_id
                and int(annotation.get("iscrowd", 0)) == 0
            }
        )
    selected_ids = {
        image_id
        for image_ids in image_ids_by_category.values()
        for image_id in (
            image_ids
            if image_limit_per_category is None
            else image_ids[:image_limit_per_category]
        )
    }
    images = sorted(
        (image for image in source["images"] if int(image["id"]) in selected_ids),
        key=lambda image: int(image["id"]),
    )
    if not images:
        raise ValueError("No validation images contain exact PRD ground truth.")
    return images


def _segmentation_prediction_to_coco(
    prediction: Any,
    *,
    image_id: int,
    category_ids: dict[str, int],
) -> list[dict[str, Any]]:
    """Convert exact-category part instances into COCO result records."""
    results: list[dict[str, Any]] = []
    for instance in prediction.instances:
        category_id = category_ids.get(instance.category_label)
        if category_id is None:
            continue
        polygons = [polygon for polygon in instance.mask if len(polygon) >= 6]
        width = instance.box.x_max - instance.box.x_min
        height = instance.box.y_max - instance.box.y_min
        if not polygons or width <= 0.0 or height <= 0.0:
            continue
        results.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [
                    instance.box.x_min,
                    instance.box.y_min,
                    width,
                    height,
                ],
                "score": instance.confidence,
                "segmentation": polygons,
            }
        )
    return results


def _build_acceptance_result(
    *,
    evaluation: dict[str, Any],
    summary: dict[str, Any],
    coverage: list[Any],
) -> dict[str, Any]:
    """Attach a bounded PRD decision without overstating annotation coverage."""
    direct = evaluation["segm_direct_iou"]
    category_names = evaluation["categories"]
    category_recalls = [
        _finite_float(direct.get(f"Recall50-{category_name}"))
        for category_name in category_names
    ]
    evaluable_recalls = [value for value in category_recalls if value is not None]
    macro_recall = (
        sum(evaluable_recalls) / len(evaluable_recalls) if evaluable_recalls else None
    )
    exact_regions = [region for region in coverage if region.status == "exact"]
    uncovered_regions = [
        region.english_name for region in coverage if region.status != "exact"
    ]
    return {
        "benchmark": "prd_3_1_2_exact_ground_truth_accuracy",
        "measurement_boundary": {
            "included": (
                "deployment preprocessing, automatic person ROI, supervised "
                "Mask2Former inference, and mask postprocessing"
            ),
            "excluded": (
                "PRD regions without equivalent Fashionpedia masks and "
                "open-vocabulary fallback"
            ),
            "query_routing": (
                "deterministic query-to-label routing is validated separately "
                "by unit tests and API acceptance"
            ),
        },
        "coverage": {
            "prd_region_count": len(coverage),
            "exact_gt_prd_region_count": len(exact_regions),
            "exact_gt_prd_regions": [region.english_name for region in exact_regions],
            "unscored_prd_regions": uncovered_regions,
            "exact_source_categories": category_names,
        },
        "accuracy_contract": {
            "provisional_metric": "macro Recall50 across exact source categories",
            "target_percent": ACCURACY_TARGET_PERCENT,
            "measured_percent": macro_recall,
            "exact_gt_scope_passed": (
                macro_recall >= ACCURACY_TARGET_PERCENT
                if macro_recall is not None
                else False
            ),
            "overall_prd_accuracy_passed": None,
            "reason": (
                "Five PRD regions lack equivalent full validation ground truth; "
                "an exact-GT pass cannot establish overall eight-region accuracy."
            ),
        },
        "run": summary,
        "evaluation": evaluation,
    }


def _read_coco(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read a complete COCO mapping."""
    with path.open("r", encoding="utf-8") as file:
        source = json.load(file)
    required_fields = ("images", "annotations", "categories")
    if not isinstance(source, dict) or not all(
        isinstance(source.get(field), list) for field in required_fields
    ):
        raise ValueError(f"Expected a complete COCO mapping: {path}")
    return source


def _print_progress(
    *,
    completed: int,
    total: int,
    prediction_count: int,
    started_at: float,
) -> None:
    """Print one line suitable for live nohup monitoring."""
    elapsed = time.perf_counter() - started_at
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    eta = (total - completed) / rate if rate > 0.0 else 0.0
    print(
        f"[{completed}/{total}] predictions={prediction_count} "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
        flush=True,
    )


def _finite_float(value: Any) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats before strict JSON serialization."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _resolve_path(path: str, resolver: Any) -> Path:
    """Use absolute paths directly and resolve project-relative paths."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolver(candidate)


if __name__ == "__main__":
    main()
