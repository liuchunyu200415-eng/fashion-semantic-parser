"""Scan score and Top-K settings on saved PRD 3.1.2 candidates."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

DEFAULT_VALIDATION_JSON = (
    "data/processed/autodl/localization/fashionpedia_parts_validation.json"
)
DEFAULT_THRESHOLDS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
DEFAULT_TOP_K = (1, 3, 5, 10, 0)
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
    """Parse saved-candidate operating-point scan arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate score thresholds and per-image/per-category Top-K values "
            "without rerunning the localization model."
        )
    )
    parser.add_argument("--val-json", default=DEFAULT_VALIDATION_JSON)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--run-summary", default=None)
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        dest="thresholds",
        help="Repeat to replace the default threshold grid.",
    )
    parser.add_argument(
        "--top-k",
        action="append",
        type=int,
        dest="top_k_values",
        help="Repeat to replace the default grid; zero means unlimited.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    """Evaluate all requested operating points and save compact diagnostics."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.taxonomy import (
        PRD_LOCALIZATION_REGION_COVERAGE,
    )
    from scripts.benchmark_localization_accuracy import (
        _exact_prd_source_categories,
    )
    from scripts.evaluate_localization_predictions import (
        evaluate_localization_categories,
    )

    thresholds = _normalize_thresholds(args.thresholds or DEFAULT_THRESHOLDS)
    top_k_values = _normalize_top_k(args.top_k_values or DEFAULT_TOP_K)
    validation_path = _resolve_path(args.val_json, resolve_project_path)
    prediction_path = _resolve_path(args.predictions, resolve_project_path)
    summary_path = (
        _resolve_path(args.run_summary, resolve_project_path)
        if args.run_summary
        else prediction_path.with_name(f"{prediction_path.stem}_summary.json")
    )
    output_path = _resolve_path(args.output, resolve_project_path)
    category_names = _exact_prd_source_categories(PRD_LOCALIZATION_REGION_COVERAGE)

    rows: list[dict[str, Any]] = []
    total = len(thresholds) * len(top_k_values)
    for index, (threshold, top_k) in enumerate(
        ((threshold, top_k) for threshold in thresholds for top_k in top_k_values),
        start=1,
    ):
        evaluation = evaluate_localization_categories(
            validation_path=validation_path,
            prediction_path=prediction_path,
            category_names=category_names,
            score_threshold=threshold,
            top_k=top_k,
            summary_path=summary_path,
        )
        row = _summarize_operating_point(evaluation)
        rows.append(row)
        print(
            f"[{index}/{total}] threshold={threshold:.2f} "
            f"top_k={_display_top_k(top_k)} "
            f"kept={row['candidate_count']} "
            f"macro_R50={_display_metric(row['macro_recall50'])} "
            f"P50={_display_metric(row['precision50'])} "
            f"R50={_display_metric(row['recall50'])} "
            f"F1={_display_metric(row['f1_50'])}",
            flush=True,
        )

    result = _build_scan_result(
        rows=rows,
        validation_path=validation_path,
        prediction_path=prediction_path,
        summary_path=summary_path,
        category_names=category_names,
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized, flush=True)


def _normalize_thresholds(values: Any) -> list[float]:
    """Validate, deduplicate, and sort score thresholds."""
    thresholds = sorted({float(value) for value in values})
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Thresholds must be between 0 and 1.")
    return thresholds


def _normalize_top_k(values: Any) -> list[int | None]:
    """Validate Top-K values and convert zero to unlimited."""
    normalized = {None if int(value) == 0 else int(value) for value in values}
    if not normalized or any(value is not None and value < 1 for value in normalized):
        raise ValueError("Top-K values must be positive, or zero for unlimited.")
    return sorted(normalized, key=lambda value: math.inf if value is None else value)


def _summarize_operating_point(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Reduce one full evaluation to selection and per-class diagnostics."""
    direct = evaluation["segm_direct_iou"]
    category_names = evaluation["categories"]
    per_category = {
        category_name: _finite_float(direct.get(f"Recall50-{category_name}"))
        for category_name in category_names
    }
    recalls = [value for value in per_category.values() if value is not None]
    macro_recall = sum(recalls) / len(recalls) if recalls else None
    return {
        "score_threshold": evaluation["score_threshold"],
        "top_k": evaluation["top_k"],
        "candidate_count": evaluation["candidate_count_after_filter"],
        "macro_recall50": macro_recall,
        "precision50": _finite_float(direct.get("Precision50")),
        "recall50": _finite_float(direct.get("Recall50")),
        "f1_50": _finite_float(direct.get("F1_50")),
        "matched_mean_iou": _finite_float(direct.get("MatchedMeanIoU")),
        "all_gt_mean_iou": _finite_float(direct.get("AllGTMeanIoU")),
        "all_gt_iou85_rate": _finite_float(direct.get("AllGTIoU85Rate")),
        "ap50": _finite_float(evaluation["segm_coco"].get("AP50")),
        "per_category_recall50": per_category,
    }


def _build_scan_result(
    *,
    rows: list[dict[str, Any]],
    validation_path: Path,
    prediction_path: Path,
    summary_path: Path,
    category_names: list[str],
) -> dict[str, Any]:
    """Build the scan report and select recall/F1 operating points."""
    if not rows:
        raise ValueError("At least one operating point is required.")
    best_macro = max(rows, key=lambda row: _selection_value(row["macro_recall50"]))
    best_f1 = max(rows, key=lambda row: _selection_value(row["f1_50"]))
    return {
        "benchmark": "prd_3_1_2_operating_point_scan",
        "validation_json": str(validation_path),
        "predictions_json": str(prediction_path),
        "run_summary_json": str(summary_path) if summary_path.is_file() else None,
        "exact_source_categories": category_names,
        "target_macro_recall50": ACCURACY_TARGET_PERCENT,
        "any_operating_point_passed": any(
            _selection_value(row["macro_recall50"]) >= ACCURACY_TARGET_PERCENT
            for row in rows
        ),
        "best_macro_recall": best_macro,
        "best_micro_f1": best_f1,
        "rows": rows,
    }


def _selection_value(value: Any) -> float:
    """Map missing metrics below every finite candidate for selection."""
    numeric = _finite_float(value)
    return numeric if numeric is not None else -math.inf


def _finite_float(value: Any) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _display_top_k(top_k: int | None) -> str:
    """Format unlimited Top-K compactly for progress output."""
    return "all" if top_k is None else str(top_k)


def _display_metric(value: Any) -> str:
    """Format one optional percentage for progress output."""
    numeric = _finite_float(value)
    return "N/A" if numeric is None else f"{numeric:.2f}"


def _resolve_path(path: str, resolver: Any) -> Path:
    """Use absolute paths directly and resolve project-relative paths."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolver(candidate)


if __name__ == "__main__":
    main()
