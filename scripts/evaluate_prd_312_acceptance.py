"""Evaluate the locked PRD 3.1.2 single-query Top-1 Mask contract."""

# Direct execution adds project paths before importing local modules.
# pylint: disable=import-outside-toplevel

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

DEFAULT_OUTPUT = "outputs/localization/prd_312_acceptance/metrics.json"


def add_src_to_python_path() -> None:
    """Add the project and local package when they are not installed."""
    project_root = Path(__file__).resolve().parents[1]
    for path in (project_root, project_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    """Parse final PRD acceptance evaluation arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a locked PRD 3.1.2 benchmark using only the first "
            "returned Mask for each complete natural-language query."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--responses-dir", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:  # pylint: disable=too-many-locals
    """Evaluate every required response and write a JSON-safe acceptance report."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import (
        resolve_project_path,
        to_project_relative_path,
    )
    from fashion_semantic_parser.dao.localization.prd_312_acceptance import (
        load_prd_312_acceptance_manifest,
    )

    manifest_path = resolve_project_path(args.manifest)
    responses_dir = resolve_project_path(args.responses_dir)
    output_path = resolve_project_path(args.output)
    manifest = load_prd_312_acceptance_manifest(manifest_path)
    _validate_response_coverage(manifest.cases, responses_dir)

    rows: list[dict[str, Any]] = []
    total = len(manifest.cases)
    for index, case in enumerate(manifest.cases, start=1):
        response_path = responses_dir / f"{case.id}.json"
        response = _read_json(response_path)
        image_path = resolve_project_path(case.image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read acceptance image: {case.image_path}")
        row = evaluate_acceptance_case(
            case=case,
            response=response,
            image_shape=(int(image.shape[0]), int(image.shape[1])),
            threshold=manifest.contract.mask_iou_threshold,
        )
        row["response_json"] = to_project_relative_path(response_path)
        rows.append(row)
        print(
            f"[{index}/{total}] id={case.id} "
            f"top1_iou={row['top1_mask_iou_percent']:.2f}% "
            f"passed={row['query_passed']}",
            flush=True,
        )

    report = build_acceptance_report(
        manifest_path=to_project_relative_path(manifest_path),
        manifest=manifest,
        responses_dir=to_project_relative_path(responses_dir),
        rows=rows,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized, flush=True)


def evaluate_acceptance_case(
    *,
    case: Any,
    response: Any,
    image_shape: tuple[int, int],
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Score only the first result against the union of referenced GT Masks."""
    if not isinstance(response, dict):
        raise ValueError("Acceptance response must be a JSON object.")
    if response.get("case_id") != case.id:
        raise ValueError("Acceptance response case_id does not match the manifest.")
    if response.get("query") != case.query:
        raise ValueError("Acceptance response query does not match the manifest.")
    regions = response.get("regions")
    if not isinstance(regions, list):
        raise ValueError("Acceptance response is missing a regions list.")
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive height and width.")

    inference_error = response.get("error")
    top1_mask = None
    if inference_error is None and regions:
        from scripts.evaluate_referring_localization import _segmentation_to_mask

        first_region = regions[0]
        if not isinstance(first_region, dict):
            raise ValueError("Acceptance Top-1 region must be a JSON object.")
        top1_mask = _segmentation_to_mask(
            first_region.get("mask"),
            height=height,
            width=width,
        )
    target_mask = _union_target_masks(
        case.targets,
        height=height,
        width=width,
    )
    mask_iou = _binary_mask_iou(top1_mask, target_mask)
    query_passed = _passes_mask_iou(mask_iou, threshold)
    return {
        "case_id": case.id,
        "query": case.query,
        "language": case.language,
        "primary_dimension": case.primary_dimension,
        "dimensions": case.dimensions,
        "novelty": case.novelty,
        "target_region": case.target_region,
        "reference_frame": case.reference_frame,
        "target_label": case.target_label,
        "target_count": len(case.targets),
        "prediction_count": len(regions),
        "top1_mask_valid": top1_mask is not None,
        "top1_mask_iou_percent": mask_iou * 100.0,
        "query_passed": query_passed,
        "inference_error": inference_error,
        "elapsed_seconds": _finite_float(response.get("elapsed_seconds")),
        "includes_model_load": bool(response.get("includes_model_load", False)),
    }


def build_acceptance_report(  # pylint: disable=too-many-locals
    *,
    manifest_path: str,
    manifest: Any,
    responses_dir: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build overall and locked-composition query success summaries."""
    primary_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    dimension_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    novelty_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    language_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    label_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    region_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    cardinality_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    composite_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        primary_groups[str(row["primary_dimension"])].append(row)
        for dimension in row["dimensions"]:
            dimension_groups[str(dimension)].append(row)
        novelty_groups[str(row["novelty"])].append(row)
        language_groups[str(row["language"])].append(row)
        label_groups[str(row["target_label"])].append(row)
        region_groups[str(row["target_region"])].append(row)
        cardinality = (
            "single_target" if int(row["target_count"]) == 1 else "multi_target"
        )
        cardinality_groups[cardinality].append(row)
        composite = len(set(row["dimensions"]) - {"basic"}) >= 2
        composite_groups["composite" if composite else "non_composite"].append(row)
    overall = summarize_acceptance_rows(rows)
    required_accuracy = float(manifest.contract.required_accuracy)
    accuracy = overall["query_accuracy"]
    return {
        "schema_version": 1,
        "benchmark_scope": "prd_3_1_2_final_acceptance",
        "manifest": manifest_path,
        "manifest_name": manifest.name,
        "responses_dir": responses_dir,
        "contract": manifest.contract.model_dump(mode="json"),
        "success_definition": {
            "result": "single_query_top1",
            "metric": "mask_iou",
            "comparison": "strictly_greater_than",
            "threshold": manifest.contract.mask_iou_threshold,
            "multi_target_policy": manifest.contract.multi_target_policy,
        },
        "overall": overall,
        "by_primary_dimension": _summarize_groups(primary_groups),
        "by_dimension": _summarize_groups(dimension_groups),
        "by_novelty": _summarize_groups(novelty_groups),
        "by_language": _summarize_groups(language_groups),
        "by_target_label": _summarize_groups(label_groups),
        "by_target_region": _summarize_groups(region_groups),
        "by_target_cardinality": _summarize_groups(cardinality_groups),
        "by_composite_status": _summarize_groups(composite_groups),
        "prd_accuracy_passed": (accuracy is not None and accuracy >= required_accuracy),
        "cases": rows,
    }


def summarize_acceptance_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize query-level success without Top-k or instance matching."""
    count = len(rows)
    passed = sum(row["query_passed"] is True for row in rows)
    ious = [float(row["top1_mask_iou_percent"]) for row in rows]
    error_count = sum(row["inference_error"] is not None for row in rows)
    return {
        "query_count": count,
        "passed_query_count": passed,
        "failed_query_count": count - passed,
        "query_accuracy": passed / count if count else None,
        "query_accuracy_percent": 100.0 * passed / count if count else None,
        "mean_top1_mask_iou_percent": float(np.mean(ious)) if ious else None,
        "empty_top1_count": sum(not row["top1_mask_valid"] for row in rows),
        "inference_error_count": error_count,
        "latency_seconds": _latency_summary(rows),
    }


def _union_target_masks(
    targets: list[Any],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Decode and union every instance referenced by one natural-language query."""
    from scripts.evaluate_referring_localization import _segmentation_to_mask

    result = np.zeros((height, width), dtype=bool)
    for target in targets:
        mask = _segmentation_to_mask(
            target.segmentation,
            height=height,
            width=width,
        )
        if mask is None:
            raise ValueError("Acceptance target Mask could not be decoded.")
        result = np.logical_or(result, mask)
    if not result.any():
        raise ValueError("Acceptance target union cannot be empty.")
    return cast(np.ndarray, np.asarray(result, dtype=bool))


def _binary_mask_iou(
    prediction: np.ndarray | None,
    target: np.ndarray,
) -> float:
    """Return zero for a missing Top-1 Mask and IoU otherwise."""
    if prediction is None:
        return 0.0
    union = np.logical_or(prediction, target).sum()
    if not union:
        return 0.0
    intersection = np.logical_and(prediction, target).sum()
    return float(intersection / union)


def _passes_mask_iou(mask_iou: float, threshold: float) -> bool:
    """Apply the locked strict-greater-than success boundary."""
    return mask_iou > threshold


def _summarize_groups(
    groups: defaultdict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Return deterministic summaries for one grouping axis."""
    return {
        name: summarize_acceptance_rows(group) for name, group in sorted(groups.items())
    }


def _latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep cold and warm request latency separate from accuracy."""
    cold = [
        float(row["elapsed_seconds"])
        for row in rows
        if row["elapsed_seconds"] is not None and row["includes_model_load"]
    ]
    warm = [
        float(row["elapsed_seconds"])
        for row in rows
        if row["elapsed_seconds"] is not None and not row["includes_model_load"]
    ]
    return {
        "cold_case_count": len(cold),
        "cold_first_case": cold[0] if cold else None,
        "warm_case_count": len(warm),
        "warm_mean": float(np.mean(warm)) if warm else None,
        "warm_p95": float(np.percentile(warm, 95)) if warm else None,
    }


def _validate_response_coverage(cases: list[Any], responses_dir: Path) -> None:
    """Reject incomplete or contaminated acceptance runs before scoring."""
    expected_ids = {case.id for case in cases}
    actual_ids = {path.stem for path in responses_dir.glob("*.json")}
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing or unexpected:
        raise ValueError(
            "Acceptance response coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _finite_float(value: Any) -> float | None:
    """Convert optional numeric latency without leaking NaN to JSON."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _read_json(path: Path) -> Any:
    """Read one required UTF-8 JSON response."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
