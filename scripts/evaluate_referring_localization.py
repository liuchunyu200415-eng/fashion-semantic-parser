"""Evaluate saved open-language localization responses at the query level."""

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

DEFAULT_OUTPUT = "outputs/localization/referring_smoke/metrics.json"


def add_src_to_python_path() -> None:
    """Add the project and local package when they are not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    for path in (project_root, src_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    """Parse saved-response evaluation arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate referring-expression cases with one-to-one target "
            "matching and grouped query-level metrics."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--responses-dir", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-iou", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    """Evaluate every saved case and write a JSON-safe feasibility report."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import (
        resolve_project_path,
        to_project_relative_path,
    )
    from fashion_semantic_parser.dao.localization.referring_smoke import (
        load_referring_smoke_manifest,
    )

    if not 0.0 <= args.min_iou <= 1.0:
        raise ValueError("--min-iou must be between 0 and 1.")
    manifest_path = resolve_project_path(args.manifest)
    responses_dir = resolve_project_path(args.responses_dir)
    output_path = resolve_project_path(args.output)
    manifest = load_referring_smoke_manifest(manifest_path)
    expected_response_ids = {case.id for case in manifest.cases}
    actual_response_ids = {path.stem for path in responses_dir.glob("*.json")}
    unexpected_response_ids = sorted(actual_response_ids - expected_response_ids)
    if unexpected_response_ids:
        raise ValueError(
            "Responses directory contains cases outside the manifest: "
            f"{unexpected_response_ids}"
        )

    rows: list[dict[str, Any]] = []
    total = len(manifest.cases)
    for index, case in enumerate(manifest.cases, start=1):
        response_path = responses_dir / f"{case.id}.json"
        response = _read_json(response_path)
        image_path = resolve_project_path(case.image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read benchmark image: {case.image_path}")
        row = evaluate_referring_case(
            case=case,
            response=response,
            image_shape=(int(image.shape[0]), int(image.shape[1])),
            min_iou=args.min_iou,
        )
        row["response_json"] = to_project_relative_path(response_path)
        rows.append(row)
        print(
            f"[{index}/{total}] id={case.id} scored={row['scored']} "
            f"matched={row['matched_count']}/{row['ground_truth_count']} "
            f"query_passed={row['query_passed']}",
            flush=True,
        )

    report = build_referring_report(
        manifest_path=to_project_relative_path(manifest_path),
        manifest_name=manifest.name,
        responses_dir=to_project_relative_path(responses_dir),
        rows=rows,
        min_iou=args.min_iou,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized, flush=True)


def evaluate_referring_case(
    *,
    case: Any,
    response: Any,
    image_shape: tuple[int, int],
    min_iou: float = 0.50,
) -> dict[str, Any]:
    """Evaluate one expression without collapsing it to a category label."""
    if not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iou must be between 0 and 1.")
    if not isinstance(response, dict):
        raise ValueError("Referring response must be a JSON object.")
    if response.get("case_id") != case.id:
        raise ValueError(
            f"Response case_id {response.get('case_id')!r} does not match {case.id!r}."
        )
    if response.get("error") is not None:
        raise RuntimeError(
            f"Referring inference did not complete for case {case.id!r}: "
            f"{response['error']}"
        )
    regions = response.get("regions")
    if not isinstance(regions, list):
        raise ValueError(f"Response for {case.id!r} is missing a regions list.")
    prediction_count = len(regions)
    expected_count_passed = (
        None if case.expected_count is None else prediction_count == case.expected_count
    )
    common = {
        "case_id": case.id,
        "query": case.query,
        "grounding_prompt": case.grounding_prompt,
        "dimensions": case.dimensions,
        "novelty": case.novelty,
        "reference_frame": case.reference_frame,
        "annotation_status": case.annotation_status,
        "expected_count": case.expected_count,
        "contrast_set_id": case.contrast_set_id,
        "prediction_count": prediction_count,
        "empty_prediction": prediction_count == 0,
        "extra_prediction_count": (
            max(0, prediction_count - case.expected_count)
            if case.expected_count is not None
            else None
        ),
        "expected_count_passed": expected_count_passed,
        "elapsed_seconds": _finite_float(response.get("elapsed_seconds")),
        "includes_model_load": bool(response.get("includes_model_load", False)),
    }

    if case.annotation_status == "unlabelled":
        return {
            **common,
            "scored": False,
            "metric": None,
            "ground_truth_count": 0,
            "matched_count": 0,
            "matched_ious_percent": [],
            "target_recall_passed": None,
            "query_passed": None,
        }
    if case.annotation_status == "negative":
        passed = prediction_count == 0
        return {
            **common,
            "scored": True,
            "metric": "negative",
            "ground_truth_count": 0,
            "matched_count": 0,
            "matched_ious_percent": [],
            "target_recall_passed": passed,
            "query_passed": passed,
        }

    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive height and width.")
    if case.annotation_status == "mask":
        metric = "mask_iou"
        iou_matrix = _mask_iou_matrix(
            regions,
            case.targets,
            height=height,
            width=width,
        )
    else:
        metric = "box_iou"
        iou_matrix = _box_iou_matrix(regions, case.targets)

    from fashion_semantic_parser.service.segmentation_metrics import (
        _greedy_match_iou_pairs,
    )

    matches = _greedy_match_iou_pairs(iou_matrix, min_iou=min_iou)
    matched_ious = [iou for _, _, iou in matches]
    ground_truth_count = len(case.targets)
    matched_count = len(matches)
    target_recall_passed = matched_count == ground_truth_count
    query_passed = target_recall_passed and expected_count_passed is not False
    return {
        **common,
        "scored": True,
        "metric": metric,
        "ground_truth_count": ground_truth_count,
        "matched_count": matched_count,
        "matched_ious_percent": [iou * 100.0 for iou in matched_ious],
        "matched_mean_iou_percent": (
            100.0 * float(np.mean(matched_ious)) if matched_ious else None
        ),
        "all_gt_mean_iou_percent": (
            100.0 * sum(matched_ious) / ground_truth_count
            if ground_truth_count
            else None
        ),
        "precision50_percent": _percent(matched_count, prediction_count),
        "recall50_percent": _percent(matched_count, ground_truth_count),
        "target_recall_passed": target_recall_passed,
        "query_passed": query_passed,
        "matches": [
            {
                "prediction_index": prediction_index,
                "ground_truth_index": ground_truth_index,
                "iou_percent": iou * 100.0,
            }
            for prediction_index, ground_truth_index, iou in matches
        ],
    }


def build_referring_report(
    *,
    manifest_path: str,
    manifest_name: str,
    responses_dir: str,
    rows: list[dict[str, Any]],
    min_iou: float,
) -> dict[str, Any]:
    """Build bounded overall, linguistic-dimension, and novelty summaries."""
    dimensions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    novelty_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    contrast_sets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for dimension in row["dimensions"]:
            dimensions[str(dimension)].append(row)
        novelty_groups[str(row["novelty"])].append(row)
        if row["contrast_set_id"] is not None:
            contrast_sets[str(row["contrast_set_id"])].append(row)
    return {
        "schema_version": 1,
        "benchmark_scope": "open_language_feasibility_smoke",
        "manifest": manifest_path,
        "manifest_name": manifest_name,
        "responses_dir": responses_dir,
        "match_iou_threshold_percent": min_iou * 100.0,
        "overall": summarize_referring_rows(rows),
        "by_dimension": {
            key: summarize_referring_rows(group)
            for key, group in sorted(dimensions.items())
        },
        "by_novelty": {
            key: summarize_referring_rows(group)
            for key, group in sorted(novelty_groups.items())
        },
        "contrast_sets": {
            key: _summarize_contrast_set(group)
            for key, group in sorted(contrast_sets.items())
        },
        "cases": rows,
        "accuracy_boundary": {
            "prd_accuracy_passed": None,
            "reason": (
                "This bounded smoke set selects a model direction; it is not "
                "the independent PRD acceptance set."
            ),
        },
    }


def summarize_referring_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate query success separately from positive-instance recall."""
    scored_rows = [row for row in rows if row["scored"]]
    positive_rows = [
        row for row in scored_rows if row["annotation_status"] in ("mask", "box")
    ]
    negative_rows = [
        row for row in scored_rows if row["annotation_status"] == "negative"
    ]
    count_rows = [row for row in rows if row["expected_count_passed"] is not None]
    mask_rows = [row for row in positive_rows if row["annotation_status"] == "mask"]
    box_rows = [row for row in positive_rows if row["annotation_status"] == "box"]
    return {
        "case_count": len(rows),
        "scored_case_count": len(scored_rows),
        "unlabelled_case_count": len(rows) - len(scored_rows),
        "mask_scored_case_count": sum(
            row["annotation_status"] == "mask" for row in scored_rows
        ),
        "box_scored_case_count": sum(
            row["annotation_status"] == "box" for row in scored_rows
        ),
        "negative_case_count": len(negative_rows),
        "query_success_rate_percent": _rate(
            scored_rows,
            lambda row: row["query_passed"] is True,
        ),
        "target_recall_pass_rate_percent": _rate(
            positive_rows,
            lambda row: row["target_recall_passed"] is True,
        ),
        "expected_count_pass_rate_percent": _rate(
            count_rows,
            lambda row: row["expected_count_passed"] is True,
        ),
        "negative_query_success_rate_percent": _rate(
            negative_rows,
            lambda row: row["query_passed"] is True,
        ),
        "negative_false_positive_rate_percent": _rate(
            negative_rows,
            lambda row: int(row["prediction_count"]) > 0,
        ),
        "empty_prediction_rate_percent": _rate(
            rows,
            lambda row: row["empty_prediction"] is True,
        ),
        "mean_extra_prediction_count": _mean_optional_counts(
            row["extra_prediction_count"] for row in rows
        ),
        "positive_match_counts": _summarize_positive_match_counts(positive_rows),
        "mask_instance_metrics": _summarize_instance_rows(mask_rows),
        "box_instance_metrics": _summarize_instance_rows(box_rows),
        "latency_seconds": _partitioned_latency_summary(rows),
    }


def _summarize_positive_match_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine thresholded counts while leaving Mask and Box IoU separate."""
    ground_truth_count = sum(int(row["ground_truth_count"]) for row in rows)
    prediction_count = sum(int(row["prediction_count"]) for row in rows)
    matched_count = sum(int(row["matched_count"]) for row in rows)
    precision = matched_count / prediction_count if prediction_count else None
    recall = matched_count / ground_truth_count if ground_truth_count else None
    return {
        "ground_truth_count": ground_truth_count,
        "prediction_count": prediction_count,
        "matched_count": matched_count,
        "precision50_percent": _optional_percentage(precision),
        "recall50_percent": _optional_percentage(recall),
        "f1_50_percent": _f1_percent(precision, recall),
    }


def _summarize_instance_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one homogeneous Mask-IoU or Box-IoU evaluation scope."""
    counts = _summarize_positive_match_counts(rows)
    ground_truth_count = int(counts["ground_truth_count"])
    matched_ious = [
        float(iou) / 100.0 for row in rows for iou in row["matched_ious_percent"]
    ]
    return {
        "eligible_case_count": len(rows),
        **counts,
        "matched_mean_iou_percent": (
            100.0 * float(np.mean(matched_ious)) if matched_ious else None
        ),
        "all_gt_mean_iou_percent": (
            100.0 * sum(matched_ious) / ground_truth_count
            if ground_truth_count
            else None
        ),
    }


def _mask_iou_matrix(
    regions: list[Any],
    targets: list[Any],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Return prediction-by-target mask IoUs, retaining invalid predictions."""
    prediction_masks = [
        _segmentation_to_mask(region.get("mask"), height=height, width=width)
        for region in regions
    ]
    target_masks = [
        _segmentation_to_mask(target.segmentation, height=height, width=width)
        for target in targets
    ]
    matrix = np.zeros((len(prediction_masks), len(target_masks)), dtype=float)
    for prediction_index, prediction_mask in enumerate(prediction_masks):
        if prediction_mask is None:
            continue
        for target_index, target_mask in enumerate(target_masks):
            if target_mask is None:
                continue
            union = np.logical_or(prediction_mask, target_mask).sum()
            if union:
                intersection = np.logical_and(prediction_mask, target_mask).sum()
                matrix[prediction_index, target_index] = intersection / union
    return matrix


def _box_iou_matrix(regions: list[Any], targets: list[Any]) -> np.ndarray:
    """Return prediction-by-target box IoUs for xyxy boxes."""
    matrix = np.zeros((len(regions), len(targets)), dtype=float)
    for prediction_index, region in enumerate(regions):
        prediction_box = _box_tuple(region.get("box"))
        if prediction_box is None:
            continue
        for target_index, target in enumerate(targets):
            target_box = _box_tuple(target.box)
            if target_box is not None:
                matrix[prediction_index, target_index] = _box_iou(
                    prediction_box,
                    target_box,
                )
    return matrix


def _segmentation_to_mask(
    segmentation: Any,
    *,
    height: int,
    width: int,
) -> np.ndarray | None:
    """Decode polygon or COCO RLE segmentation into one binary mask."""
    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=np.uint8)
        polygons = []
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                continue
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, max(width - 1, 0))
            points[:, 1] = np.clip(points[:, 1], 0, max(height - 1, 0))
            polygons.append(np.rint(points).astype(np.int32))
        if not polygons:
            return None
        cv2.fillPoly(mask, polygons, (1,))
        return mask.astype(bool)
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError(
                "pycocotools is required to decode RLE masks."
            ) from error
        rle = segmentation
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        decoded = np.asarray(mask_utils.decode(rle), dtype=bool)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return decoded
    return None


def _box_tuple(box: Any) -> tuple[float, float, float, float] | None:
    """Normalize typed or dictionary xyxy boxes."""
    if box is None:
        return None
    if hasattr(box, "model_dump"):
        box = box.model_dump()
    if not isinstance(box, dict):
        return None
    try:
        result = (
            float(box["x_min"]),
            float(box["y_min"]),
            float(box["x_max"]),
            float(box["y_max"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return intersection-over-union for two xyxy boxes."""
    x_min = max(first[0], second[0])
    y_min = max(first[1], second[1])
    x_max = min(first[2], second[2])
    y_max = min(first[3], second[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _partitioned_latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the run-level cold marker even inside filtered metric groups."""
    cold_values = [
        float(row["elapsed_seconds"])
        for row in rows
        if row["elapsed_seconds"] is not None and row["includes_model_load"]
    ]
    warm_values = [
        float(row["elapsed_seconds"])
        for row in rows
        if row["elapsed_seconds"] is not None and not row["includes_model_load"]
    ]
    all_values = cold_values + warm_values
    return {
        "cold_case_count": len(cold_values),
        "cold_first_case": cold_values[0] if cold_values else None,
        "warm_case_count": len(warm_values),
        "warm_mean": float(np.mean(warm_values)) if warm_values else None,
        "warm_p95": float(np.percentile(warm_values, 95)) if warm_values else None,
        "all_wall_clock_mean": (float(np.mean(all_values)) if all_values else None),
        "all_wall_clock_p95": (
            float(np.percentile(all_values, 95)) if all_values else None
        ),
    }


def _summarize_contrast_set(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require every modifier variant in one same-image set to succeed."""
    scored_rows = [row for row in rows if row["scored"]]
    return {
        "case_ids": [str(row["case_id"]) for row in rows],
        "case_count": len(rows),
        "scored_case_count": len(scored_rows),
        "all_cases_scored": len(scored_rows) == len(rows),
        "all_scored_cases_passed": (
            all(row["query_passed"] is True for row in scored_rows)
            if scored_rows
            else None
        ),
    }


def _finite_float(value: Any) -> float | None:
    """Convert one optional numeric value without leaking NaN to JSON."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _percent(numerator: int, denominator: int) -> float | None:
    """Return one percentage or None when the denominator is empty."""
    return 100.0 * numerator / denominator if denominator else None


def _optional_percentage(value: float | None) -> float | None:
    """Convert a zero-to-one value to percent while preserving missing data."""
    return 100.0 * value if value is not None else None


def _f1_percent(precision: float | None, recall: float | None) -> float | None:
    """Return F1 in percent when both positive-instance rates are defined."""
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 100.0 * 2.0 * precision * recall / (precision + recall)


def _mean_optional_counts(values: Iterable[int | None]) -> float | None:
    """Average defined per-query extra-candidate counts."""
    defined = [int(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def _rate(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> float | None:
    """Return the percentage of rows satisfying a boolean predicate."""
    return _percent(sum(predicate(row) for row in rows), len(rows))


def _read_json(path: Path) -> Any:
    """Read one required UTF-8 JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Referring response not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
