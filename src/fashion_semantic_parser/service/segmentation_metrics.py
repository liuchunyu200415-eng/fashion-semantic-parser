"""PRD-focused metrics for garment instance segmentation."""

from typing import Any

_COCO_AREA_RANGES = {
    "small": (0.0, float(32**2)),
    "medium": (float(32**2), float(96**2)),
    "large": (float(96**2), float("inf")),
}


def _coco_ap_at_iou(
    coco_eval: Any,
    iou_threshold: float,
    category_index: int | None = None,
) -> float:
    """Return COCO average precision at one exact IoU threshold."""
    import numpy as np

    iou_thresholds = np.asarray(coco_eval.params.iouThrs, dtype=float)
    threshold_indices = np.flatnonzero(
        np.isclose(iou_thresholds, iou_threshold, atol=1e-6)
    )
    precision = coco_eval.eval.get("precision")
    if len(threshold_indices) == 0 or precision is None:
        return float("nan")

    values = np.asarray(precision)[threshold_indices[0], :, :, 0, -1]
    if category_index is not None:
        if category_index >= values.shape[1]:
            return float("nan")
        values = values[:, category_index]
    valid_values = values[values > -1]
    if valid_values.size == 0:
        return float("nan")
    return float(np.mean(valid_values) * 100.0)


def _coco_matched_mask_iou_metrics(
    coco_eval: Any,
    class_names: list[str] | None = None,
    match_iou_threshold: float = 0.50,
    target_iou_threshold: float = 0.85,
) -> dict[str, float]:
    """Report direct one-to-one mask IoU metrics from a COCO evaluation."""
    category_stats: list[dict[str, Any]] = []
    for category_id in coco_eval.params.catIds:
        category_stats.append(
            _coco_category_mask_iou_stats(
                coco_eval,
                category_id=category_id,
                match_iou_threshold=match_iou_threshold,
            )
        )

    evaluable_stats = [
        stats for stats in category_stats if stats["ground_truth_count"] > 0
    ]
    all_matched_ious = [
        iou for stats in evaluable_stats for iou in stats["matched_ious"]
    ]
    results = _summarize_mask_iou_matches(
        matched_ious=all_matched_ious,
        ground_truth_count=sum(
            stats["ground_truth_count"] for stats in evaluable_stats
        ),
        prediction_count=sum(stats["prediction_count"] for stats in evaluable_stats),
        match_iou_threshold=match_iou_threshold,
        target_iou_threshold=target_iou_threshold,
    )

    area_metrics = (
        "MatchedCount",
        "GroundTruthCount",
        "MatchedMeanIoU",
        "AllGTMeanIoU",
        "Recall50",
        "AllGTIoU85Rate",
    )
    for area_name in _COCO_AREA_RANGES:
        matched_ious = [
            iou
            for stats in evaluable_stats
            for iou in stats["area_stats"][area_name]["matched_ious"]
        ]
        area_summary = _summarize_mask_iou_matches(
            matched_ious=matched_ious,
            ground_truth_count=sum(
                stats["area_stats"][area_name]["ground_truth_count"]
                for stats in evaluable_stats
            ),
            prediction_count=0,
            match_iou_threshold=match_iou_threshold,
            target_iou_threshold=target_iou_threshold,
        )
        for metric_name in area_metrics:
            results[f"{metric_name}-{area_name}"] = area_summary[metric_name]

    if class_names is None:
        return results

    for category_index, category_name in enumerate(class_names):
        if category_index >= len(category_stats):
            break
        stats = category_stats[category_index]
        category_summary = _summarize_mask_iou_matches(
            matched_ious=stats["matched_ious"],
            ground_truth_count=stats["ground_truth_count"],
            prediction_count=stats["prediction_count"],
            match_iou_threshold=match_iou_threshold,
            target_iou_threshold=target_iou_threshold,
        )
        for metric_name in area_metrics:
            results[f"{metric_name}-{category_name}"] = category_summary[metric_name]
    return results


def _coco_category_mask_iou_stats(
    coco_eval: Any,
    category_id: int,
    match_iou_threshold: float,
) -> dict[str, Any]:
    """Collect direct mask IoU matches for one COCO category."""
    import numpy as np

    ground_truth_count = 0
    prediction_count = 0
    matched_ious: list[float] = []
    area_stats = {
        area_name: {"matched_ious": [], "ground_truth_count": 0}
        for area_name in _COCO_AREA_RANGES
    }
    max_detections = int(coco_eval.params.maxDets[-1])

    for image_id in coco_eval.params.imgIds:
        key = (image_id, category_id)
        ground_truth = list(coco_eval._gts.get(key, []))
        valid_gt_indices = [
            index
            for index, annotation in enumerate(ground_truth)
            if not annotation.get("ignore", 0) and not annotation.get("iscrowd", 0)
        ]
        detection_count = min(len(coco_eval._dts.get(key, [])), max_detections)
        ground_truth_count += len(valid_gt_indices)
        prediction_count += detection_count
        valid_ground_truth = [ground_truth[index] for index in valid_gt_indices]
        for annotation in valid_ground_truth:
            area_name = _coco_area_name(annotation)
            if area_name is not None:
                area_stats[area_name]["ground_truth_count"] += 1
        if not valid_gt_indices or detection_count == 0:
            continue

        iou_matrix = np.asarray(coco_eval.ious.get(key, []), dtype=float)
        if iou_matrix.size == 0:
            continue
        if iou_matrix.ndim != 2:
            expected_size = detection_count * len(ground_truth)
            if iou_matrix.size != expected_size:
                continue
            iou_matrix = iou_matrix.reshape(detection_count, len(ground_truth))
        iou_matrix = iou_matrix[:detection_count, valid_gt_indices]
        matches = _greedy_match_iou_pairs(
            iou_matrix,
            min_iou=match_iou_threshold,
        )
        matched_ious.extend(iou for _, _, iou in matches)
        for _, ground_truth_index, iou in matches:
            area_name = _coco_area_name(valid_ground_truth[ground_truth_index])
            if area_name is not None:
                area_stats[area_name]["matched_ious"].append(iou)

    return {
        "matched_ious": matched_ious,
        "ground_truth_count": ground_truth_count,
        "prediction_count": prediction_count,
        "area_stats": area_stats,
    }


def _greedy_match_ious(iou_matrix: Any, min_iou: float = 0.50) -> list[float]:
    """Greedily make one-to-one prediction/ground-truth matches by mask IoU."""
    return [
        iou
        for _, _, iou in _greedy_match_iou_pairs(
            iou_matrix,
            min_iou=min_iou,
        )
    ]


def _greedy_match_iou_pairs(
    iou_matrix: Any,
    min_iou: float = 0.50,
) -> list[tuple[int, int, float]]:
    """Return one-to-one prediction, ground-truth, and mask-IoU matches."""
    import numpy as np

    matrix = np.asarray(iou_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return []
    candidates = np.argwhere(matrix >= min_iou)
    if candidates.size == 0:
        return []

    candidate_ious = matrix[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(-candidate_ious, kind="stable")
    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for candidate_index in order:
        prediction_index, ground_truth_index = candidates[candidate_index]
        prediction_index = int(prediction_index)
        ground_truth_index = int(ground_truth_index)
        if (
            prediction_index in used_predictions
            or ground_truth_index in used_ground_truth
        ):
            continue
        used_predictions.add(prediction_index)
        used_ground_truth.add(ground_truth_index)
        matches.append(
            (
                prediction_index,
                ground_truth_index,
                float(matrix[prediction_index, ground_truth_index]),
            )
        )
    return matches


def _coco_area_name(annotation: dict[str, Any]) -> str | None:
    """Map a COCO annotation to its standard small, medium, or large bucket."""
    area = annotation.get("area")
    if area is None:
        bbox = annotation.get("bbox")
        if bbox is None or len(bbox) < 4:
            return None
        area = float(bbox[2]) * float(bbox[3])
    area = float(area)
    for area_name, (minimum, maximum) in _COCO_AREA_RANGES.items():
        if minimum <= area < maximum:
            return area_name
    return None


def _summarize_mask_iou_matches(
    matched_ious: list[float],
    ground_truth_count: int,
    prediction_count: int,
    match_iou_threshold: float = 0.50,
    target_iou_threshold: float = 0.85,
) -> dict[str, float]:
    """Summarize mask matches as percentages consistent with COCO metrics."""
    import numpy as np

    matched_count = len(matched_ious)
    target_count = sum(iou >= target_iou_threshold for iou in matched_ious)
    return {
        "MatchedCount": float(matched_count),
        "GroundTruthCount": float(ground_truth_count),
        "PredictionCount": float(prediction_count),
        "MatchIoUThreshold": match_iou_threshold * 100.0,
        "TargetIoUThreshold": target_iou_threshold * 100.0,
        "MatchedMeanIoU": _percentage_or_nan(
            float(np.mean(matched_ious)) if matched_ious else None
        ),
        "MatchedMedianIoU": _percentage_or_nan(
            float(np.median(matched_ious)) if matched_ious else None
        ),
        "AllGTMeanIoU": _percentage_or_nan(
            sum(matched_ious) / ground_truth_count if ground_truth_count > 0 else None
        ),
        "Precision50": _percentage_or_nan(
            matched_count / prediction_count if prediction_count > 0 else None
        ),
        "Recall50": _percentage_or_nan(
            matched_count / ground_truth_count if ground_truth_count > 0 else None
        ),
        "MatchedIoU85Rate": _percentage_or_nan(
            target_count / matched_count if matched_count > 0 else None
        ),
        "AllGTIoU85Rate": _percentage_or_nan(
            target_count / ground_truth_count if ground_truth_count > 0 else None
        ),
    }


def _percentage_or_nan(value: float | None) -> float:
    """Convert a zero-to-one value to percent while preserving missing data."""
    if value is None:
        return float("nan")
    return float(value * 100.0)
