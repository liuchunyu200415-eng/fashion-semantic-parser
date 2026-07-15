"""PRD-focused metrics for garment instance segmentation."""

from typing import Any


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

    if class_names is None:
        return results

    per_category_metrics = (
        "MatchedCount",
        "GroundTruthCount",
        "MatchedMeanIoU",
        "AllGTMeanIoU",
        "Recall50",
        "AllGTIoU85Rate",
    )
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
        for metric_name in per_category_metrics:
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
        matched_ious.extend(_greedy_match_ious(iou_matrix, min_iou=match_iou_threshold))

    return {
        "matched_ious": matched_ious,
        "ground_truth_count": ground_truth_count,
        "prediction_count": prediction_count,
    }


def _greedy_match_ious(iou_matrix: Any, min_iou: float = 0.50) -> list[float]:
    """Greedily make one-to-one prediction/ground-truth matches by mask IoU."""
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
    matched_ious: list[float] = []
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
        matched_ious.append(float(matrix[prediction_index, ground_truth_index]))
    return matched_ious


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
