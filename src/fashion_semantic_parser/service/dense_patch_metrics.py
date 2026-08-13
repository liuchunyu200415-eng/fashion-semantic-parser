"""Training-only threshold selection metrics for dense patch Masks."""

import numpy as np


def patch_probability_metrics(
    probabilities: np.ndarray,
    target_fractions: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Score query-level binary patch Masks at one frozen threshold.

    Args:
        probabilities: Finite calibrated foreground probabilities shaped ``QxP``.
        target_fractions: Soft target patch fractions with the same shape.
        threshold: Foreground probability boundary in ``(0, 1)``.

    Returns:
        Query count, Recall50 numerator/rate, and mean patch IoU.

    Raises:
        ValueError: If arrays, targets, or threshold are invalid.
    """
    predicted_probabilities = np.asarray(probabilities, dtype=np.float32)
    targets = np.asarray(target_fractions, dtype=np.float32)
    if (
        predicted_probabilities.ndim != 2
        or predicted_probabilities.shape != targets.shape
        or not len(predicted_probabilities)
    ):
        raise ValueError(
            "Patch probabilities and targets must share a non-empty QxP shape."
        )
    if (
        not np.all(np.isfinite(predicted_probabilities))
        or np.any(predicted_probabilities < 0.0)
        or np.any(predicted_probabilities > 1.0)
        or not np.all(np.isfinite(targets))
        or np.any(targets < 0.0)
        or np.any(targets > 1.0)
    ):
        raise ValueError("Patch probabilities and target fractions must be in [0, 1].")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Patch probability threshold must be in (0, 1).")
    target_masks = targets > 0.0
    if not np.all(target_masks.any(axis=1)) or not np.all(
        np.logical_not(target_masks).any(axis=1)
    ):
        raise ValueError("Every patch query requires foreground and background.")
    predicted_masks = predicted_probabilities >= threshold
    intersections = np.logical_and(predicted_masks, target_masks).sum(axis=1)
    unions = np.logical_or(predicted_masks, target_masks).sum(axis=1)
    ious = intersections / np.maximum(unions, 1)
    return {
        "query_count": len(ious),
        "patch_recall50_count": int(np.sum(ious >= 0.50)),
        "patch_recall50": float(np.mean(ious >= 0.50)),
        "mean_patch_iou": float(np.mean(ious)),
    }


def select_patch_probability_threshold(
    probabilities: np.ndarray,
    target_fractions: np.ndarray,
    thresholds: tuple[float, ...],
) -> tuple[float, dict[str, dict[str, float | int]]]:
    """Freeze one threshold using training patch Masks only.

    Args:
        probabilities: Calibrated training foreground probabilities shaped ``QxP``.
        target_fractions: Training target patch fractions with the same shape.
        thresholds: Unique ascending candidate thresholds in ``(0, 1)``.

    Returns:
        Selected threshold and complete candidate metric audit. Selection
        maximizes Recall50, then mean IoU, then the tighter threshold.

    Raises:
        ValueError: If threshold candidates are empty, unordered, or invalid.
    """
    if (
        not thresholds
        or tuple(sorted(set(thresholds))) != thresholds
        or any(not 0.0 < value < 1.0 for value in thresholds)
    ):
        raise ValueError("Patch thresholds must be unique, ascending, and in (0, 1).")
    metrics = {
        f"{threshold:.3f}": patch_probability_metrics(
            probabilities,
            target_fractions,
            threshold=threshold,
        )
        for threshold in thresholds
    }
    selected = max(
        thresholds,
        key=lambda value: (
            float(metrics[f"{value:.3f}"]["patch_recall50"]),
            float(metrics[f"{value:.3f}"]["mean_patch_iou"]),
            value,
        ),
    )
    return selected, metrics
