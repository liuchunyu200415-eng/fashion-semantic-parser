"""Convert aligned dense DINOv2 patch features into image-space region Masks."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DenseMaskCandidate:
    """One score-quantile Mask and its derived image-coordinate Box."""

    quantile: float
    threshold: float
    mask: np.ndarray
    box: tuple[float, float, float, float] | None


def dense_similarity_scores(
    patch_features: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    """Return cosine similarity grids for normalized patch and query features.

    Args:
        patch_features: Finite ``HxWxD`` DINOv2 feature grid.
        query_features: Finite ``QxD`` projected text features.

    Returns:
        Float32 ``QxHxW`` cosine-similarity grids.

    Raises:
        ValueError: If feature dimensions or values are invalid.
    """
    patches = np.asarray(patch_features, dtype=np.float32)
    queries = np.asarray(query_features, dtype=np.float32)
    if patches.ndim != 3 or not patches.size:
        raise ValueError("Dense patch features must be a non-empty HxWxD array.")
    if queries.ndim != 2 or not len(queries):
        raise ValueError("Dense query features must be a non-empty QxD array.")
    if patches.shape[2] != queries.shape[1]:
        raise ValueError("Dense patch and query feature dimensions must match.")
    if not np.all(np.isfinite(patches)) or not np.all(np.isfinite(queries)):
        raise ValueError("Dense localization features must be finite.")
    patch_norms = np.linalg.norm(patches, axis=2, keepdims=True)
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if np.any(patch_norms <= 0.0) or np.any(query_norms <= 0.0):
        raise ValueError("Dense localization features cannot contain zero vectors.")
    normalized_patches = patches / patch_norms
    normalized_queries = queries / query_norms
    scores = np.einsum(
        "hwd,qd->qhw",
        normalized_patches,
        normalized_queries,
        optimize=True,
    )
    return np.asarray(scores, dtype=np.float32)


def quantile_mask_candidates(
    image_scores: np.ndarray,
    quantiles: tuple[float, ...],
) -> list[DenseMaskCandidate]:
    """Threshold one image-space similarity map at fixed score quantiles.

    Args:
        image_scores: Finite two-dimensional similarity map.
        quantiles: Unique ascending quantiles in the open interval ``(0, 1)``.

    Returns:
        Non-empty binary Mask candidates in quantile order.

    Raises:
        ValueError: If scores or quantiles are invalid.
    """
    scores = np.asarray(image_scores, dtype=np.float32)
    if scores.ndim != 2 or not scores.size or not np.all(np.isfinite(scores)):
        raise ValueError("Image similarity scores must be one finite 2D map.")
    if (
        not quantiles
        or tuple(sorted(set(quantiles))) != quantiles
        or any(not 0.0 < value < 1.0 for value in quantiles)
    ):
        raise ValueError("Score quantiles must be unique, ascending, and in (0, 1).")
    candidates = []
    for quantile in quantiles:
        threshold = float(np.quantile(scores, quantile))
        mask = np.asarray(scores >= threshold, dtype=bool)
        candidates.append(
            DenseMaskCandidate(
                quantile=quantile,
                threshold=threshold,
                mask=mask,
                box=mask_box(mask),
            )
        )
    return candidates


def calibrated_dense_probabilities(
    similarity_scores: np.ndarray,
    *,
    logit_scale: float,
    logit_bias: float,
) -> np.ndarray:
    """Convert cosine similarity maps into calibrated foreground probabilities.

    Args:
        similarity_scores: Finite similarity maps with any non-empty shape.
        logit_scale: Positive finite cosine scale from dense patch training.
        logit_bias: Finite foreground logit bias from dense patch training.

    Returns:
        Float32 probabilities with the same shape as ``similarity_scores``.

    Raises:
        ValueError: If scores or calibration values are invalid.
    """
    scores = np.asarray(similarity_scores, dtype=np.float32)
    if not scores.size or not np.all(np.isfinite(scores)):
        raise ValueError("Dense similarity scores must be non-empty and finite.")
    if not np.isfinite(logit_scale) or logit_scale <= 0.0:
        raise ValueError("Dense logit scale must be positive and finite.")
    if not np.isfinite(logit_bias):
        raise ValueError("Dense logit bias must be finite.")
    logits = np.clip(scores * logit_scale + logit_bias, -80.0, 80.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return np.asarray(probabilities, dtype=np.float32)


def binary_mask_iou(target_mask: np.ndarray, prediction_mask: np.ndarray) -> float:
    """Return IoU for equal-sized Masks while retaining empty misses.

    Args:
        target_mask: Non-empty binary ground-truth Mask.
        prediction_mask: Binary prediction Mask with the same shape.

    Returns:
        Intersection over union in ``[0, 1]``.

    Raises:
        ValueError: If Mask geometry or ground truth is invalid.
    """
    target = np.asarray(target_mask, dtype=bool)
    prediction = np.asarray(prediction_mask, dtype=bool)
    if target.ndim != 2 or target.shape != prediction.shape or not target.any():
        raise ValueError("Mask IoU requires equal 2D shapes and non-empty GT.")
    intersection = int(np.logical_and(target, prediction).sum())
    union = int(np.logical_or(target, prediction).sum())
    return intersection / union if union else 0.0


def mask_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return the tight ``xyxy`` Box for one binary Mask.

    Args:
        mask: Two-dimensional binary Mask.

    Returns:
        Tight Box or ``None`` when the Mask is empty.

    Raises:
        ValueError: If the Mask is not two-dimensional.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("Mask Box requires one two-dimensional Mask.")
    y_values, x_values = np.nonzero(binary)
    if not len(x_values):
        return None
    return (
        float(x_values.min()),
        float(y_values.min()),
        float(x_values.max() + 1),
        float(y_values.max() + 1),
    )


def box_iou(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> float:
    """Return IoU for two optional ``xyxy`` boxes.

    Args:
        first: First positive-area Box or ``None``.
        second: Second positive-area Box or ``None``.

    Returns:
        Intersection over union, with missing boxes scored as zero.
    """
    if first is None or second is None:
        return 0.0
    x_min = max(first[0], second[0])
    y_min = max(first[1], second[1])
    x_max = min(first[2], second[2])
    y_max = min(first[3], second[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0,
        second[3] - second[1],
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0
