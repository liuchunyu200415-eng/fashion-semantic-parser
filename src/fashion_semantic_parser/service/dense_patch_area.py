"""Query-conditioned target-area control for dense patch localization."""

from typing import Any

import numpy as np


def build_query_area_predictor(
    feature_dimension: int,
    settings: Any,
) -> Any:
    """Build an open-query predictor for target foreground area.

    Args:
        feature_dimension: Positive shared DINOv2/text feature dimension.
        settings: Dense settings with area hidden dimension and dropout.

    Returns:
        A PyTorch module mapping global image/query features to one area logit.

    Raises:
        ValueError: If the feature or configured hidden dimension is invalid.
        RuntimeError: If PyTorch is unavailable.
    """
    hidden_dimension = int(settings.area_hidden_dimension)
    if feature_dimension < 1 or hidden_dimension < 8 or hidden_dimension % 8:
        raise ValueError("Query area predictor dimensions are invalid.")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for query area prediction.") from error
    return torch.nn.Sequential(
        torch.nn.Linear(feature_dimension * 3, hidden_dimension),
        torch.nn.LayerNorm(hidden_dimension),
        torch.nn.GELU(),
        torch.nn.Dropout(float(settings.area_dropout)),
        torch.nn.Linear(hidden_dimension, 1),
    )


def query_area_logits(
    predictor: Any,
    patch_features: Any,
    projected_text: Any,
) -> Any:
    """Predict one foreground-area logit from each complete language query.

    Args:
        predictor: Trainable query-area predictor module.
        patch_features: Tensor shaped ``BxPxD`` for the complete image.
        projected_text: Tensor shaped ``BxD`` for the complete query.

    Returns:
        Tensor shaped ``B`` containing foreground-area logits.

    Raises:
        ValueError: If batch or feature geometry is inconsistent.
        RuntimeError: If PyTorch is unavailable.
    """
    if (
        patch_features.ndim != 3
        or projected_text.ndim != 2
        or patch_features.shape[0] != projected_text.shape[0]
        or patch_features.shape[2] != projected_text.shape[1]
    ):
        raise ValueError("Query area feature geometry is invalid.")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for query area prediction.") from error
    image_features = torch.nn.functional.normalize(
        patch_features.float().mean(dim=1),
        dim=1,
    )
    query_features = torch.nn.functional.normalize(projected_text.float(), dim=1)
    fused = torch.cat(
        [image_features, query_features, image_features * query_features],
        dim=1,
    )
    return predictor(fused).squeeze(1)


def query_area_loss(
    predicted_logits: Any,
    target_patch_fractions: Any,
) -> Any:
    """Regress the soft foreground fraction on a stable logit scale.

    Args:
        predicted_logits: Tensor shaped ``B``.
        target_patch_fractions: Soft target fractions shaped ``BxP``.

    Returns:
        Scalar differentiable Smooth L1 area loss.

    Raises:
        ValueError: If predictions or targets are invalid.
        RuntimeError: If PyTorch is unavailable.
    """
    if (
        predicted_logits.ndim != 1
        or target_patch_fractions.ndim != 2
        or predicted_logits.shape[0] != target_patch_fractions.shape[0]
        or target_patch_fractions.shape[1] < 1
        or bool((target_patch_fractions < 0.0).any())
        or bool((target_patch_fractions > 1.0).any())
    ):
        raise ValueError("Query area predictions and targets are invalid.")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for query area loss.") from error
    patch_count = int(target_patch_fractions.shape[1])
    minimum_fraction = 0.5 / patch_count
    target_fractions = target_patch_fractions.mean(dim=1).clamp(
        min=minimum_fraction,
        max=1.0 - minimum_fraction,
    )
    target_logits = torch.logit(target_fractions)
    return torch.nn.functional.smooth_l1_loss(predicted_logits, target_logits)


def topk_patch_masks(
    probabilities: np.ndarray,
    area_fractions: np.ndarray,
) -> np.ndarray:
    """Select query-specific highest-scoring patches by predicted area.

    Args:
        probabilities: Finite query-to-patch probabilities shaped ``QxP``.
        area_fractions: Finite predicted foreground fractions shaped ``Q``.

    Returns:
        Boolean masks shaped ``QxP`` with at least one selected patch per query.

    Raises:
        ValueError: If geometry, values, or probability ranges are invalid.
    """
    scores = np.asarray(probabilities, dtype=np.float32)
    fractions = np.asarray(area_fractions, dtype=np.float32)
    if (
        scores.ndim != 2
        or scores.shape[1] < 1
        or fractions.shape != (scores.shape[0],)
        or not np.all(np.isfinite(scores))
        or not np.all(np.isfinite(fractions))
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
        or np.any(fractions < 0.0)
        or np.any(fractions > 1.0)
    ):
        raise ValueError("Top-k patch scores or area fractions are invalid.")
    patch_count = scores.shape[1]
    selected = np.zeros_like(scores, dtype=bool)
    selected_counts = np.clip(
        np.rint(fractions * patch_count).astype(np.int64),
        1,
        patch_count,
    )
    for query_index, selected_count in enumerate(selected_counts):
        order = np.argsort(-scores[query_index], kind="stable")
        selected[query_index, order[:selected_count]] = True
    return selected
