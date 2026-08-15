"""Shared runtime operations for supervised dense patch training."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from fashion_semantic_parser.service.dense_patch_alignment import (
    balanced_patch_mask_loss,
    dense_patch_logits,
)
from fashion_semantic_parser.service.dense_patch_area import (
    query_area_logits,
    query_area_loss,
    topk_patch_masks,
)
from fashion_semantic_parser.service.dense_patch_decoder import (
    multiscale_patch_decoder_logits,
)


@dataclass
class DenseTrainingRuntime:
    """Tensors and models shared by dense training and audit evaluation."""

    projection: Any
    decoder: Any | None
    area_predictor: Any | None
    model_type: str
    log_scale: Any
    logit_bias: Any
    text_tensor: Any
    cache: Any
    settings: Any
    device: str


def training_batch(
    runtime: DenseTrainingRuntime,
    batch_indices: np.ndarray,
) -> tuple[Any, Any, Any]:
    """Move one sampled query batch and its source-image patches to the device.

    Args:
        runtime: Dense training runtime.
        batch_indices: One-dimensional selected query indices.

    Returns:
        Patch tensor, target tensor, and device query indices.

    Raises:
        RuntimeError: If PyTorch is unavailable.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for dense patch batching.") from error
    query_indices = torch.from_numpy(batch_indices).to(
        device=runtime.device,
        dtype=torch.long,
    )
    image_indices = runtime.cache.query_image_indices[batch_indices]
    patch_tensor = torch.from_numpy(runtime.cache.image_features[image_indices]).to(
        device=runtime.device
    )
    target_tensor = torch.from_numpy(
        runtime.cache.target_patch_fractions[batch_indices]
    ).to(device=runtime.device)
    return patch_tensor, target_tensor, query_indices


def runtime_predictions(
    runtime: DenseTrainingRuntime,
    patch_tensor: Any,
    projected_text: Any,
) -> tuple[Any, Any | None]:
    """Return patch logits and optional query-area logits.

    Args:
        runtime: Dense training runtime.
        patch_tensor: Tensor shaped ``BxPxD``.
        projected_text: Tensor shaped ``BxD``.

    Returns:
        Patch logits and optional area logits.

    Raises:
        RuntimeError: If the configured runtime is internally inconsistent.
    """
    calibration = (
        runtime.log_scale,
        runtime.logit_bias,
        runtime.settings.max_logit_scale,
    )
    if runtime.model_type in {"multiscale_decoder", "multiscale_area_decoder"}:
        if runtime.decoder is None:
            raise RuntimeError("Multiscale runtime is missing its decoder.")
        logits = multiscale_patch_decoder_logits(
            runtime.decoder,
            patch_tensor,
            projected_text,
            calibration,
        )
        if runtime.model_type == "multiscale_area_decoder":
            if runtime.area_predictor is None:
                raise RuntimeError("Area runtime is missing its area predictor.")
            return logits, query_area_logits(
                runtime.area_predictor,
                patch_tensor,
                projected_text,
            )
        if runtime.area_predictor is not None:
            raise RuntimeError("Multiscale runtime unexpectedly has an area predictor.")
        return logits, None
    if (
        runtime.model_type != "cosine_calibration"
        or runtime.decoder is not None
        or runtime.area_predictor is not None
    ):
        raise RuntimeError("Dense training runtime model type is inconsistent.")
    return (
        dense_patch_logits(
            patch_tensor,
            projected_text,
            runtime.log_scale,
            runtime.logit_bias,
            max_logit_scale=runtime.settings.max_logit_scale,
        ),
        None,
    )


def runtime_loss(
    runtime: DenseTrainingRuntime,
    patch_tensor: Any,
    projected_text: Any,
    target_tensor: Any,
    sample_weights: Any | None = None,
) -> Any:
    """Return the patch loss plus optional query-area loss.

    Args:
        runtime: Dense training runtime.
        patch_tensor: Tensor shaped ``BxPxD``.
        projected_text: Tensor shaped ``BxD``.
        target_tensor: Soft patch targets shaped ``BxP``.
        sample_weights: Optional positive per-query weights shaped ``B``.

    Returns:
        Scalar differentiable combined loss.
    """
    logits, area_logits = runtime_predictions(runtime, patch_tensor, projected_text)
    loss = balanced_patch_mask_loss(logits, target_tensor, sample_weights)
    if area_logits is not None:
        loss = loss + runtime.settings.area_loss_weight * query_area_loss(
            area_logits,
            target_tensor,
        )
    return loss


def training_cache_outputs(
    runtime: DenseTrainingRuntime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Collect probabilities, targets, and optional area fractions.

    Args:
        runtime: Dense training runtime.

    Returns:
        Patch probabilities, soft targets, and optional predicted area fractions.

    Raises:
        RuntimeError: If PyTorch is unavailable.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for dense cache scoring.") from error
    runtime.projection.eval()
    if runtime.decoder is not None:
        runtime.decoder.eval()
    if runtime.area_predictor is not None:
        runtime.area_predictor.eval()
    probability_batches: list[np.ndarray] = []
    area_batches: list[np.ndarray] = []
    query_count = len(runtime.cache.query_image_indices)
    with torch.inference_mode():
        for start in range(0, query_count, runtime.settings.batch_size):
            stop = min(query_count, start + runtime.settings.batch_size)
            batch_indices = np.arange(start, stop, dtype=np.int64)
            patch_tensor, _, query_indices = training_batch(runtime, batch_indices)
            projected = runtime.projection(runtime.text_tensor[query_indices])
            logits, area_logits = runtime_predictions(
                runtime,
                patch_tensor,
                projected,
            )
            probability_batches.append(
                np.asarray(torch.sigmoid(logits).cpu().numpy(), dtype=np.float32)
            )
            if area_logits is not None:
                area_batches.append(
                    np.asarray(
                        torch.sigmoid(area_logits).cpu().numpy(),
                        dtype=np.float32,
                    )
                )
    area_fractions = np.concatenate(area_batches) if area_batches else None
    return (
        np.concatenate(probability_batches, axis=0),
        np.asarray(runtime.cache.target_patch_fractions, dtype=np.float32),
        area_fractions,
    )


def evaluate_training_cache(
    runtime: DenseTrainingRuntime,
) -> dict[str, float | int]:
    """Measure combined loss and selected patch IoU over all training queries.

    Args:
        runtime: Dense training runtime.

    Returns:
        Query count, mean loss, Recall50 count/rate, and mean patch IoU.

    Raises:
        RuntimeError: If PyTorch is unavailable.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for dense patch evaluation.") from error
    losses: list[float] = []
    query_count = len(runtime.cache.query_image_indices)
    runtime.projection.eval()
    if runtime.decoder is not None:
        runtime.decoder.eval()
    if runtime.area_predictor is not None:
        runtime.area_predictor.eval()
    with torch.inference_mode():
        for start in range(0, query_count, runtime.settings.batch_size):
            stop = min(query_count, start + runtime.settings.batch_size)
            batch_indices = np.arange(start, stop, dtype=np.int64)
            patch_tensor, target_tensor, query_indices = training_batch(
                runtime,
                batch_indices,
            )
            projected = runtime.projection(runtime.text_tensor[query_indices])
            loss = runtime_loss(runtime, patch_tensor, projected, target_tensor)
            losses.extend([float(loss.item())] * len(batch_indices))
    probabilities, targets, area_fractions = training_cache_outputs(runtime)
    if area_fractions is None:
        selected = probabilities >= runtime.settings.probability_threshold
    else:
        selected = topk_patch_masks(probabilities, area_fractions)
    target_masks = targets > 0.0
    intersections = np.logical_and(selected, target_masks).sum(axis=1)
    unions = np.logical_or(selected, target_masks).sum(axis=1)
    ious = intersections / np.maximum(unions, 1)
    return {
        "query_count": query_count,
        "mean_loss": float(np.mean(losses)),
        "patch_recall50_count": int(np.sum(ious >= 0.50)),
        "patch_recall50": float(np.mean(ious >= 0.50)),
        "mean_patch_iou": float(np.mean(ious)),
    }
