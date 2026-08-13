"""Frozen dense patch inference shared by evaluation and crop audits."""

import math
from typing import Any

import numpy as np

from fashion_semantic_parser.service.dense_patch_alignment import (
    DensePatchAlignmentCheckpoint,
)


def predict_patch_outputs(
    checkpoint: DensePatchAlignmentCheckpoint,
    patch_features: np.ndarray,
    projected_text: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return patch probabilities and optional predicted target areas.

    Args:
        checkpoint: Strictly restored dense localization checkpoint.
        patch_features: One image's finite ``HxWxD`` patch feature grid.
        projected_text: Complete-query features shaped ``QxD``.
        device: PyTorch inference device.

    Returns:
        Patch probabilities and optional query-area fractions.

    Raises:
        ValueError: If checkpoint components or model type are inconsistent.
        RuntimeError: If PyTorch is unavailable for a multiscale checkpoint.
    """
    if checkpoint.model_type == "cosine_calibration":
        from fashion_semantic_parser.service.dense_region_localization import (
            calibrated_dense_probabilities,
            dense_similarity_scores,
        )

        similarities = dense_similarity_scores(patch_features, projected_text)
        calibrated: np.ndarray = calibrated_dense_probabilities(
            similarities,
            logit_scale=checkpoint.logit_scale,
            logit_bias=checkpoint.logit_bias,
        )
        return calibrated, None
    if (
        checkpoint.model_type not in {"multiscale_decoder", "multiscale_area_decoder"}
        or checkpoint.decoder is None
    ):
        raise ValueError("Dense checkpoint model type and decoder are inconsistent.")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for multiscale evaluation.") from error
    from fashion_semantic_parser.service.dense_patch_decoder import (
        multiscale_patch_decoder_logits,
    )

    height, width, feature_dimension = patch_features.shape
    query_count = len(projected_text)
    flattened = np.asarray(
        patch_features.reshape(1, height * width, feature_dimension),
        dtype=np.float32,
    )
    patch_tensor = (
        torch.from_numpy(flattened)
        .to(device=device)
        .expand(
            query_count,
            -1,
            -1,
        )
    )
    text_tensor = torch.from_numpy(projected_text).to(device=device)
    calibration = (
        torch.tensor(
            math.log(checkpoint.logit_scale),
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            checkpoint.logit_bias,
            dtype=torch.float32,
            device=device,
        ),
        checkpoint.dense_settings.max_logit_scale,
    )
    with torch.inference_mode():
        logits = multiscale_patch_decoder_logits(
            checkpoint.decoder,
            patch_tensor,
            text_tensor,
            calibration,
        )
        probabilities = torch.sigmoid(logits).reshape(query_count, height, width)
        predicted_area_fractions = _predict_area_fractions(
            checkpoint,
            patch_tensor,
            text_tensor,
            torch,
        )
    return (
        np.asarray(probabilities.cpu().numpy(), dtype=np.float32),
        predicted_area_fractions,
    )


def _predict_area_fractions(
    checkpoint: DensePatchAlignmentCheckpoint,
    patch_tensor: Any,
    text_tensor: Any,
    torch: Any,
) -> np.ndarray | None:
    """Return schema-three query-area fractions under inference mode."""
    if checkpoint.model_type != "multiscale_area_decoder":
        if checkpoint.area_predictor is not None:
            raise ValueError("Non-area checkpoint unexpectedly has an area predictor.")
        return None
    if checkpoint.area_predictor is None:
        raise ValueError("Area checkpoint is missing its area predictor.")
    from fashion_semantic_parser.service.dense_patch_area import query_area_logits

    area_logits = query_area_logits(
        checkpoint.area_predictor,
        patch_tensor,
        text_tensor,
    )
    return np.asarray(
        torch.sigmoid(area_logits).cpu().numpy(),
        dtype=np.float32,
    )
