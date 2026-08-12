"""Train calibrated full-image patch similarity with frozen PRD encoders."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field

from fashion_semantic_parser.common.paths import resolve_project_path
from fashion_semantic_parser.service.dinov2_region_encoder import (
    DinoV2LetterboxGeometry,
)
from fashion_semantic_parser.service.region_text_alignment import (
    RegionTextAlignmentSettings,
    build_text_projection,
)


class DensePatchAlignmentSettings(BaseModel):
    """Validated optimization and probability calibration settings."""

    learning_rate: float = Field(default=2e-4, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    training_steps: int = Field(default=300, ge=1)
    batch_size: int = Field(default=32, ge=1)
    seed: int = Field(default=312, ge=0)
    initial_logit_scale: float = Field(default=1.0 / 0.07, gt=0.0)
    max_logit_scale: float = Field(default=100.0, gt=0.0)
    initial_logit_bias: float = 0.0
    probability_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)


@dataclass(frozen=True)
class DensePatchTrainingCache:
    """Frozen unique-image features and query-specific patch targets."""

    image_ids: tuple[int, ...]
    image_features: np.ndarray
    query_image_indices: np.ndarray
    target_patch_fractions: np.ndarray


@dataclass(frozen=True)
class DensePatchAlignmentCheckpoint:
    """Loaded projection and calibrated similarity parameters."""

    projection: Any
    alignment_settings: RegionTextAlignmentSettings
    dense_settings: DensePatchAlignmentSettings
    logit_scale: float
    logit_bias: float


def load_dense_patch_alignment_settings(
    config_path: str | Path = "configs/localization_dense_patch_alignment.yaml",
) -> DensePatchAlignmentSettings:
    """Load the project-relative dense patch alignment configuration.

    Args:
        config_path: Project-relative or absolute YAML configuration path.

    Returns:
        Validated dense patch alignment settings.
    """
    path = resolve_project_path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cast(
        DensePatchAlignmentSettings,
        DensePatchAlignmentSettings.model_validate(raw),
    )


def mask_to_patch_fractions(
    target_mask: np.ndarray,
    geometry: DinoV2LetterboxGeometry,
    *,
    patch_size: int,
) -> np.ndarray:
    """Map one source Mask to soft foreground fractions per DINOv2 patch.

    Args:
        target_mask: Non-empty source-image binary Mask.
        geometry: DINOv2 letterbox transform for the same source image.
        patch_size: Positive patch side length dividing the model input.

    Returns:
        Float32 ``grid_height x grid_width`` foreground fractions.

    Raises:
        ValueError: If Mask geometry, content, or patch size is invalid.
    """
    mask = np.asarray(target_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape != (
        geometry.original_height,
        geometry.original_width,
    ):
        raise ValueError("Target Mask does not match DINOv2 source geometry.")
    if not mask.any():
        raise ValueError("Dense patch supervision requires a non-empty Mask.")
    if patch_size < 1 or geometry.output_size % patch_size:
        raise ValueError("Patch size must divide the DINOv2 input size.")
    resized = cv2.resize(
        np.asarray(mask, dtype=np.float32),
        (geometry.resized_width, geometry.resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros(
        (geometry.output_size, geometry.output_size),
        dtype=np.float32,
    )
    canvas[
        geometry.top : geometry.top + geometry.resized_height,
        geometry.left : geometry.left + geometry.resized_width,
    ] = resized
    if not canvas.any():
        source_y, source_x = np.nonzero(mask)
        mapped_y = geometry.top + round(
            (float(source_y.mean()) + 0.5)
            * geometry.resized_height
            / geometry.original_height
            - 0.5
        )
        mapped_x = geometry.left + round(
            (float(source_x.mean()) + 0.5)
            * geometry.resized_width
            / geometry.original_width
            - 0.5
        )
        canvas[
            min(geometry.output_size - 1, max(0, mapped_y)),
            min(geometry.output_size - 1, max(0, mapped_x)),
        ] = 1.0
    grid_size = geometry.output_size // patch_size
    fractions = canvas.reshape((grid_size, patch_size, grid_size, patch_size)).mean(
        axis=(1, 3)
    )
    return np.asarray(fractions, dtype=np.float32)


def build_dense_patch_training_cache(
    items: list[Any],
    encoder: Any,
) -> DensePatchTrainingCache:
    """Extract each image once and align every query with patch supervision.

    Args:
        items: Referring items ordered exactly like their text embeddings.
        encoder: Loaded-on-demand DINOv2 encoder exposing ``encode_dense``.

    Returns:
        Compact unique-image features and query target fractions.

    Raises:
        ValueError: If items, image identity, or extracted features are invalid.
    """
    if not items:
        raise ValueError("Dense patch training requires at least one query.")
    groups: dict[int, list[int]] = defaultdict(list)
    for item_index, item in enumerate(items):
        groups[item.sample.source_image_id].append(item_index)
    image_features: list[np.ndarray] = []
    image_ids: list[int] = []
    query_image_indices = np.empty(len(items), dtype=np.int64)
    target_fractions: list[np.ndarray | None] = [None] * len(items)
    for image_index, (image_id, item_indices) in enumerate(groups.items()):
        first_item = items[item_indices[0]]
        dense = encoder.encode_dense(first_item.image_rgb)
        features = np.asarray(dense.features, dtype=np.float32)
        if features.ndim != 3 or not np.all(np.isfinite(features)):
            raise ValueError("DINOv2 returned invalid dense patch features.")
        image_ids.append(image_id)
        image_features.append(np.asarray(features.reshape(-1, features.shape[2])))
        for item_index in item_indices:
            item = items[item_index]
            if not np.array_equal(item.image_rgb, first_item.image_rgb):
                raise ValueError(f"Image {image_id} decoded inconsistently.")
            union_mask = np.asarray(item.target_masks.any(axis=0), dtype=bool)
            fractions = mask_to_patch_fractions(
                union_mask,
                dense.geometry,
                patch_size=encoder.settings.patch_size,
            )
            if fractions.shape != features.shape[:2]:
                raise ValueError("Patch target and DINOv2 feature grids differ.")
            query_image_indices[item_index] = image_index
            target_fractions[item_index] = fractions.reshape(-1)
    if any(value is None for value in target_fractions):
        raise ValueError("At least one dense query target was not constructed.")
    return DensePatchTrainingCache(
        image_ids=tuple(image_ids),
        image_features=np.asarray(np.stack(image_features), dtype=np.float16),
        query_image_indices=query_image_indices,
        target_patch_fractions=np.asarray(
            np.stack([cast(np.ndarray, value) for value in target_fractions]),
            dtype=np.float32,
        ),
    )


def dense_patch_logits(
    patch_features: Any,
    projected_text: Any,
    log_scale: Any,
    logit_bias: Any,
    *,
    max_logit_scale: float,
) -> Any:
    """Compute calibrated query-to-patch cosine logits in PyTorch.

    Args:
        patch_features: Tensor shaped ``BxPxD``.
        projected_text: Tensor shaped ``BxD``.
        log_scale: Learnable logarithm of the positive cosine scale.
        logit_bias: Learnable scalar foreground bias.
        max_logit_scale: Positive upper bound for numerical stability.

    Returns:
        Tensor shaped ``BxP`` containing calibrated foreground logits.

    Raises:
        ValueError: If tensor ranks, batch sizes, or dimensions are inconsistent.
    """
    if (
        patch_features.ndim != 3
        or projected_text.ndim != 2
        or patch_features.shape[0] != projected_text.shape[0]
        or patch_features.shape[2] != projected_text.shape[1]
    ):
        raise ValueError("Dense patch and projected text tensors are incompatible.")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for dense patch logits.") from error
    patches = torch.nn.functional.normalize(patch_features.float(), dim=2)
    queries = torch.nn.functional.normalize(projected_text.float(), dim=1)
    scale = torch.clamp(log_scale.exp(), max=max_logit_scale)
    return scale * torch.einsum("bpd,bd->bp", patches, queries) + logit_bias


def balanced_patch_mask_loss(logits: Any, target_fractions: Any) -> Any:
    """Return foreground-balanced BCE plus soft Dice patch supervision.

    Args:
        logits: Calibrated foreground logits shaped ``BxP``.
        target_fractions: Soft target fractions in ``[0, 1]`` with the same shape.

    Returns:
        Scalar differentiable training loss.

    Raises:
        ValueError: If target geometry, range, or foreground/background is invalid.
    """
    if logits.ndim != 2 or logits.shape != target_fractions.shape:
        raise ValueError("Dense patch logits and targets must share a BxP shape.")
    if bool((target_fractions < 0.0).any()) or bool((target_fractions > 1.0).any()):
        raise ValueError("Dense patch targets must remain in [0, 1].")
    positive_mass = target_fractions.sum(dim=1)
    negative_mass = (1.0 - target_fractions).sum(dim=1)
    if bool((positive_mass <= 0.0).any()) or bool((negative_mass <= 0.0).any()):
        raise ValueError("Every dense query requires foreground and background.")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for dense patch loss.") from error
    positive_bce = (torch.nn.functional.softplus(-logits) * target_fractions).sum(
        dim=1
    ) / positive_mass
    negative_bce = (
        torch.nn.functional.softplus(logits) * (1.0 - target_fractions)
    ).sum(dim=1) / negative_mass
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target_fractions).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + 1e-6) / (
        probabilities.sum(dim=1) + positive_mass + 1e-6
    )
    return (0.5 * (positive_bce + negative_bce) + dice).mean()


def load_dense_patch_alignment_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str,
) -> DensePatchAlignmentCheckpoint:
    """Load a strictly validated dense patch alignment checkpoint.

    Args:
        checkpoint_path: Project-relative or absolute checkpoint path.
        device: PyTorch destination device.

    Returns:
        Frozen text projection plus calibrated dense similarity parameters.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If checkpoint schema or PRD encoder metadata is invalid.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for dense checkpoint loading."
        ) from error
    path = resolve_project_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dense patch checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported dense patch checkpoint schema.")
    if payload.get("base_encoders_frozen") is not True:
        raise ValueError("Dense patch checkpoint must record frozen base encoders.")
    if payload.get("dinov2_model") != "dinov2_vits14":
        raise ValueError("Dense patch checkpoint has an unexpected DINOv2 model.")
    if payload.get("text_model") != "BAAI/bge-m3":
        raise ValueError("Dense patch checkpoint has an unexpected text model.")
    alignment_settings = RegionTextAlignmentSettings.model_validate(
        payload.get("alignment_settings")
    )
    dense_settings = DensePatchAlignmentSettings.model_validate(
        payload.get("dense_settings")
    )
    state_dict = payload.get("projection_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Dense patch projection state is missing or invalid.")
    logit_scale = payload.get("logit_scale")
    logit_bias = payload.get("logit_bias")
    if (
        not isinstance(logit_scale, (int, float))
        or not 0.0 < float(logit_scale) <= dense_settings.max_logit_scale
        or not isinstance(logit_bias, (int, float))
        or not np.isfinite(float(logit_bias))
    ):
        raise ValueError("Dense patch calibration values are invalid.")
    projection = build_text_projection(alignment_settings).to(device)
    projection.load_state_dict(state_dict, strict=True)
    return DensePatchAlignmentCheckpoint(
        projection=projection.eval(),
        alignment_settings=alignment_settings,
        dense_settings=dense_settings,
        logit_scale=float(logit_scale),
        logit_bias=float(logit_bias),
    )
