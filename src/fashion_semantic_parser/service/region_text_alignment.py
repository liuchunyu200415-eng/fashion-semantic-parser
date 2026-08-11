"""Train a lightweight projection between text and DINOv2 region features."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator

from fashion_semantic_parser.common.paths import resolve_project_path


class RegionTextAlignmentSettings(BaseModel):
    """Validated dimensions and optimization defaults for alignment training."""

    text_dimension: int = Field(default=1024, ge=1)
    region_dimension: int = Field(default=384, ge=1)
    hidden_dimension: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.07, gt=0.0, le=1.0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    training_steps: int = Field(default=20, ge=1)
    seed: int = Field(default=312, ge=0)

    @model_validator(mode="after")
    def validate_projection_geometry(self) -> "RegionTextAlignmentSettings":
        """Reject a hidden bottleneck smaller than the aligned feature space."""
        if self.hidden_dimension < self.region_dimension:
            raise ValueError(
                "Alignment hidden_dimension cannot be smaller than region_dimension."
            )
        return self


def load_region_text_alignment_settings(
    config_path: str | Path = "configs/localization_region_text_alignment.yaml",
) -> RegionTextAlignmentSettings:
    """Load the project-relative region-text alignment configuration."""
    path = resolve_project_path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cast(
        RegionTextAlignmentSettings,
        RegionTextAlignmentSettings.model_validate(raw),
    )


def build_text_projection(settings: RegionTextAlignmentSettings) -> Any:
    """Build the only trainable component in the frozen-encoder smoke path."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for region-text alignment.") from error
    return torch.nn.Sequential(
        torch.nn.Linear(settings.text_dimension, settings.hidden_dimension),
        torch.nn.GELU(),
        torch.nn.Linear(settings.hidden_dimension, settings.region_dimension),
    )


def build_positive_region_mask(
    query_annotation_ids: list[tuple[int, ...]],
    region_annotation_ids: list[int],
) -> np.ndarray:
    """Mark every source annotation referenced by each complete query as positive."""
    if not query_annotation_ids:
        raise ValueError("At least one query target set is required.")
    if not region_annotation_ids:
        raise ValueError("At least one candidate region is required.")
    if len(region_annotation_ids) != len(set(region_annotation_ids)):
        raise ValueError("Candidate region annotation IDs must be unique.")
    region_index = {
        annotation_id: index
        for index, annotation_id in enumerate(region_annotation_ids)
    }
    positive_mask = np.zeros(
        (len(query_annotation_ids), len(region_annotation_ids)),
        dtype=np.bool_,
    )
    for query_index, annotation_ids in enumerate(query_annotation_ids):
        if not annotation_ids:
            raise ValueError("Every alignment query requires at least one target.")
        for annotation_id in annotation_ids:
            if annotation_id not in region_index:
                raise ValueError(
                    f"Query references missing candidate annotation {annotation_id}."
                )
            positive_mask[query_index, region_index[annotation_id]] = True
    if not np.all(positive_mask.any(axis=0)):
        raise ValueError("Every candidate region must be positive for a query.")
    return positive_mask


def multi_positive_contrastive_loss(
    text_features: Any,
    region_features: Any,
    positive_mask: Any,
    *,
    temperature: float,
) -> tuple[Any, Any]:
    """Return symmetric multi-positive InfoNCE loss and cosine logits."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for region-text alignment.") from error
    if text_features.ndim != 2 or region_features.ndim != 2:
        raise ValueError("Text and region features must be rank-two tensors.")
    if text_features.shape[1] != region_features.shape[1]:
        raise ValueError("Text and region feature dimensions must match.")
    if tuple(positive_mask.shape) != (
        text_features.shape[0],
        region_features.shape[0],
    ):
        raise ValueError("Positive mask shape must be queries by candidate regions.")
    if not 0.0 < temperature <= 1.0:
        raise ValueError("temperature must be in the interval (0, 1].")
    positive_mask = positive_mask.to(device=text_features.device, dtype=torch.bool)
    if not torch.all(positive_mask.any(dim=1)):
        raise ValueError("Every query must have at least one positive region.")
    if not torch.all(positive_mask.any(dim=0)):
        raise ValueError("Every candidate region must be positive for a query.")
    text_features = torch.nn.functional.normalize(text_features.float(), dim=1)
    region_features = torch.nn.functional.normalize(region_features.float(), dim=1)
    logits = text_features @ region_features.transpose(0, 1)
    logits = logits / temperature
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, negative_infinity)
    text_to_region = -(
        torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)
    ).mean()
    region_to_text = -(
        torch.logsumexp(positive_logits, dim=0) - torch.logsumexp(logits, dim=0)
    ).mean()
    return 0.5 * (text_to_region + region_to_text), logits


def positive_top1_accuracy(logits: Any, positive_mask: Any) -> float:
    """Measure whether each query's highest-similarity region is a positive."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for alignment metrics.") from error
    if tuple(positive_mask.shape) != tuple(logits.shape):
        raise ValueError("Top-1 metric requires a positive mask matching logits.")
    top_indices = logits.argmax(dim=1)
    row_indices = torch.arange(logits.shape[0], device=logits.device)
    correct = positive_mask.to(device=logits.device, dtype=torch.bool)[
        row_indices, top_indices
    ]
    return float(correct.float().mean().item())
