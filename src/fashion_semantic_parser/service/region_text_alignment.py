"""Train and evaluate projection between text and DINOv2 region features."""

from collections import defaultdict
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


def load_text_projection_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str,
) -> tuple[Any, RegionTextAlignmentSettings]:
    """Load one project-owned projection checkpoint with strict metadata checks."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for region-text alignment.") from error
    path = resolve_project_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Alignment checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported region-text alignment checkpoint schema.")
    if payload.get("base_encoders_frozen") is not True:
        raise ValueError("Alignment checkpoint must record frozen base encoders.")
    if payload.get("dinov2_model") != "dinov2_vits14":
        raise ValueError("Alignment checkpoint has an unexpected DINOv2 model.")
    if payload.get("text_model") != "BAAI/bge-m3":
        raise ValueError("Alignment checkpoint has an unexpected text model.")
    settings = RegionTextAlignmentSettings.model_validate(
        payload.get("alignment_settings")
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Alignment checkpoint state_dict is missing or invalid.")
    projection = build_text_projection(settings).to(device)
    projection.load_state_dict(state_dict, strict=True)
    return projection.eval(), settings


def extract_unique_region_features(
    items: list[Any],
    encoder: Any,
) -> dict[int, np.ndarray]:
    """Encode each unique source Mask once, grouped by source image."""
    if not items:
        raise ValueError("Region feature extraction requires at least one item.")
    grouped: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"image_rgb": None, "masks": {}}
    )
    for item in items:
        image_id = item.sample.source_image_id
        group = grouped[image_id]
        if group["image_rgb"] is None:
            group["image_rgb"] = item.image_rgb
        elif not np.array_equal(group["image_rgb"], item.image_rgb):
            raise ValueError(f"Image {image_id} decoded inconsistently.")
        for annotation_id, mask in zip(
            item.source_annotation_ids,
            item.target_masks,
        ):
            existing = group["masks"].get(annotation_id)
            if existing is not None and not np.array_equal(existing, mask):
                raise ValueError(f"Annotation {annotation_id} decoded inconsistently.")
            group["masks"][annotation_id] = mask

    features_by_id: dict[int, np.ndarray] = {}
    for group in grouped.values():
        annotation_ids = sorted(group["masks"])
        masks = np.stack([group["masks"][value] for value in annotation_ids])
        features = encoder.encode(group["image_rgb"], masks)
        if len(features) != len(annotation_ids):
            raise ValueError("Region encoder returned an unexpected row count.")
        for annotation_id, feature in zip(annotation_ids, features):
            features_by_id[annotation_id] = np.asarray(feature, dtype=np.float32)
    return features_by_id


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


def same_image_contrastive_loss(
    text_features: Any,
    region_features: Any,
    positive_mask: Any,
    *,
    query_image_ids: list[int],
    region_image_ids: list[int],
    temperature: float,
) -> tuple[Any, int, int]:
    """Average multi-positive loss only over competitive same-image pools."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for region-text alignment.") from error
    if len(query_image_ids) != text_features.shape[0]:
        raise ValueError("query_image_ids must match text feature rows.")
    if len(region_image_ids) != region_features.shape[0]:
        raise ValueError("region_image_ids must match region feature rows.")
    if tuple(positive_mask.shape) != (
        text_features.shape[0],
        region_features.shape[0],
    ):
        raise ValueError("Positive mask shape must be queries by candidate regions.")
    if set(query_image_ids) != set(region_image_ids):
        raise ValueError("Query and candidate region image sets must match.")

    weighted_losses: list[Any] = []
    weights: list[int] = []
    negative_pair_count = 0
    competitive_image_count = 0
    for image_id in sorted(set(query_image_ids)):
        query_indices = [
            index for index, value in enumerate(query_image_ids) if value == image_id
        ]
        region_indices = [
            index for index, value in enumerate(region_image_ids) if value == image_id
        ]
        query_index_tensor = torch.as_tensor(
            query_indices,
            device=text_features.device,
            dtype=torch.long,
        )
        region_index_tensor = torch.as_tensor(
            region_indices,
            device=region_features.device,
            dtype=torch.long,
        )
        local_text = text_features.index_select(0, query_index_tensor)
        local_regions = region_features.index_select(0, region_index_tensor)
        local_positive = positive_mask.index_select(0, query_index_tensor).index_select(
            1, region_index_tensor
        )
        local_negative_count = int((~local_positive.bool()).sum().item())
        if local_negative_count == 0:
            continue
        local_loss, _ = multi_positive_contrastive_loss(
            local_text,
            local_regions,
            local_positive,
            temperature=temperature,
        )
        weighted_losses.append(local_loss * len(query_indices))
        weights.append(len(query_indices))
        negative_pair_count += local_negative_count
        competitive_image_count += 1
    if not weighted_losses:
        raise ValueError(
            "Same-image alignment requires at least one image with a negative pair."
        )
    loss = torch.stack(weighted_losses).sum() / sum(weights)
    return loss, competitive_image_count, negative_pair_count


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


def evaluate_image_candidate_retrieval(
    *,
    query_ids: list[str],
    projected_text_features: np.ndarray,
    query_image_ids: list[int],
    query_target_ids: list[tuple[int, ...]],
    query_dimensions: list[tuple[str, ...]],
    query_languages: list[str],
    region_annotation_ids: list[int],
    region_image_ids: list[int],
    region_features: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate query-to-region ranking within each selected source image."""
    text = _normalize_numpy_features(projected_text_features, name="text")
    regions = _normalize_numpy_features(region_features, name="region")
    query_count = len(query_ids)
    query_fields = (
        query_image_ids,
        query_target_ids,
        query_dimensions,
        query_languages,
    )
    if any(len(values) != query_count for values in query_fields):
        raise ValueError("Query retrieval metadata lengths must match query_ids.")
    if text.shape[0] != query_count:
        raise ValueError("Text feature rows must match query_ids.")
    if text.shape[1] != regions.shape[1]:
        raise ValueError("Projected text and region dimensions must match.")
    if len(region_annotation_ids) != regions.shape[0] or len(region_image_ids) != len(
        region_annotation_ids
    ):
        raise ValueError("Region metadata lengths must match region feature rows.")
    if len(region_annotation_ids) != len(set(region_annotation_ids)):
        raise ValueError("Region annotation IDs must be unique during evaluation.")

    candidates_by_image: dict[int, list[int]] = defaultdict(list)
    for region_index, image_id in enumerate(region_image_ids):
        candidates_by_image[image_id].append(region_index)
    cases: list[dict[str, Any]] = []
    for query_index, query_id in enumerate(query_ids):
        candidate_indices = candidates_by_image.get(query_image_ids[query_index], [])
        if not candidate_indices:
            raise ValueError(f"Query {query_id} has no same-image candidate regions.")
        target_ids = set(query_target_ids[query_index])
        if not target_ids:
            raise ValueError(f"Query {query_id} has no target annotations.")
        candidate_ids = [region_annotation_ids[index] for index in candidate_indices]
        if not target_ids.issubset(candidate_ids):
            missing = sorted(target_ids.difference(candidate_ids))
            raise ValueError(
                f"Query {query_id} is missing target candidates: {missing}"
            )
        scores = regions[candidate_indices] @ text[query_index]
        ranked_offsets = np.argsort(-scores, kind="stable")
        ranked_ids = [candidate_ids[offset] for offset in ranked_offsets]
        top1_correct = ranked_ids[0] in target_ids
        target_count = len(target_ids)
        exact_set = set(ranked_ids[:target_count]) == target_ids
        first_positive_rank = next(
            rank
            for rank, annotation_id in enumerate(ranked_ids, start=1)
            if annotation_id in target_ids
        )
        candidate_count = len(candidate_ids)
        cases.append(
            {
                "query_id": query_id,
                "source_image_id": query_image_ids[query_index],
                "dimensions": list(query_dimensions[query_index]),
                "language": query_languages[query_index],
                "target_annotation_ids": sorted(target_ids),
                "candidate_count": candidate_count,
                "negative_candidate_count": candidate_count - target_count,
                "competitive": candidate_count > target_count,
                "top1_annotation_id": ranked_ids[0],
                "top1_correct": top1_correct,
                "exact_set_at_target_count": exact_set,
                "reciprocal_rank": 1.0 / first_positive_rank,
            }
        )

    dimensions = sorted(
        {dimension for case in cases for dimension in case["dimensions"]}
    )
    languages = sorted({str(case["language"]) for case in cases})
    summary = _aggregate_retrieval_cases(cases)
    summary["by_dimension"] = {
        dimension: _aggregate_retrieval_cases(
            [case for case in cases if dimension in case["dimensions"]]
        )
        for dimension in dimensions
    }
    summary["by_language"] = {
        language: _aggregate_retrieval_cases(
            [case for case in cases if case["language"] == language]
        )
        for language in languages
    }
    return summary, cases


def _normalize_numpy_features(features: np.ndarray, *, name: str) -> np.ndarray:
    """Validate and normalize one finite rank-two retrieval feature matrix."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError(f"{name} features must be a non-empty rank-two array.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} features must contain only finite values.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} features cannot contain a zero vector.")
    result: np.ndarray = values / norms
    return result


def _aggregate_retrieval_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Return numerator-aware all-query and competitive-only ranking metrics."""
    if not cases:
        return {
            "query_count": 0,
            "top1_correct_count": 0,
            "top1_accuracy": None,
            "exact_set_correct_count": 0,
            "exact_set_at_target_count_rate": None,
            "mean_reciprocal_rank": None,
            "competitive_query_count": 0,
            "competitive_top1_correct_count": 0,
            "competitive_top1_accuracy": None,
            "competitive_exact_set_correct_count": 0,
            "competitive_exact_set_at_target_count_rate": None,
        }
    competitive = [case for case in cases if case["competitive"]]
    top1_count = sum(bool(case["top1_correct"]) for case in cases)
    exact_set_count = sum(bool(case["exact_set_at_target_count"]) for case in cases)
    competitive_top1_count = sum(bool(case["top1_correct"]) for case in competitive)
    competitive_exact_set_count = sum(
        bool(case["exact_set_at_target_count"]) for case in competitive
    )
    return {
        "query_count": len(cases),
        "top1_correct_count": top1_count,
        "top1_accuracy": top1_count / len(cases),
        "exact_set_correct_count": exact_set_count,
        "exact_set_at_target_count_rate": exact_set_count / len(cases),
        "mean_reciprocal_rank": sum(float(case["reciprocal_rank"]) for case in cases)
        / len(cases),
        "competitive_query_count": len(competitive),
        "competitive_top1_correct_count": competitive_top1_count,
        "competitive_top1_accuracy": (
            competitive_top1_count / len(competitive) if competitive else None
        ),
        "competitive_exact_set_correct_count": competitive_exact_set_count,
        "competitive_exact_set_at_target_count_rate": (
            competitive_exact_set_count / len(competitive) if competitive else None
        ),
    }
