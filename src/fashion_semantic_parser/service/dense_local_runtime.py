"""Pinned model adapters for production dense local re-encoding."""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import yaml
from pydantic import BaseModel, Field

from fashion_semantic_parser.common.paths import resolve_project_path
from fashion_semantic_parser.service.dinov2_region_encoder import (
    DinoV2DenseFeatureMap,
    DinoV2RegionEncoder,
    letterbox_image,
)
from fashion_semantic_parser.service.subject_roi import PersonROIDetectorSettings


class DenseLocalReencodingSettings(BaseModel):
    """Validated deployment settings for the frozen coarse-to-fine path."""

    checkpoint_path: str = (
        "models/checkpoints/localization/"
        + "dinov2_multiscale_728_train1000_steps1500.pt"
    )
    dinov2_config_path: str = "configs/localization_dinov2_region_728.yaml"
    bge_m3_config_path: str = "configs/localization_bge_m3_text.yaml"
    crop_fraction: float = Field(default=0.30, gt=0.0, le=1.0)
    max_crops: int = Field(default=3, ge=1)
    min_mask_area: int = Field(default=1, ge=1)
    subject_roi_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    person_detector: PersonROIDetectorSettings = Field(
        default_factory=PersonROIDetectorSettings
    )


class _DenseQueryProjector(Protocol):
    """Project one complete query into the DINOv2 alignment space."""

    def project(self, query: str) -> np.ndarray:
        """Return one normalized query feature row."""
        ...


class _DenseImageEncoder(Protocol):
    """Encode one RGB image into a dense DINOv2 feature grid."""

    def encode_dense(self, image_rgb: np.ndarray) -> DinoV2DenseFeatureMap:
        """Return patch features and source geometry."""
        ...

    def encode_dense_batch(
        self,
        images_rgb: list[np.ndarray],
    ) -> tuple[DinoV2DenseFeatureMap, ...]:
        """Return ordered feature maps from one batched forward pass."""
        ...


class _DensePatchScorer(Protocol):
    """Score dense image features against a projected complete query."""

    threshold: float

    def score(
        self,
        patch_features: np.ndarray,
        projected_query: np.ndarray,
    ) -> np.ndarray:
        """Return one foreground probability grid."""
        ...


@dataclass(frozen=True)
class DenseLocalRuntimeBundle:
    """Injectable frozen dependencies shared across localization requests."""

    projector: _DenseQueryProjector
    image_encoder: _DenseImageEncoder
    scorer: _DensePatchScorer


class _ProductionQueryProjector:
    """BGE-M3 plus frozen learned projection for complete expressions."""

    def __init__(self, text_encoder: Any, projection: Any, device: str) -> None:
        self.text_encoder = text_encoder
        self.projection = projection
        self.device = device

    def project(self, query: str) -> np.ndarray:
        """Encode one unmodified query into the alignment space."""
        import torch  # type: ignore[import-not-found]

        embeddings = self.text_encoder.encode([query])
        with torch.inference_mode():
            projected = self.projection(
                torch.from_numpy(embeddings).to(device=self.device)
            )
            projected = torch.nn.functional.normalize(projected.float(), dim=1)
        return np.asarray(projected.cpu().numpy(), dtype=np.float32)


class _ProductionPatchScorer:
    """Apply the frozen multiscale decoder to one query and image grid."""

    def __init__(self, checkpoint: Any, device: str) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.threshold = float(checkpoint.dense_settings.probability_threshold)

    def score(
        self,
        patch_features: np.ndarray,
        projected_query: np.ndarray,
    ) -> np.ndarray:
        """Return the first and only complete-query probability grid."""
        from fashion_semantic_parser.service.dense_patch_inference import (
            predict_patch_outputs,
        )

        probabilities, _ = predict_patch_outputs(
            self.checkpoint,
            patch_features,
            projected_query,
            self.device,
        )
        if probabilities.shape[0] != 1:
            raise ValueError("Dense service requires exactly one projected query.")
        return np.asarray(probabilities[0], dtype=np.float32)


class _BatchedDinoV2Encoder:
    """Add local-crop batching without expanding the base encoder module."""

    def __init__(self, encoder: DinoV2RegionEncoder) -> None:
        self.encoder = encoder

    def encode_dense(self, image_rgb: np.ndarray) -> DinoV2DenseFeatureMap:
        """Delegate one full-image encoding to the validated base encoder."""
        return self.encoder.encode_dense(image_rgb)

    def encode_dense_batch(
        self,
        images_rgb: list[np.ndarray],
    ) -> tuple[DinoV2DenseFeatureMap, ...]:
        """Encode all fixed local crops in one DINOv2 forward pass."""
        if not images_rgb:
            raise ValueError("Dense DINOv2 batch requires at least one image.")
        self.encoder.load()
        torch = self.encoder._torch  # pylint: disable=protected-access
        model = self.encoder._model  # pylint: disable=protected-access
        if torch is None or model is None:
            raise RuntimeError("DINOv2 model did not initialize.")
        settings = self.encoder.settings
        prepared = [
            letterbox_image(image, output_size=settings.input_size)
            for image in images_rgb
        ]
        image_tensor = torch.cat(
            [
                self.encoder._normalized_image_tensor(
                    image
                )  # pylint: disable=protected-access
                for image, _ in prepared
            ],
            dim=0,
        )
        grid_size = settings.input_size // settings.patch_size
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if settings.precision == "fp16"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            output = model.forward_features(image_tensor)
            patch_tokens = output["x_norm_patchtokens"]
            expected_shape = (
                len(images_rgb),
                grid_size * grid_size,
                settings.feature_dimension,
            )
            if patch_tokens.shape != expected_shape:
                raise ValueError(
                    "Unexpected batched DINOv2 patch-token shape: "
                    + f"{tuple(patch_tokens.shape)}"
                )
            features = torch.nn.functional.normalize(patch_tokens.float(), dim=2)
        values = np.asarray(features.cpu().numpy(), dtype=np.float32)
        return tuple(
            DinoV2DenseFeatureMap(
                features=values[index].reshape(
                    grid_size,
                    grid_size,
                    settings.feature_dimension,
                ),
                geometry=geometry,
            )
            for index, (_, geometry) in enumerate(prepared)
        )


def load_dense_local_reencoding_settings(
    config_path: str | Path = "configs/localization_dense_local_reencoding.yaml",
) -> DenseLocalReencodingSettings:
    """Load one coarse-to-fine deployment configuration.

    Args:
        config_path: Project-relative YAML path.

    Returns:
        Validated frozen runtime settings.
    """
    path = resolve_project_path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cast(
        DenseLocalReencodingSettings,
        DenseLocalReencodingSettings.model_validate(raw),
    )


def build_dense_local_runtime(
    settings: DenseLocalReencodingSettings,
) -> DenseLocalRuntimeBundle:
    """Build the pinned model bundle on first service use.

    Args:
        settings: Validated paths and inference settings.

    Returns:
        Reusable BGE-M3, DINOv2, projection, and decoder adapters.

    Raises:
        RuntimeError: If the PyTorch runtime is unavailable.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for dense localization.") from error
    from fashion_semantic_parser.service.bge_m3_text_encoder import (
        BgeM3TextEncoder,
        load_bge_m3_text_settings,
    )
    from fashion_semantic_parser.service.dense_patch_alignment import (
        apply_finetuned_dinov2_checkpoint,
        load_dense_patch_alignment_checkpoint,
    )
    from fashion_semantic_parser.service.dinov2_region_encoder import (
        load_dinov2_region_settings,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = load_dense_patch_alignment_checkpoint(
        settings.checkpoint_path,
        device=device,
    )
    image_encoder = DinoV2RegionEncoder(
        load_dinov2_region_settings(settings.dinov2_config_path)
    )
    apply_finetuned_dinov2_checkpoint(image_encoder, checkpoint)
    return DenseLocalRuntimeBundle(
        projector=_ProductionQueryProjector(
            BgeM3TextEncoder(load_bge_m3_text_settings(settings.bge_m3_config_path)),
            checkpoint.projection,
            device,
        ),
        image_encoder=_BatchedDinoV2Encoder(image_encoder),
        scorer=_ProductionPatchScorer(checkpoint, device),
    )
