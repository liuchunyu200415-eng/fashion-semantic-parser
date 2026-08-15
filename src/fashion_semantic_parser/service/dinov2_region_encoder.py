"""Extract Mask-pooled local-region features with the official DINOv2 model."""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator

from fashion_semantic_parser.common.paths import resolve_project_path


class DinoV2RegionEncoderSettings(BaseModel):
    """Validated DINOv2 region-feature smoke configuration."""

    model_name: Literal["dinov2_vits14"] = "dinov2_vits14"
    torch_hub_repo: Literal["facebookresearch/dinov2"] = "facebookresearch/dinov2"
    repo_path: str = "external/dinov2"
    repo_commit: str = Field(
        default="7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        pattern=r"^[0-9a-f]{40}$",
    )
    weights_path: str = "models/checkpoints/localization/dinov2_vits14_pretrain.pth"
    weights_size_bytes: int = Field(default=88283115, ge=1)
    input_size: int = Field(default=518, ge=14)
    patch_size: Literal[14] = 14
    feature_dimension: Literal[384] = 384
    device: Literal["cuda", "cpu"] = "cuda"
    precision: Literal["fp16", "fp32"] = "fp16"

    @model_validator(mode="after")
    def validate_patch_grid(self) -> "DinoV2RegionEncoderSettings":
        """Require a square input that maps to an exact patch-token grid."""
        if self.input_size % self.patch_size != 0:
            raise ValueError("DINOv2 input_size must be divisible by patch_size.")
        if self.device == "cpu" and self.precision == "fp16":
            raise ValueError("DINOv2 fp16 smoke requires the CUDA device.")
        return self


@dataclass(frozen=True)
class DinoV2LetterboxGeometry:
    """Coordinate transform between a source image and square model input."""

    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    top: int
    left: int
    output_size: int


@dataclass(frozen=True)
class DinoV2DenseFeatureMap:
    """Unit-normalized DINOv2 patch features with source-image geometry."""

    features: np.ndarray
    geometry: DinoV2LetterboxGeometry


class DinoV2RegionEncoder:
    """Frozen official DINOv2 backbone with independent Mask pooling."""

    def __init__(self, settings: DinoV2RegionEncoderSettings) -> None:
        self.settings = settings
        self._torch: Any | None = None
        self._model: Any | None = None

    def load(self) -> None:
        """Load official pretrained weights through Meta's Torch Hub entrypoint."""
        if self._model is not None:
            return
        repo_path = resolve_project_path(self.settings.repo_path)
        weights_path = resolve_project_path(self.settings.weights_path)
        self._validate_local_assets(repo_path, weights_path)
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required for DINOv2 region encoding."
            ) from error
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DINOv2 CUDA smoke requested but CUDA is unavailable.")
        model = torch.hub.load(
            str(repo_path),
            self.settings.model_name,
            source="local",
            trust_repo=True,
            weights=str(weights_path),
        )
        self._model = model.eval().to(self.settings.device)
        self._torch = torch

    def _validate_local_assets(self, repo_path: Path, weights_path: Path) -> None:
        """Reject missing, drifting, or incomplete official local assets."""
        head_path = repo_path / ".git" / "HEAD"
        if not head_path.is_file():
            raise RuntimeError(
                "Official DINOv2 checkout is missing; run "
                "scripts/setup_dinov2_region_model.sh."
            )
        actual_commit = head_path.read_text(encoding="utf-8").strip()
        if actual_commit != self.settings.repo_commit:
            raise RuntimeError(
                "DINOv2 checkout is not at the pinned commit: "
                f"expected={self.settings.repo_commit} actual={actual_commit}"
            )
        if not weights_path.is_file():
            raise RuntimeError(
                "Official DINOv2 weights are missing; run "
                "scripts/setup_dinov2_region_model.sh."
            )
        actual_size = weights_path.stat().st_size
        if actual_size != self.settings.weights_size_bytes:
            raise RuntimeError(
                "DINOv2 weights have an unexpected size: "
                f"expected={self.settings.weights_size_bytes} actual={actual_size}"
            )

    def encode(self, image_rgb: np.ndarray, target_masks: np.ndarray) -> np.ndarray:
        """Return one unit-normalized DINOv2 feature per independent target Mask."""
        self.load()
        torch = self._torch
        model = self._model
        if torch is None or model is None:
            raise RuntimeError("DINOv2 model did not initialize.")
        image, masks = letterbox_image_and_masks(
            image_rgb,
            target_masks,
            output_size=self.settings.input_size,
        )
        image_tensor = self._normalized_image_tensor(image)
        occupancy_array = masks_to_patch_occupancy(
            masks,
            patch_size=self.settings.patch_size,
        )
        occupancy = torch.from_numpy(occupancy_array).to(
            device=self.settings.device,
            dtype=torch.bool,
        )
        grid_size = self.settings.input_size // self.settings.patch_size
        occupancy = occupancy.reshape(occupancy.shape[0], -1)
        if not torch.all(occupancy.any(dim=1)):
            raise ValueError("At least one target does not occupy a DINOv2 patch.")

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.settings.precision == "fp16"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            output = model.forward_features(image_tensor)
            patch_tokens = output["x_norm_patchtokens"][0]
            if patch_tokens.shape != (
                grid_size * grid_size,
                self.settings.feature_dimension,
            ):
                raise ValueError(
                    "Unexpected DINOv2 patch-token shape: "
                    + f"{tuple(patch_tokens.shape)}"
                )
            weights = occupancy.to(dtype=patch_tokens.dtype)
            features = weights @ patch_tokens
            features = features / weights.sum(dim=1, keepdim=True)
            features = torch.nn.functional.normalize(features.float(), dim=1)
        result: np.ndarray = np.asarray(features.cpu().numpy(), dtype=np.float32)
        return result

    def encode_dense(self, image_rgb: np.ndarray) -> DinoV2DenseFeatureMap:
        """Return a normalized patch-feature grid for one complete RGB image.

        Args:
            image_rgb: Source ``HxWx3`` uint8 RGB image.

        Returns:
            Dense patch features and the reversible letterbox geometry.

        Raises:
            RuntimeError: If the model runtime cannot be initialized.
            ValueError: If the image or returned patch-token shape is invalid.
        """
        self.load()
        torch = self._torch
        model = self._model
        if torch is None or model is None:
            raise RuntimeError("DINOv2 model did not initialize.")
        image, geometry = letterbox_image(
            image_rgb,
            output_size=self.settings.input_size,
        )
        image_tensor = self._normalized_image_tensor(image)
        grid_size = self.settings.input_size // self.settings.patch_size
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.settings.precision == "fp16"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            output = model.forward_features(image_tensor)
            patch_tokens = output["x_norm_patchtokens"][0]
            if patch_tokens.shape != (
                grid_size * grid_size,
                self.settings.feature_dimension,
            ):
                raise ValueError(
                    "Unexpected DINOv2 patch-token shape: "
                    f"{tuple(patch_tokens.shape)}"
                )
            features = torch.nn.functional.normalize(patch_tokens.float(), dim=1)
        values = np.asarray(features.cpu().numpy(), dtype=np.float32)
        return DinoV2DenseFeatureMap(
            features=values.reshape(
                grid_size,
                grid_size,
                self.settings.feature_dimension,
            ),
            geometry=geometry,
        )

    def configure_finetuning(self, unfreeze_last_blocks: int) -> list[tuple[str, Any]]:
        """Enable gradients only for the requested final DINOv2 blocks."""
        self.load()
        if self._model is None:
            raise RuntimeError("DINOv2 model did not initialize.")
        from fashion_semantic_parser.service.dense_patch_finetuning import (
            configure_dinov2_last_blocks,
        )

        trainable = configure_dinov2_last_blocks(
            self._model,
            unfreeze_last_blocks,
        )
        self._model.train()
        return trainable

    def set_finetuning_mode(self, training: bool) -> None:
        """Switch the loaded backbone between train and deterministic audit mode."""
        if self._model is None:
            raise RuntimeError("DINOv2 model did not initialize.")
        self._model.train(training)

    def encode_dense_trainable_batch(
        self,
        images_rgb: list[np.ndarray],
    ) -> tuple[Any, tuple[DinoV2LetterboxGeometry, ...]]:
        """Return differentiable ``BxPxD`` features for an image batch."""
        self.load()
        torch = self._torch
        model = self._model
        if torch is None or model is None:
            raise RuntimeError("DINOv2 model did not initialize.")
        if not images_rgb:
            raise ValueError("Trainable DINOv2 batching requires at least one image.")
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("DINOv2 fine-tuning has no trainable parameters.")
        tensors: list[Any] = []
        geometries: list[DinoV2LetterboxGeometry] = []
        for image_rgb in images_rgb:
            image, geometry = letterbox_image(
                image_rgb,
                output_size=self.settings.input_size,
            )
            tensors.append(self._normalized_image_tensor(image))
            geometries.append(geometry)
        image_tensor = torch.cat(tensors, dim=0)
        grid_size = self.settings.input_size // self.settings.patch_size
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.settings.precision == "fp16"
            else nullcontext()
        )
        with autocast_context:
            output = model.forward_features(image_tensor)
            patch_tokens = output["x_norm_patchtokens"]
            if patch_tokens.shape != (
                len(images_rgb),
                grid_size * grid_size,
                self.settings.feature_dimension,
            ):
                raise ValueError(
                    "Unexpected trainable DINOv2 patch-token shape: "
                    f"{tuple(patch_tokens.shape)}"
                )
            features = torch.nn.functional.normalize(patch_tokens.float(), dim=2)
        return features, tuple(geometries)

    def trainable_state_dict(self) -> dict[str, Any]:
        """Return only the explicitly unfrozen DINOv2 parameter tensors."""
        if self._model is None:
            raise RuntimeError("DINOv2 model did not initialize.")
        from fashion_semantic_parser.service.dense_patch_finetuning import (
            trainable_dinov2_state_dict,
        )

        return trainable_dinov2_state_dict(self._model)

    def load_finetuned_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        unfreeze_last_blocks: int,
    ) -> None:
        """Restore the exact recorded DINOv2 fine-tuned parameter subset."""
        self.load()
        if self._model is None:
            raise RuntimeError("DINOv2 model did not initialize.")
        from fashion_semantic_parser.service.dense_patch_finetuning import (
            load_trainable_dinov2_state_dict,
        )

        load_trainable_dinov2_state_dict(
            self._model,
            state_dict,
            unfreeze_last_blocks=unfreeze_last_blocks,
        )

    def _normalized_image_tensor(self, image: np.ndarray) -> Any:
        """Convert one square uint8 image into normalized model input."""
        torch = self._torch
        if torch is None:
            raise RuntimeError("DINOv2 PyTorch runtime did not initialize.")
        image_tensor = torch.from_numpy(image).to(
            device=self.settings.device,
            dtype=torch.float32,
        )
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=self.settings.device,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=self.settings.device,
        ).view(1, 3, 1, 1)
        return (image_tensor - mean) / std

    def synchronize(self) -> None:
        """Synchronize CUDA so smoke latency excludes queued GPU work."""
        if (
            self._torch is not None
            and self.settings.device == "cuda"
            and self._torch.cuda.is_available()
        ):
            self._torch.cuda.synchronize()


def load_dinov2_region_settings(
    config_path: str | Path = "configs/localization_dinov2_region.yaml",
) -> DinoV2RegionEncoderSettings:
    """Load one project-relative DINOv2 region configuration."""
    path = resolve_project_path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cast(
        DinoV2RegionEncoderSettings,
        DinoV2RegionEncoderSettings.model_validate(raw),
    )


def letterbox_image_and_masks(
    image_rgb: np.ndarray,
    target_masks: np.ndarray,
    *,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize image and Masks together while preserving their aspect ratio."""
    image = np.asarray(image_rgb)
    masks = np.asarray(target_masks)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("DINOv2 input image must be an HxWx3 uint8 RGB array.")
    if masks.ndim != 3 or masks.shape[1:] != image.shape[:2] or not len(masks):
        raise ValueError("Target Masks must be a non-empty NxHxW array.")
    binary_masks = masks != 0
    if not np.all(binary_masks.any(axis=(1, 2))):
        raise ValueError("Every source target Mask must contain at least one pixel.")
    if output_size < 1:
        raise ValueError("output_size must be positive.")
    geometry = letterbox_geometry(
        (int(image.shape[0]), int(image.shape[1])),
        output_size=output_size,
    )
    resized_height = geometry.resized_height
    resized_width = geometry.resized_width
    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_masks = np.stack(
        [
            _resize_binary_mask_preserving_target(
                mask,
                output_height=resized_height,
                output_width=resized_width,
            )
            for mask in binary_masks
        ],
        axis=0,
    )
    top = geometry.top
    left = geometry.left
    canvas = np.empty((output_size, output_size, 3), dtype=np.uint8)
    canvas[:, :] = [124, 116, 104]
    canvas[top : top + resized_height, left : left + resized_width] = resized_image
    mask_canvas = np.zeros(
        (len(masks), output_size, output_size),
        dtype=np.uint8,
    )
    mask_canvas[
        :,
        top : top + resized_height,
        left : left + resized_width,
    ] = resized_masks
    if not np.all(mask_canvas.any(axis=(1, 2))):
        raise ValueError("Letterbox resize removed at least one target Mask.")
    return canvas, mask_canvas


def letterbox_image(
    image_rgb: np.ndarray,
    *,
    output_size: int,
) -> tuple[np.ndarray, DinoV2LetterboxGeometry]:
    """Letterbox one RGB image and return its reversible geometry.

    Args:
        image_rgb: Source ``HxWx3`` uint8 RGB image.
        output_size: Positive square model-input side length.

    Returns:
        Letterboxed image and its reversible coordinate transform.

    Raises:
        ValueError: If image type, shape, or output size is invalid.
    """
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("DINOv2 input image must be an HxWx3 uint8 RGB array.")
    geometry = letterbox_geometry(
        (int(image.shape[0]), int(image.shape[1])),
        output_size=output_size,
    )
    resized = cv2.resize(
        image,
        (geometry.resized_width, geometry.resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.empty((output_size, output_size, 3), dtype=np.uint8)
    canvas[:, :] = [124, 116, 104]
    canvas[
        geometry.top : geometry.top + geometry.resized_height,
        geometry.left : geometry.left + geometry.resized_width,
    ] = resized
    return canvas, geometry


def letterbox_geometry(
    image_shape: tuple[int, int],
    *,
    output_size: int,
) -> DinoV2LetterboxGeometry:
    """Return the deterministic aspect-preserving square-input transform.

    Args:
        image_shape: Positive source ``(height, width)`` pair.
        output_size: Positive square model-input side length.

    Returns:
        Integer resize and padding geometry.

    Raises:
        ValueError: If any source or output dimension is not positive.
    """
    height, width = image_shape
    if height < 1 or width < 1 or output_size < 1:
        raise ValueError("Letterbox dimensions must be positive.")
    scale = output_size / max(height, width)
    resized_height = min(output_size, max(1, round(height * scale)))
    resized_width = min(output_size, max(1, round(width * scale)))
    return DinoV2LetterboxGeometry(
        original_height=height,
        original_width=width,
        resized_height=resized_height,
        resized_width=resized_width,
        top=(output_size - resized_height) // 2,
        left=(output_size - resized_width) // 2,
        output_size=output_size,
    )


def patch_scores_to_image(
    patch_scores: np.ndarray,
    geometry: DinoV2LetterboxGeometry,
) -> np.ndarray:
    """Upsample one patch-score grid into original-image coordinates.

    Args:
        patch_scores: Non-empty finite two-dimensional patch score grid.
        geometry: Letterbox transform produced for the source image.

    Returns:
        Float32 similarity map with the source image height and width.

    Raises:
        ValueError: If the patch-score grid is empty, non-finite, or not 2D.
    """
    scores = np.asarray(patch_scores, dtype=np.float32)
    if scores.ndim != 2 or not scores.size or not np.all(np.isfinite(scores)):
        raise ValueError("Patch scores must be one non-empty finite 2D grid.")
    square = cv2.resize(
        scores,
        (geometry.output_size, geometry.output_size),
        interpolation=cv2.INTER_LINEAR,
    )
    content = square[
        geometry.top : geometry.top + geometry.resized_height,
        geometry.left : geometry.left + geometry.resized_width,
    ]
    result = cv2.resize(
        content,
        (geometry.original_width, geometry.original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.asarray(result, dtype=np.float32)


def _resize_binary_mask_preserving_target(
    mask: np.ndarray,
    *,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    """Resize one valid Mask without deleting a sub-pixel target entirely."""
    binary_mask = np.asarray(mask, dtype=np.uint8)
    resized = cv2.resize(
        binary_mask,
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    )
    if resized.any():
        return resized

    source_y, source_x = np.nonzero(binary_mask)
    source_height, source_width = binary_mask.shape
    center_y = float(source_y.mean())
    center_x = float(source_x.mean())
    target_y = round((center_y + 0.5) * output_height / source_height - 0.5)
    target_x = round((center_x + 0.5) * output_width / source_width - 0.5)
    target_y = min(output_height - 1, max(0, target_y))
    target_x = min(output_width - 1, max(0, target_x))
    resized[target_y, target_x] = 1
    return resized


def masks_to_patch_occupancy(
    masks: np.ndarray,
    *,
    patch_size: int,
) -> np.ndarray:
    """Mark every DINOv2 patch touched by each target, including tiny parts."""
    binary_masks = np.asarray(masks) != 0
    if binary_masks.ndim != 3 or not len(binary_masks):
        raise ValueError("Patch occupancy requires a non-empty NxHxW Mask array.")
    if patch_size < 1:
        raise ValueError("patch_size must be positive.")
    _, height, width = binary_masks.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Mask dimensions must be divisible by patch_size.")
    grid_height = height // patch_size
    grid_width = width // patch_size
    occupancy: np.ndarray = (
        binary_masks.reshape(
            len(binary_masks),
            grid_height,
            patch_size,
            grid_width,
            patch_size,
        )
        .any(axis=4)
        .any(axis=2)
    )
    return occupancy
