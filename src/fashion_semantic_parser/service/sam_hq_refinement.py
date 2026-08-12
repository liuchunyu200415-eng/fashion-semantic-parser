"""Refine externally supplied image-coordinate boxes with official SAM-HQ."""

import math
import sys
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

from fashion_semantic_parser.common.exceptions import ModelNotReadyError
from fashion_semantic_parser.common.paths import resolve_project_path
from fashion_semantic_parser.service.grounded_sam_hq import (
    _load_sam_hq_module,
    _precision_context,
    _required_asset,
)
from fashion_semantic_parser.service.sam_hq_proposals import (
    SAMHQProposalSettings,
    validate_local_sam_hq_assets,
)


@dataclass(frozen=True)
class SAMHQBoxPromptResult:
    """One SAM-HQ Mask produced from an externally supplied Box prompt."""

    prompt_box: tuple[float, float, float, float]
    mask_box: tuple[float, float, float, float] | None
    mask: np.ndarray
    mask_quality: float


class SAMHQBoxPromptRefiner:
    """Lazy reusable official SAM-HQ refiner for externally supplied boxes."""

    def __init__(
        self,
        settings: SAMHQProposalSettings,
        *,
        predictor: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        """Create a refiner with optional injected test dependencies.

        Args:
            settings: Validated official SAM-HQ asset and runtime settings.
            predictor: Optional predictor test double.
            torch_module: Optional PyTorch-compatible test double.
        """
        self.settings = settings
        self._predictor = predictor
        self._torch = torch_module
        self._model_init_lock = Lock()
        self._inference_lock = Lock()
        self._prepare_external_path()

    def refine(
        self,
        image_rgb: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
    ) -> list[SAMHQBoxPromptResult]:
        """Refine valid image-coordinate Box prompts into binary Masks.

        Args:
            image_rgb: Input uint8 RGB image.
            boxes: Positive-area ``xyxy`` prompts in image coordinates.

        Returns:
            One result per prompt, preserving prompt order.

        Raises:
            ValueError: If the image or any prompt is invalid.
            ModelNotReadyError: If the official runtime returns invalid outputs.
        """
        candidate_groups = self.refine_candidates(
            image_rgb,
            boxes,
            multimask_output=False,
        )
        return [group[0] for group in candidate_groups]

    def refine_candidates(
        self,
        image_rgb: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
        *,
        multimask_output: bool,
    ) -> list[list[SAMHQBoxPromptResult]]:
        """Return every SAM-HQ candidate generated for each Box prompt.

        Args:
            image_rgb: Input uint8 RGB image.
            boxes: Positive-area ``xyxy`` prompts in image coordinates.
            multimask_output: Whether to request ambiguity-aware Mask candidates.

        Returns:
            Candidate groups preserving Box prompt order.

        Raises:
            ValueError: If the image or any prompt is invalid.
            ModelNotReadyError: If the official runtime returns invalid outputs.
        """
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                "SAM-HQ refinement input must be an HxWx3 uint8 RGB array."
            )
        normalized_boxes = _validate_prompt_boxes(
            boxes,
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
        )
        if not normalized_boxes:
            return []
        predictor = self._get_predictor()
        torch = self._torch
        if torch is None:
            raise ModelNotReadyError("SAM-HQ refinement requires PyTorch.")
        with self._inference_lock:
            with _precision_context(
                torch,
                device=self.settings.device,
                precision=self.settings.precision,
            ):
                predictor.set_image(image)
                input_boxes = torch.as_tensor(
                    normalized_boxes,
                    dtype=torch.float32,
                    device=predictor.device,
                )
                transformed_boxes = predictor.transform.apply_boxes_torch(
                    input_boxes,
                    image.shape[:2],
                )
                masks, mask_scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    mask_input=None,
                    multimask_output=multimask_output,
                    hq_token_only=self.settings.hq_token_only,
                )
        mask_values = _tensor_to_numpy(masks)
        score_values = _tensor_to_numpy(mask_scores)
        if (
            mask_values.ndim != 4
            or mask_values.shape[0] != len(normalized_boxes)
            or mask_values.shape[1] < 1
            or mask_values.shape[2:] != image.shape[:2]
        ):
            raise ModelNotReadyError(
                "SAM-HQ returned invalid prompted Mask dimensions."
            )
        if score_values.shape != mask_values.shape[:2]:
            raise ModelNotReadyError(
                "SAM-HQ returned invalid prompted score dimensions."
            )
        if not multimask_output and mask_values.shape[1] != 1:
            raise ModelNotReadyError(
                "SAM-HQ returned multiple Masks when multimask output was disabled."
            )
        results: list[list[SAMHQBoxPromptResult]] = []
        for box, prompt_masks, prompt_scores in zip(
            normalized_boxes,
            mask_values,
            score_values,
            strict=True,
        ):
            candidates = []
            for mask_value, score_value in zip(
                prompt_masks,
                prompt_scores,
                strict=True,
            ):
                mask = np.asarray(mask_value, dtype=bool)
                quality = float(score_value)
                if not math.isfinite(quality):
                    raise ModelNotReadyError(
                        "SAM-HQ prompted Mask score must be finite."
                    )
                candidates.append(
                    SAMHQBoxPromptResult(
                        prompt_box=box,
                        mask_box=_mask_box(mask),
                        mask=mask,
                        mask_quality=quality,
                    )
                )
            results.append(candidates)
        return results

    def synchronize(self) -> None:
        """Synchronize CUDA so refinement timing includes queued GPU work."""
        if (
            self._torch is not None
            and self.settings.device == "cuda"
            and self._torch.cuda.is_available()
        ):
            self._torch.cuda.synchronize()

    def _prepare_external_path(self) -> None:
        """Expose the official SAM-HQ checkout without global path changes."""
        if not self.settings.sam_hq_repo:
            return
        try:
            path = resolve_project_path(self.settings.sam_hq_repo)
        except ValueError:
            return
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    def _get_predictor(self) -> Any:
        """Load the pinned official SAM-HQ predictor once."""
        if self._predictor is not None:
            return self._predictor
        with self._model_init_lock:
            if self._predictor is not None:
                return self._predictor
            if not self.settings.sam_hq_repo:
                raise ModelNotReadyError(
                    "SAM-HQ Box refinement requires the pinned source checkout."
                )
            repo_path = resolve_project_path(self.settings.sam_hq_repo)
            weights_path = _required_asset(
                self.settings.sam_hq_weights,
                "SAM-HQ weights",
            )
            validate_local_sam_hq_assets(self.settings, repo_path, weights_path)
            module = _load_sam_hq_module(self.settings.sam_hq_module)
            try:
                import torch  # type: ignore[import-not-found]

                registry = getattr(module, "sam_model_registry")
                predictor_class = getattr(module, "SamPredictor")
                sam = registry[self.settings.sam_hq_model_type](
                    checkpoint=str(weights_path)
                )
                sam.to(device=self.settings.device)
                sam.eval()
                self._predictor = predictor_class(sam)
                self._torch = torch
            except (
                AttributeError,
                ImportError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
            ) as error:
                raise ModelNotReadyError(
                    "SAM-HQ Box prompt refiner could not be loaded."
                ) from error
        return self._predictor


def _validate_prompt_boxes(
    boxes: list[tuple[float, float, float, float]],
    *,
    image_width: int,
    image_height: int,
) -> list[tuple[float, float, float, float]]:
    """Validate and clamp prompt boxes to image coordinates."""
    normalized = []
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
            raise ValueError("SAM-HQ prompt boxes must contain four finite values.")
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        clamped = (
            max(0.0, min(float(image_width), x_min)),
            max(0.0, min(float(image_height), y_min)),
            max(0.0, min(float(image_width), x_max)),
            max(0.0, min(float(image_height), y_max)),
        )
        if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
            raise ValueError("SAM-HQ prompt boxes must have positive in-image area.")
        normalized.append(clamped)
    return normalized


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Move tensor-like outputs to CPU NumPy arrays."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _mask_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return one mask-derived ``xyxy`` box or ``None`` for an empty Mask."""
    y_values, x_values = np.nonzero(mask)
    if not len(x_values):
        return None
    return (
        float(x_values.min()),
        float(y_values.min()),
        float(x_values.max() + 1),
        float(y_values.max() + 1),
    )
