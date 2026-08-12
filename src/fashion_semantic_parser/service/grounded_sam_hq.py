"""Grounding DINO box proposals refined by SAM-HQ masks."""

import importlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, ContextManager, Literal, Mapping, Protocol, cast

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field

from fashion_semantic_parser.common.exceptions import (
    ConfigurationError,
    ModelNotReadyError,
)
from fashion_semantic_parser.common.paths import resolve_project_path


class GroundedSAMHQSettings(BaseModel):
    """Deployment settings for the first executable PRD 3.1.2 baseline."""

    grounding_dino_repo: str = "external/GroundingDINO"
    grounding_dino_config: str = (
        "external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    grounding_dino_weights: str = (
        "models/checkpoints/localization/groundingdino_swint_ogc.pth"
    )
    sam_hq_repo: str | None = "external/sam-hq"
    sam_hq_weights: str = "models/checkpoints/localization/sam_hq_vit_b.pth"
    sam_hq_model_type: Literal["vit_b", "vit_l", "vit_h", "vit_tiny"] = "vit_b"
    sam_hq_module: Literal["auto", "segment_anything_hq", "segment_anything"] = "auto"
    device: str = "cuda"
    precision: Literal["fp32", "fp16"] = "fp16"
    box_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    max_regions: int = Field(default=5, ge=1)
    min_mask_area: int = Field(default=16, ge=1)
    hq_token_only: bool = False
    subject_roi_margin: float = Field(default=0.35, ge=0.0, le=1.0)


@dataclass(frozen=True)
class GroundedMaskCandidate:
    """One text-grounded candidate with a SAM-HQ-refined binary mask."""

    box: tuple[float, float, float, float]
    confidence: float
    phrase: str
    mask: np.ndarray
    mask_quality: float


class GroundedMaskPredictor(Protocol):
    """Small injectable contract used by the localization service."""

    def predict(
        self,
        image_bgr: np.ndarray,
        prompt: str,
    ) -> list[GroundedMaskCandidate]:
        """Return text-grounded masks in coordinates of the supplied image."""
        ...


class GroundedSAMHQPredictor:
    """Lazy, reusable Grounding DINO and SAM-HQ inference bundle."""

    def __init__(self, settings: GroundedSAMHQSettings) -> None:
        self.settings = settings
        self._grounding_model: Any | None = None
        self._sam_predictor: Any | None = None
        self._model_init_lock = Lock()
        self._inference_lock = Lock()
        self._prepare_external_paths()

    def predict(
        self,
        image_bgr: np.ndarray,
        prompt: str,
    ) -> list[GroundedMaskCandidate]:
        """Detect prompt-aligned boxes and refine them into high-quality masks."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Localization expects a three-channel BGR image.")
        if not prompt.strip():
            raise ValueError("Grounding prompt cannot be empty.")

        torch = _load_torch_module()
        grounding_model = self._get_grounding_model()
        sam_predictor = self._get_sam_predictor()
        with self._inference_lock:
            with _precision_context(
                torch,
                device=self.settings.device,
                precision=self.settings.precision,
            ):
                detections, phrases = grounding_model.predict_with_caption(
                    image=image_bgr,
                    caption=prompt,
                    box_threshold=self.settings.box_threshold,
                    text_threshold=self.settings.text_threshold,
                )

            boxes, confidences, phrases = _rank_grounding_results(
                detections,
                phrases,
                image_width=image_bgr.shape[1],
                image_height=image_bgr.shape[0],
                limit=self.settings.max_regions,
            )
            if not boxes:
                return []

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            with _precision_context(
                torch,
                device=self.settings.device,
                precision=self.settings.precision,
            ):
                sam_predictor.set_image(image_rgb)
                input_boxes = torch.as_tensor(
                    boxes,
                    dtype=torch.float32,
                    device=sam_predictor.device,
                )
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                    input_boxes,
                    image_rgb.shape[:2],
                )
                masks, mask_scores, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    mask_input=None,
                    multimask_output=False,
                    hq_token_only=self.settings.hq_token_only,
                )

        mask_values = masks[:, 0].detach().cpu().numpy().astype(bool)
        score_values = mask_scores[:, 0].detach().cpu().numpy()
        return [
            GroundedMaskCandidate(
                box=(
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ),
                confidence=float(confidence),
                phrase=phrase,
                mask=mask,
                mask_quality=float(mask_quality),
            )
            for box, confidence, phrase, mask, mask_quality in zip(
                boxes,
                confidences,
                phrases,
                mask_values,
                score_values,
                strict=False,
            )
        ]

    def _prepare_external_paths(self) -> None:
        """Make official source checkouts importable without global PYTHONPATH."""
        repositories = [self.settings.grounding_dino_repo]
        if self.settings.sam_hq_repo:
            repositories.append(self.settings.sam_hq_repo)
        for repository in repositories:
            try:
                path = resolve_project_path(repository)
            except ValueError:
                continue
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

    def _get_grounding_model(self) -> Any:
        """Load the official Grounding DINO model once."""
        if self._grounding_model is not None:
            return self._grounding_model
        with self._model_init_lock:
            if self._grounding_model is None:
                config_path = _required_asset(
                    self.settings.grounding_dino_config,
                    "Grounding DINO config",
                )
                weights_path = _required_asset(
                    self.settings.grounding_dino_weights,
                    "Grounding DINO weights",
                )
                try:
                    inference = importlib.import_module("groundingdino.util.inference")
                    model_class = getattr(inference, "Model")
                    self._grounding_model = model_class(
                        model_config_path=str(config_path),
                        model_checkpoint_path=str(weights_path),
                        device=self.settings.device,
                    )
                except (
                    ImportError,
                    AttributeError,
                    OSError,
                    RuntimeError,
                    TypeError,
                ) as error:
                    raise ModelNotReadyError(
                        "Grounding DINO could not be loaded. Install the official "
                        "repository and rebuild its CUDA extension for this GPU."
                    ) from error
        return self._grounding_model

    def _get_sam_predictor(self) -> Any:
        """Load the official SAM-HQ predictor once."""
        if self._sam_predictor is not None:
            return self._sam_predictor
        with self._model_init_lock:
            if self._sam_predictor is None:
                weights_path = _required_asset(
                    self.settings.sam_hq_weights,
                    "SAM-HQ weights",
                )
                module = _load_sam_hq_module(self.settings.sam_hq_module)
                try:
                    registry = getattr(module, "sam_model_registry")
                    predictor_class = getattr(module, "SamPredictor")
                    sam = registry[self.settings.sam_hq_model_type](
                        checkpoint=str(weights_path)
                    )
                    sam.to(device=self.settings.device)
                    sam.eval()
                    self._sam_predictor = predictor_class(sam)
                except (
                    AttributeError,
                    KeyError,
                    OSError,
                    RuntimeError,
                    TypeError,
                ) as error:
                    raise ModelNotReadyError(
                        "SAM-HQ could not be loaded. Check the selected model type, "
                        "module, checkpoint, and CUDA device."
                    ) from error
        return self._sam_predictor


def load_grounded_sam_hq_settings(
    config_path: str = "configs/localization_grounded_sam_hq.yaml",
    *,
    overrides: Mapping[str, Any] | None = None,
) -> GroundedSAMHQSettings:
    """Load and validate one project-relative localization YAML."""
    try:
        resolved_path = resolve_project_path(config_path)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    if not resolved_path.is_file():
        raise ConfigurationError(f"Localization config file not found: {config_path}")
    try:
        raw_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid localization YAML: {config_path}") from error
    if not isinstance(raw_config, dict):
        raise ConfigurationError(
            f"Expected a mapping in localization config: {config_path}"
        )
    raw_config.update(dict(overrides or {}))
    try:
        return GroundedSAMHQSettings.model_validate(raw_config)
    except ValueError as error:
        raise ConfigurationError(
            f"Invalid localization settings in {config_path}: {error}"
        ) from error


def validate_grounded_sam_hq_assets(settings: GroundedSAMHQSettings) -> None:
    """Fail early with a concise message when model assets are absent."""
    missing = []
    for label, relative_path in (
        ("Grounding DINO repository", settings.grounding_dino_repo),
        ("Grounding DINO config", settings.grounding_dino_config),
        ("Grounding DINO weights", settings.grounding_dino_weights),
        ("SAM-HQ weights", settings.sam_hq_weights),
    ):
        try:
            exists = resolve_project_path(relative_path).exists()
        except ValueError:
            exists = False
        if not exists:
            missing.append(f"{label}: {relative_path}")
    if missing:
        raise ModelNotReadyError(
            "Grounding DINO + SAM-HQ localization assets are missing: "
            + "; ".join(missing)
        )


def _rank_grounding_results(
    detections: Any,
    phrases: list[str],
    *,
    image_width: int,
    image_height: int,
    limit: int,
) -> tuple[list[list[float]], list[float], list[str]]:
    """Clamp and rank official Grounding DINO detections by confidence."""
    raw_boxes = np.asarray(getattr(detections, "xyxy", []), dtype=float).reshape(-1, 4)
    raw_confidences = np.asarray(
        getattr(detections, "confidence", []),
        dtype=float,
    ).reshape(-1)
    ranked = []
    for box, confidence, phrase in zip(
        raw_boxes,
        raw_confidences,
        phrases,
        strict=False,
    ):
        x_min = max(0.0, min(float(image_width), float(box[0])))
        y_min = max(0.0, min(float(image_height), float(box[1])))
        x_max = max(0.0, min(float(image_width), float(box[2])))
        y_max = max(0.0, min(float(image_height), float(box[3])))
        if x_max <= x_min or y_max <= y_min:
            continue
        ranked.append(
            (
                float(confidence),
                [x_min, y_min, x_max, y_max],
                str(phrase).strip(),
            )
        )
    ranked.sort(key=lambda row: row[0], reverse=True)
    selected = ranked[:limit]
    return (
        [row[1] for row in selected],
        [row[0] for row in selected],
        [row[2] for row in selected],
    )


def _load_sam_hq_module(module_name: str) -> Any:
    """Import the official pip package or source-checkout module."""
    candidates = (
        ("segment_anything_hq", "segment_anything")
        if module_name == "auto"
        else (module_name,)
    )
    errors = []
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except (ImportError, OSError) as error:
            errors.append(f"{candidate}: {error}")
    raise ModelNotReadyError(
        "SAM-HQ is not importable. Install segment-anything-hq or clone the "
        "official sam-hq repository. Attempts: " + "; ".join(errors)
    )


def _required_asset(relative_path: str, label: str) -> Path:
    """Resolve one required project asset."""
    try:
        path = resolve_project_path(relative_path)
    except ValueError as error:
        raise ModelNotReadyError(f"{label} path is invalid: {relative_path}") from error
    if not path.exists():
        raise ModelNotReadyError(f"{label} not found: {relative_path}")
    return Path(path)


def _load_torch_module() -> Any:
    """Import torch only in the model-dependent path."""
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ModelNotReadyError(
            "PyTorch is required for Grounding DINO + SAM-HQ inference."
        ) from error


def _precision_context(
    torch: Any,
    *,
    device: str,
    precision: str,
) -> ContextManager[Any]:
    """Use CUDA autocast for the configured FP16 inference path."""
    if precision == "fp32":
        return nullcontext()
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ModelNotReadyError("FP16 localization inference requires CUDA.")
    return cast(
        ContextManager[Any],
        torch.autocast(device_type="cuda", dtype=torch.float16),
    )
