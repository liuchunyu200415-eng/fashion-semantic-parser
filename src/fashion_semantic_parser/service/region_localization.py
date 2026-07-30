"""Runtime service for PRD 3.1.2 language-guided region localization."""

import math
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping, Protocol

import cv2
import numpy as np

from fashion_semantic_parser.common.exceptions import (
    ConfigurationError,
    InvalidImageInputError,
    ModelNotReadyError,
)
from fashion_semantic_parser.common.paths import resolve_project_path
from fashion_semantic_parser.dao.localization.taxonomy import (
    LocalizationPrompt,
    resolve_localization_prompt,
)
from fashion_semantic_parser.models.localization import (
    LocalizationBoundingBox,
    LocalizedRegion,
    RegionLocalizationPrediction,
)
from fashion_semantic_parser.models.segmentation import (
    SegmentationSubjectROI,
    SubjectROISource,
)
from fashion_semantic_parser.service.grounded_sam_hq import (
    GroundedMaskCandidate,
    GroundedMaskPredictor,
    GroundedSAMHQPredictor,
    GroundedSAMHQSettings,
    load_grounded_sam_hq_settings,
    validate_grounded_sam_hq_assets,
)
from fashion_semantic_parser.service.subject_roi import (
    Detectron2PersonROIDetector,
    PersonROIDetectorSettings,
)


class RegionLocalizationRuntime(Protocol):
    """Minimal interface implemented by language-guided localization runtimes."""

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Localize the image region described by natural language."""
        ...


class SubjectROIDetector(Protocol):
    """Person ROI dependency used before local-part grounding."""

    def detect(self, image_path: Path) -> SegmentationSubjectROI | None:
        """Return the primary subject ROI or None."""
        ...


PredictorFactory = Callable[[GroundedSAMHQSettings], GroundedMaskPredictor]
SubjectROIDetectorFactory = Callable[
    [GroundedSAMHQSettings],
    SubjectROIDetector,
]


class GroundedSAMHQRegionLocalizationService:
    """Localize a text-described fashion part with person ROI and Grounded SAM."""

    def __init__(
        self,
        config_path: str = "configs/localization_grounded_sam_hq.yaml",
        *,
        predictor: GroundedMaskPredictor | None = None,
        predictor_factory: PredictorFactory = GroundedSAMHQPredictor,
        subject_roi_detector: SubjectROIDetector | None = None,
        subject_roi_detector_factory: SubjectROIDetectorFactory | None = None,
        settings: GroundedSAMHQSettings | None = None,
        settings_overrides: Mapping[str, Any] | None = None,
        grounding_prompt_override: str | None = None,
    ) -> None:
        """Create a service that loads both foundation models on first use."""
        if grounding_prompt_override is not None:
            grounding_prompt_override = " ".join(
                grounding_prompt_override.strip().split()
            )
            if not grounding_prompt_override:
                raise ValueError("Grounding prompt override cannot be empty.")
        self.config_path = config_path
        self._predictor = predictor
        self._predictor_factory = predictor_factory
        self._predictor_lock = Lock()
        self._subject_roi_detector = subject_roi_detector
        self._subject_roi_detector_factory = (
            subject_roi_detector_factory or _build_default_subject_roi_detector
        )
        self._subject_roi_detector_lock = Lock()
        self._settings = settings
        self._settings_lock = Lock()
        self._settings_overrides = dict(settings_overrides or {})
        self._grounding_prompt_override = grounding_prompt_override

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Return query-aligned masks and mask-derived boxes for one image."""
        if subject_roi is not None and auto_subject_roi:
            raise InvalidImageInputError(
                "subject_roi and auto_subject_roi cannot be used together"
            )
        resolved_image_path = self._resolve_image_path(image_path)
        image = cv2.imread(str(resolved_image_path))
        if image is None:
            raise InvalidImageInputError(f"Unable to read image: {image_path}")

        settings = self._get_settings()
        if (
            self._predictor is None
            and self._predictor_factory is GroundedSAMHQPredictor
        ):
            validate_grounded_sam_hq_assets(settings)

        effective_roi = subject_roi
        roi_source: SubjectROISource | None = (
            "manual" if subject_roi is not None else None
        )
        try:
            if auto_subject_roi:
                effective_roi = self._get_subject_roi_detector().detect(
                    resolved_image_path
                )
                roi_source = (
                    "detected" if effective_roi is not None else "full_image_fallback"
                )
            crop, coordinate_offset = _crop_to_subject_roi(
                image,
                effective_roi,
                margin=settings.subject_roi_margin,
            )
            prompt = resolve_localization_prompt(query)
            if self._grounding_prompt_override is not None:
                prompt = prompt.model_copy(
                    update={
                        "grounding_prompt": self._grounding_prompt_override,
                    }
                )
            candidates = self._get_predictor().predict(
                crop,
                prompt.grounding_prompt,
            )
        except (ConfigurationError, ModelNotReadyError, InvalidImageInputError):
            raise
        except OSError as error:
            raise ModelNotReadyError(
                f"Localization runtime asset could not be loaded: {error}"
            ) from error
        except ValueError as error:
            raise InvalidImageInputError(str(error)) from error
        except (AttributeError, RuntimeError, TypeError) as error:
            raise ModelNotReadyError(
                f"Grounding DINO + SAM-HQ inference failed: {error}"
            ) from error

        regions = [
            region
            for candidate in candidates
            if (
                region := _candidate_to_localized_region(
                    candidate,
                    prompt=prompt,
                    coordinate_offset=coordinate_offset,
                    expected_shape=(int(crop.shape[0]), int(crop.shape[1])),
                    min_mask_area=settings.min_mask_area,
                )
            )
            is not None
        ]
        return RegionLocalizationPrediction(
            image_path=image_path,
            query=query,
            regions=regions,
            subject_roi=effective_roi,
            subject_roi_source=roi_source,
        )

    def _get_settings(self) -> GroundedSAMHQSettings:
        """Load the deployment YAML once."""
        if self._settings is not None:
            return self._settings
        with self._settings_lock:
            if self._settings is None:
                self._settings = load_grounded_sam_hq_settings(
                    self.config_path,
                    overrides=self._settings_overrides,
                )
        return self._settings

    def _get_predictor(self) -> GroundedMaskPredictor:
        """Initialize and reuse the Grounding DINO + SAM-HQ bundle."""
        if self._predictor is not None:
            return self._predictor
        with self._predictor_lock:
            if self._predictor is None:
                self._predictor = self._predictor_factory(self._get_settings())
        return self._predictor

    def _get_subject_roi_detector(self) -> SubjectROIDetector:
        """Initialize and reuse the accepted primary-person detector."""
        if self._subject_roi_detector is not None:
            return self._subject_roi_detector
        with self._subject_roi_detector_lock:
            if self._subject_roi_detector is None:
                self._subject_roi_detector = self._subject_roi_detector_factory(
                    self._get_settings()
                )
        return self._subject_roi_detector

    @staticmethod
    def _resolve_image_path(image_path: str) -> Path:
        """Validate that the API image path stays inside the project."""
        try:
            resolved_path = resolve_project_path(image_path)
        except ValueError as error:
            raise InvalidImageInputError(str(error)) from error
        if not resolved_path.is_file():
            raise InvalidImageInputError(f"Input image not found: {image_path}")
        return resolved_path


class UnavailableRegionLocalizationService:
    """Explicit fallback for deployments that disable localization."""

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Reject inference instead of returning an invented localization."""
        raise ModelNotReadyError(
            "PRD 3.1.2 language-guided localization is not configured. "
            "Install the official Grounding DINO + SAM-HQ runtime and weights "
            "before calling /v1/localize."
        )


def _build_default_subject_roi_detector(
    settings: GroundedSAMHQSettings,
) -> SubjectROIDetector:
    """Build the same COCO-person ROI detector used by PRD 3.1.1."""
    return Detectron2PersonROIDetector(
        PersonROIDetectorSettings(
            device=settings.device,
            precision=settings.precision,
        )
    )


def _crop_to_subject_roi(
    image: np.ndarray,
    subject_roi: SegmentationSubjectROI | None,
    *,
    margin: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Crop an expanded subject box and retain its image-coordinate offset."""
    if subject_roi is None:
        return image, (0.0, 0.0)

    image_height, image_width = image.shape[:2]
    roi_width = subject_roi.x_max - subject_roi.x_min
    roi_height = subject_roi.y_max - subject_roi.y_min
    x_margin = roi_width * margin
    y_margin = roi_height * margin
    x_min = max(0, int(math.floor(subject_roi.x_min - x_margin)))
    y_min = max(0, int(math.floor(subject_roi.y_min - y_margin)))
    x_max = min(image_width, int(math.ceil(subject_roi.x_max + x_margin)))
    y_max = min(image_height, int(math.ceil(subject_roi.y_max + y_margin)))
    if x_max <= x_min or y_max <= y_min:
        raise InvalidImageInputError("Subject ROI does not overlap the input image.")
    return image[y_min:y_max, x_min:x_max], (float(x_min), float(y_min))


def _candidate_to_localized_region(
    candidate: GroundedMaskCandidate,
    *,
    prompt: LocalizationPrompt,
    coordinate_offset: tuple[float, float],
    expected_shape: tuple[int, int],
    min_mask_area: int,
) -> LocalizedRegion | None:
    """Convert one dense local mask to API polygons and a mask-derived box."""
    mask = np.asarray(candidate.mask, dtype=bool)
    if mask.shape != expected_shape:
        raise ModelNotReadyError(
            "SAM-HQ returned a mask whose dimensions do not match its input image."
        )
    if int(mask.sum()) < min_mask_area:
        return None

    y_values, x_values = np.nonzero(mask)
    x_offset, y_offset = coordinate_offset
    box = LocalizationBoundingBox(
        x_min=float(x_values.min()) + x_offset,
        y_min=float(y_values.min()) + y_offset,
        x_max=float(x_values.max() + 1) + x_offset,
        y_max=float(y_values.max() + 1) + y_offset,
    )
    polygons = _mask_to_polygons(
        mask,
        coordinate_offset=coordinate_offset,
    )
    if not polygons:
        return None
    return LocalizedRegion(
        region_label=prompt.region_label,
        matched_text=candidate.phrase or prompt.matched_term,
        confidence=max(0.0, min(1.0, candidate.confidence)),
        box=box,
        mask=polygons,
    )


def _mask_to_polygons(
    mask: np.ndarray,
    *,
    coordinate_offset: tuple[float, float],
) -> list[list[float]]:
    """Convert one binary mask to external contour polygons."""
    contours, _ = cv2.findContours(
        mask.astype("uint8") * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    x_offset, y_offset = coordinate_offset
    polygons = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        polygon = [
            coordinate
            for x_value, y_value in points
            for coordinate in (
                float(x_value) + x_offset,
                float(y_value) + y_offset,
            )
        ]
        polygons.append(polygon)
    return polygons
