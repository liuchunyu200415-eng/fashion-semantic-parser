"""Application-facing runtime service for garment instance segmentation."""

from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping, Protocol, cast

import yaml

from fashion_semantic_parser.common.exceptions import (
    ConfigurationError,
    InvalidImageInputError,
    ModelNotReadyError,
)
from fashion_semantic_parser.common.paths import resolve_project_path
from fashion_semantic_parser.models.segmentation import (
    SegmentationPrediction,
    SegmentationSubjectROI,
)
from fashion_semantic_parser.service.segmentation_baseline import (
    Detectron2SegmentationBaseline,
    SegmentationBaselineSettings,
)
from fashion_semantic_parser.service.subject_roi import (
    Detectron2PersonROIDetector,
    PersonROIDetectorSettings,
)


class SegmentationPredictor(Protocol):
    """Minimal predictor contract used by the service layer."""

    def predict_image(
        self,
        image_path: Path,
        subject_roi: SegmentationSubjectROI | None = None,
    ) -> SegmentationPrediction:
        """Predict garment instances for one image."""


class SegmentationRuntime(Protocol):
    """Application-layer segmentation contract used by API orchestration."""

    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = False,
    ) -> SegmentationPrediction:
        """Segment one project-relative image."""


class SubjectROIDetector(Protocol):
    """Minimal subject detector contract used by the runtime."""

    def detect(self, image_path: Path) -> SegmentationSubjectROI | None:
        """Return the primary subject box or None."""


PredictorFactory = Callable[[SegmentationBaselineSettings], SegmentationPredictor]
SubjectROIDetectorFactory = Callable[
    [SegmentationBaselineSettings],
    SubjectROIDetector,
]


class GarmentSegmentationService:
    """Load and reuse the configured PRD 3.1.1 segmentation predictor."""

    def __init__(
        self,
        config_path: str = "configs/segmentation_mask2former_deployment.yaml",
        *,
        predictor: SegmentationPredictor | None = None,
        predictor_factory: PredictorFactory = Detectron2SegmentationBaseline,
        subject_roi_detector: SubjectROIDetector | None = None,
        subject_roi_detector_factory: SubjectROIDetectorFactory | None = None,
        settings_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a lazy segmentation runtime.

        The heavyweight model is initialized only when the first valid image is
        requested, then reused by later API calls.
        """
        self.config_path = config_path
        self._predictor = predictor
        self._predictor_factory = predictor_factory
        self._predictor_lock = Lock()
        self._subject_roi_detector = subject_roi_detector
        self._subject_roi_detector_factory = (
            subject_roi_detector_factory or _build_default_subject_roi_detector
        )
        self._subject_roi_detector_lock = Lock()
        self._settings_overrides = dict(settings_overrides or {})

    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = False,
    ) -> SegmentationPrediction:
        """Segment one image, optionally using a cropped subject ROI."""
        if subject_roi is not None and auto_subject_roi:
            raise InvalidImageInputError(
                "subject_roi and auto_subject_roi cannot be used together"
            )
        resolved_image_path = self._resolve_image_path(image_path)
        try:
            effective_roi = subject_roi
            roi_source = "manual" if subject_roi is not None else None
            if auto_subject_roi:
                effective_roi = self._get_subject_roi_detector().detect(
                    resolved_image_path
                )
                roi_source = (
                    "detected" if effective_roi is not None else "full_image_fallback"
                )
            prediction = self._get_predictor().predict_image(
                resolved_image_path,
                subject_roi=effective_roi,
            )
        except (ConfigurationError, ModelNotReadyError):
            raise
        except OSError as error:
            raise ModelNotReadyError(
                f"Segmentation runtime asset could not be loaded: {error}"
            ) from error
        except ValueError as error:
            message = str(error)
            if "Unable to read image" in message or "Subject ROI" in message:
                raise InvalidImageInputError(message) from error
            raise ModelNotReadyError(
                f"Segmentation runtime is not usable: {message}"
            ) from error
        return prediction.model_copy(
            update={
                "subject_roi": effective_roi,
                "subject_roi_source": roi_source,
            }
        )

    def _get_predictor(self) -> SegmentationPredictor:
        """Initialize the configured predictor once in a thread-safe manner."""
        if self._predictor is not None:
            return self._predictor

        with self._predictor_lock:
            if self._predictor is None:
                self._predictor = self._predictor_factory(self._load_settings())
        return self._predictor

    def _get_subject_roi_detector(self) -> SubjectROIDetector:
        """Initialize the optional person detector once."""
        if self._subject_roi_detector is not None:
            return self._subject_roi_detector

        with self._subject_roi_detector_lock:
            if self._subject_roi_detector is None:
                self._subject_roi_detector = self._subject_roi_detector_factory(
                    self._load_settings()
                )
        return self._subject_roi_detector

    def _load_settings(self) -> SegmentationBaselineSettings:
        """Load the deployment YAML used by CLI and API inference."""
        try:
            config_path = resolve_project_path(self.config_path)
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
        if not config_path.is_file():
            raise ConfigurationError(
                f"Segmentation config file not found: {self.config_path}"
            )

        try:
            with config_path.open("r", encoding="utf-8") as file:
                raw_config = yaml.safe_load(file) or {}
        except yaml.YAMLError as error:
            raise ConfigurationError(
                f"Invalid segmentation YAML: {self.config_path}"
            ) from error
        if not isinstance(raw_config, dict):
            raise ConfigurationError(
                f"Expected a mapping in segmentation config: {self.config_path}"
            )
        raw_config.update(self._settings_overrides)

        try:
            return SegmentationBaselineSettings.model_validate(raw_config)
        except ValueError as error:
            raise ConfigurationError(
                f"Invalid segmentation settings in {self.config_path}: {error}"
            ) from error

    @staticmethod
    def _resolve_image_path(image_path: str) -> Path:
        """Validate that an API image path stays inside the project checkout."""
        try:
            resolved_path = resolve_project_path(image_path)
        except ValueError as error:
            raise InvalidImageInputError(str(error)) from error
        if not resolved_path.is_file():
            raise InvalidImageInputError(f"Input image not found: {image_path}")
        return Path(resolved_path)


def _build_default_subject_roi_detector(
    settings: SegmentationBaselineSettings,
) -> SubjectROIDetector:
    """Build a COCO person detector aligned with segmentation device settings."""
    return cast(
        SubjectROIDetector,
        Detectron2PersonROIDetector(
            PersonROIDetectorSettings(
                device=settings.device,
                precision=settings.precision,
            )
        ),
    )
