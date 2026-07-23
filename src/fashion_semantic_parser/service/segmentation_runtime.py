"""Application-facing runtime service for garment instance segmentation."""

from pathlib import Path
from threading import Lock
from typing import Callable, Protocol

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
    filter_prediction_by_subject_roi,
)


class SegmentationPredictor(Protocol):
    """Minimal predictor contract used by the service layer."""

    def predict_image(self, image_path: Path) -> SegmentationPrediction:
        """Predict garment instances for one image."""


class SegmentationRuntime(Protocol):
    """Application-layer segmentation contract used by API orchestration."""

    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
    ) -> SegmentationPrediction:
        """Segment one project-relative image."""


PredictorFactory = Callable[[SegmentationBaselineSettings], SegmentationPredictor]


class GarmentSegmentationService:
    """Load and reuse the configured PRD 3.1.1 segmentation predictor."""

    def __init__(
        self,
        config_path: str = "configs/segmentation_mask2former_deployment.yaml",
        *,
        predictor: SegmentationPredictor | None = None,
        predictor_factory: PredictorFactory = Detectron2SegmentationBaseline,
    ) -> None:
        """Create a lazy segmentation runtime.

        The heavyweight model is initialized only when the first valid image is
        requested, then reused by later API calls.
        """
        self.config_path = config_path
        self._predictor = predictor
        self._predictor_factory = predictor_factory
        self._predictor_lock = Lock()

    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
    ) -> SegmentationPrediction:
        """Segment one project-relative image and optionally apply a subject ROI."""
        resolved_image_path = self._resolve_image_path(image_path)
        try:
            prediction = self._get_predictor().predict_image(resolved_image_path)
        except (ConfigurationError, ModelNotReadyError):
            raise
        except OSError as error:
            raise ModelNotReadyError(
                f"Segmentation runtime asset could not be loaded: {error}"
            ) from error
        except ValueError as error:
            message = str(error)
            if "Unable to read image" in message:
                raise InvalidImageInputError(message) from error
            raise ModelNotReadyError(
                f"Segmentation runtime is not usable: {message}"
            ) from error

        if subject_roi is not None:
            prediction = filter_prediction_by_subject_roi(prediction, subject_roi)
        return prediction

    def _get_predictor(self) -> SegmentationPredictor:
        """Initialize the configured predictor once in a thread-safe manner."""
        if self._predictor is not None:
            return self._predictor

        with self._predictor_lock:
            if self._predictor is None:
                self._predictor = self._predictor_factory(self._load_settings())
        return self._predictor

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
        return resolved_path
