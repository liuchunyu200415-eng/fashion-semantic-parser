"""Tests for the application-facing garment segmentation runtime."""

from pathlib import Path
from typing import Any

import pytest

from fashion_semantic_parser.common.exceptions import (
    InvalidImageInputError,
    ModelNotReadyError,
)
from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
    SegmentationSubjectROI,
)
from fashion_semantic_parser.service.segmentation_runtime import (
    GarmentSegmentationService,
)


class _FakePredictor:
    def __init__(self) -> None:
        self.image_paths: list[Path] = []
        self.subject_rois: list[SegmentationSubjectROI | None] = []

    def predict_image(
        self,
        image_path: Path,
        subject_roi: SegmentationSubjectROI | None = None,
    ) -> SegmentationPrediction:
        self.image_paths.append(image_path)
        self.subject_rois.append(subject_roi)
        return SegmentationPrediction(
            image_path="README.md",
            instances=[
                _instance("top", 1, (10.0, 10.0, 50.0, 80.0)),
                _instance("bag", 7, (200.0, 200.0, 260.0, 280.0)),
            ],
        )


class _MissingAssetPredictor:
    def predict_image(
        self,
        image_path: Path,
        subject_roi: SegmentationSubjectROI | None = None,
    ) -> SegmentationPrediction:
        raise FileNotFoundError("missing model weights")


class _FakeSubjectROIDetector:
    def __init__(self, roi: SegmentationSubjectROI | None) -> None:
        self.roi = roi
        self.image_paths: list[Path] = []

    def detect(self, image_path: Path) -> SegmentationSubjectROI | None:
        self.image_paths.append(image_path)
        return self.roi


def _instance(
    label: str,
    category_id: int,
    box: tuple[float, float, float, float],
) -> SegmentationInstance:
    return SegmentationInstance(
        category_id=category_id,
        category_label=label,
        confidence=0.9,
        box=SegmentationBoundingBox(
            x_min=box[0],
            y_min=box[1],
            x_max=box[2],
            y_max=box[3],
        ),
        mask=[[box[0], box[1], box[2], box[3]]],
    )


def test_runtime_loads_config_once_and_reuses_predictor() -> None:
    """The deployment model adapter should be constructed only once."""
    fake_predictor = _FakePredictor()
    loaded_settings: list[Any] = []

    def predictor_factory(settings: Any) -> _FakePredictor:
        loaded_settings.append(settings)
        return fake_predictor

    service = GarmentSegmentationService(predictor_factory=predictor_factory)

    first = service.segment("README.md")
    second = service.segment("README.md")

    assert first == second
    assert len(loaded_settings) == 1
    assert loaded_settings[0].model_family == "mask2former"
    assert loaded_settings[0].weights.endswith("model_0001999.pth")
    assert loaded_settings[0].num_classes == 8
    assert loaded_settings[0].score_threshold == 0.6
    assert loaded_settings[0].min_size_test == 512
    assert loaded_settings[0].max_size_test == 853
    assert loaded_settings[0].subject_roi_margin == 0.15
    assert loaded_settings[0].precision == "fp16"
    assert len(fake_predictor.image_paths) == 2


def test_runtime_forwards_optional_subject_roi_to_predictor() -> None:
    """Manual ROI should trigger crop-aware model inference."""
    predictor = _FakePredictor()
    service = GarmentSegmentationService(predictor=predictor)
    subject_roi = SegmentationSubjectROI(
        x_min=0.0,
        y_min=0.0,
        x_max=100.0,
        y_max=100.0,
    )

    prediction = service.segment(
        "README.md",
        subject_roi=subject_roi,
    )

    assert predictor.subject_rois == [subject_roi]
    assert len(prediction.instances) == 2
    assert prediction.subject_roi == subject_roi
    assert prediction.subject_roi_source == "manual"


def test_runtime_detects_automatic_subject_roi_before_segmentation() -> None:
    """Automatic mode should detect once and segment the detected crop."""
    subject_roi = SegmentationSubjectROI(
        x_min=10.0,
        y_min=20.0,
        x_max=200.0,
        y_max=300.0,
    )
    detector = _FakeSubjectROIDetector(subject_roi)
    predictor = _FakePredictor()
    service = GarmentSegmentationService(
        predictor=predictor,
        subject_roi_detector=detector,
    )

    prediction = service.segment("README.md", auto_subject_roi=True)

    assert len(detector.image_paths) == 1
    assert predictor.subject_rois == [subject_roi]
    assert prediction.subject_roi == subject_roi
    assert prediction.subject_roi_source == "detected"


def test_runtime_falls_back_to_full_image_when_person_is_not_detected() -> None:
    """Product-only images should remain usable without a person box."""
    detector = _FakeSubjectROIDetector(None)
    predictor = _FakePredictor()
    service = GarmentSegmentationService(
        predictor=predictor,
        subject_roi_detector=detector,
    )

    prediction = service.segment("README.md", auto_subject_roi=True)

    assert predictor.subject_rois == [None]
    assert prediction.subject_roi is None
    assert prediction.subject_roi_source == "full_image_fallback"


def test_subject_roi_rejects_reversed_coordinates() -> None:
    """An invalid xyxy ROI should fail before model inference."""
    with pytest.raises(ValueError, match="max coordinates"):
        SegmentationSubjectROI(
            x_min=100.0,
            y_min=20.0,
            x_max=10.0,
            y_max=80.0,
        )


def test_runtime_reports_missing_model_asset_as_not_ready() -> None:
    """Missing weights should become a stable service-level readiness error."""
    service = GarmentSegmentationService(predictor=_MissingAssetPredictor())

    with pytest.raises(ModelNotReadyError, match="missing model weights"):
        service.segment("README.md")


@pytest.mark.parametrize(
    "image_path",
    ["missing-image.jpg", "../outside-project.jpg", "/tmp/absolute.jpg"],
)
def test_runtime_rejects_invalid_image_paths(image_path: str) -> None:
    """Inference requests must reference an existing project-local file."""
    service = GarmentSegmentationService(predictor=_FakePredictor())

    with pytest.raises(InvalidImageInputError):
        service.segment(image_path)
