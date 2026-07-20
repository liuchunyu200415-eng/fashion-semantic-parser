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

    def predict_image(self, image_path: Path) -> SegmentationPrediction:
        self.image_paths.append(image_path)
        return SegmentationPrediction(
            image_path="README.md",
            instances=[
                _instance("top", 1, (10.0, 10.0, 50.0, 80.0)),
                _instance("bag", 7, (200.0, 200.0, 260.0, 280.0)),
            ],
        )


class _MissingAssetPredictor:
    def predict_image(self, image_path: Path) -> SegmentationPrediction:
        raise FileNotFoundError("missing model weights")


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
    assert loaded_settings[0].score_threshold == 0.8
    assert len(fake_predictor.image_paths) == 2


def test_runtime_applies_optional_subject_roi() -> None:
    """Manual ROI remains available at the service boundary."""
    service = GarmentSegmentationService(predictor=_FakePredictor())

    prediction = service.segment(
        "README.md",
        subject_roi=SegmentationSubjectROI(
            x_min=0.0,
            y_min=0.0,
            x_max=100.0,
            y_max=100.0,
        ),
    )

    assert [instance.category_label for instance in prediction.instances] == ["top"]


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
