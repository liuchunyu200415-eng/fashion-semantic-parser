"""Tests for FastAPI segmentation route wiring."""

from fastapi import HTTPException
import pytest

from fashion_semantic_parser.api.app import create_app
from fashion_semantic_parser.common.exceptions import (
    InvalidImageInputError,
    ModelNotReadyError,
)
from fashion_semantic_parser.models.segmentation import (
    SegmentationPrediction,
    SegmentationRequest,
    SegmentationSubjectROI,
)


class _FakeSegmentationService:
    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
    ) -> SegmentationPrediction:
        return SegmentationPrediction(image_path=image_path, instances=[])


class _InvalidImageSegmentationService:
    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
    ) -> SegmentationPrediction:
        raise InvalidImageInputError(f"Input image not found: {image_path}")


class _UnavailableSegmentationService:
    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
    ) -> SegmentationPrediction:
        raise ModelNotReadyError("segmentation weights are unavailable")


def _segment_endpoint(app: object) -> object:
    return next(
        route.endpoint
        for route in app.routes  # type: ignore[attr-defined]
        if getattr(route, "path", None) == "/v1/segment"
    )


def test_segment_route_returns_typed_prediction() -> None:
    """The API should expose the exact PRD 3.1.1 prediction schema."""
    app = create_app(
        segmentation_service=_FakeSegmentationService(),
    )

    response = _segment_endpoint(app)(  # type: ignore[operator]
        SegmentationRequest(image_path="data/example.jpg")
    )

    assert response.image_path == "data/example.jpg"
    assert response.instances == []


def test_segment_route_maps_invalid_image_to_http_400() -> None:
    """Bad client image paths should not be reported as model outages."""
    invalid_service = _InvalidImageSegmentationService()
    app = create_app(
        segmentation_service=invalid_service,
    )

    with pytest.raises(HTTPException) as captured:
        _segment_endpoint(app)(  # type: ignore[operator]
            SegmentationRequest(image_path="missing.jpg")
        )

    assert captured.value.status_code == 400


def test_segment_route_maps_model_outage_to_http_503() -> None:
    """Missing runtime dependencies should be reported as service unavailable."""
    app = create_app(segmentation_service=_UnavailableSegmentationService())

    with pytest.raises(HTTPException) as captured:
        _segment_endpoint(app)(  # type: ignore[operator]
            SegmentationRequest(image_path="data/example.jpg")
        )

    assert captured.value.status_code == 503
