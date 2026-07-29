"""Tests for FastAPI segmentation route wiring."""

from fastapi import HTTPException
import pytest

from fashion_semantic_parser.api.app import create_app
from fashion_semantic_parser.common.exceptions import (
    InvalidImageInputError,
    ModelNotReadyError,
)
from fashion_semantic_parser.config import Settings
from fashion_semantic_parser.models.schemas import MultimodalQueryRequest
from fashion_semantic_parser.models.segmentation import (
    SegmentationPrediction,
    SegmentationRequest,
    SegmentationSubjectROI,
)


class _FakeSegmentationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, SegmentationSubjectROI | None, bool]] = []

    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = False,
    ) -> SegmentationPrediction:
        self.calls.append((image_path, subject_roi, auto_subject_roi))
        return SegmentationPrediction(image_path=image_path, instances=[])


class _InvalidImageSegmentationService:
    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = False,
    ) -> SegmentationPrediction:
        raise InvalidImageInputError(f"Input image not found: {image_path}")


class _UnavailableSegmentationService:
    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = False,
    ) -> SegmentationPrediction:
        raise ModelNotReadyError("segmentation weights are unavailable")


def _segment_endpoint(app: object) -> object:
    return next(
        route.endpoint
        for route in app.routes  # type: ignore[attr-defined]
        if getattr(route, "path", None) == "/v1/segment"
    )


def _query_endpoint(app: object) -> object:
    return next(
        route.endpoint
        for route in app.routes  # type: ignore[attr-defined]
        if getattr(route, "path", None) == "/v1/query"
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


def test_segment_route_forwards_automatic_subject_roi_mode() -> None:
    """The API should expose automatic person-crop inference explicitly."""
    service = _FakeSegmentationService()
    app = create_app(segmentation_service=service)

    _segment_endpoint(app)(  # type: ignore[operator]
        SegmentationRequest(
            image_path="data/example.jpg",
            auto_subject_roi=True,
        )
    )

    assert service.calls == [("data/example.jpg", None, True)]


def test_query_route_defaults_to_configured_automatic_subject_roi() -> None:
    """The main query path should use the accepted automatic ROI pipeline."""
    service = _FakeSegmentationService()
    app = create_app(segmentation_service=service)

    _query_endpoint(app)(  # type: ignore[operator]
        MultimodalQueryRequest(
            image_path="data/example.jpg",
            query="图中有哪些服饰？",
        )
    )

    assert service.calls == [("data/example.jpg", None, True)]


def test_query_route_can_disable_configured_automatic_subject_roi() -> None:
    """Application configuration should retain a full-image deployment mode."""
    service = _FakeSegmentationService()
    settings = Settings.model_validate(
        {"segmentation": {"query_auto_subject_roi": False}}
    )
    app = create_app(settings=settings, segmentation_service=service)

    _query_endpoint(app)(  # type: ignore[operator]
        MultimodalQueryRequest(
            image_path="data/example.jpg",
            query="图中有哪些服饰？",
        )
    )

    assert service.calls == [("data/example.jpg", None, False)]


def test_segment_request_rejects_manual_and_automatic_roi_together() -> None:
    """One request cannot select two competing ROI sources."""
    with pytest.raises(ValueError, match="cannot be used together"):
        SegmentationRequest(
            image_path="data/example.jpg",
            subject_roi=SegmentationSubjectROI(
                x_min=1.0,
                y_min=1.0,
                x_max=10.0,
                y_max=10.0,
            ),
            auto_subject_roi=True,
        )


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
