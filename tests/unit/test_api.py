"""Tests for FastAPI segmentation route wiring."""

from fastapi import HTTPException
import pytest

from fashion_semantic_parser.api.app import create_app
from fashion_semantic_parser.common.exceptions import (
    InvalidImageInputError,
    ModelNotReadyError,
)
from fashion_semantic_parser.config import Settings
from fashion_semantic_parser.models.localization import (
    LocalizationBoundingBox,
    LocalizedRegion,
    RegionLocalizationPrediction,
    RegionLocalizationRequest,
)
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


class _FakeLocalizationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, SegmentationSubjectROI | None, bool]] = []

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        self.calls.append((image_path, query, subject_roi, auto_subject_roi))
        return RegionLocalizationPrediction(
            image_path=image_path,
            query=query,
            regions=[
                LocalizedRegion(
                    region_label="collar",
                    matched_text="领口",
                    confidence=0.94,
                    box=LocalizationBoundingBox(
                        x_min=10,
                        y_min=20,
                        x_max=50,
                        y_max=60,
                    ),
                    mask=[[10, 20, 50, 20, 50, 60, 10, 60]],
                )
            ],
        )


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


def _localize_endpoint(app: object) -> object:
    return next(
        route.endpoint
        for route in app.routes  # type: ignore[attr-defined]
        if getattr(route, "path", None) == "/v1/localize"
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


def test_localize_route_returns_mask_and_box_for_language_query() -> None:
    """The PRD 3.1.2 route should expose a stable injectable output contract."""
    localization_service = _FakeLocalizationService()
    app = create_app(
        segmentation_service=_FakeSegmentationService(),
        localization_service=localization_service,
    )

    response = _localize_endpoint(app)(  # type: ignore[operator]
        RegionLocalizationRequest(
            image_path="data/example.jpg",
            query="这件衣服的领口",
        )
    )

    assert response.query == "这件衣服的领口"
    assert response.regions[0].region_label == "collar"
    assert response.regions[0].mask
    assert localization_service.calls == [
        ("data/example.jpg", "这件衣服的领口", None, True)
    ]


def test_localize_route_manual_roi_disables_automatic_default() -> None:
    """Manual subject selection should be forwarded without person detection."""
    subject_roi = SegmentationSubjectROI(
        x_min=1,
        y_min=2,
        x_max=100,
        y_max=200,
    )
    localization_service = _FakeLocalizationService()
    app = create_app(
        segmentation_service=_FakeSegmentationService(),
        localization_service=localization_service,
    )

    _localize_endpoint(app)(  # type: ignore[operator]
        RegionLocalizationRequest(
            image_path="data/example.jpg",
            query="口袋",
            subject_roi=subject_roi,
        )
    )

    assert localization_service.calls == [
        ("data/example.jpg", "口袋", subject_roi, False)
    ]


def test_localize_route_is_explicitly_unavailable_before_model_setup() -> None:
    """The endpoint must not invent results before its runtime is configured."""
    app = create_app(segmentation_service=_FakeSegmentationService())

    with pytest.raises(HTTPException) as captured:
        _localize_endpoint(app)(  # type: ignore[operator]
            RegionLocalizationRequest(
                image_path="data/example.jpg",
                query="袖口",
            )
        )

    assert captured.value.status_code == 503
    assert "Grounding DINO + SAM-HQ" in captured.value.detail


def test_localize_request_rejects_manual_and_automatic_roi_together() -> None:
    """Localization uses exactly one subject ROI source per request."""
    with pytest.raises(ValueError, match="cannot be used together"):
        RegionLocalizationRequest(
            image_path="data/example.jpg",
            query="领口",
            subject_roi=SegmentationSubjectROI(
                x_min=1,
                y_min=2,
                x_max=100,
                y_max=200,
            ),
            auto_subject_roi=True,
        )
