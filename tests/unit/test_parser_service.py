"""Tests for multimodal parser orchestration."""

import pytest

from fashion_semantic_parser.models.localization import (
    LocalizationBoundingBox,
    LocalizedRegion,
    RegionLocalizationPrediction,
)
from fashion_semantic_parser.models.schemas import MultimodalQueryRequest
from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
    SegmentationSubjectROI,
)
from fashion_semantic_parser.service.parser_service import FashionParserService


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
        return SegmentationPrediction(
            image_path=image_path,
            instances=[
                SegmentationInstance(
                    category_id=1,
                    category_label="top",
                    confidence=0.93,
                    box=SegmentationBoundingBox(
                        x_min=10.2,
                        y_min=20.8,
                        x_max=110.1,
                        y_max=220.9,
                    ),
                    mask=[[10.2, 20.8, 110.1, 220.9]],
                )
            ],
        )


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
                    matched_text="衣领",
                    confidence=0.88,
                    box=LocalizationBoundingBox(
                        x_min=20.2,
                        y_min=30.8,
                        x_max=60.1,
                        y_max=80.9,
                    ),
                    mask=[[20.2, 30.8, 60.1, 30.8, 60.1, 80.9]],
                )
            ],
        )


class _GarmentAwareFakeLocalizationService(_FakeLocalizationService):
    def __init__(self) -> None:
        super().__init__()
        self.garment_predictions: list[SegmentationPrediction] = []

    def localize_with_garment_prediction(
        self,
        image_path: str,
        query: str,
        garment_prediction: SegmentationPrediction,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        self.garment_predictions.append(garment_prediction)
        return self.localize(
            image_path,
            query,
            subject_roi=subject_roi,
            auto_subject_roi=auto_subject_roi,
        )


def test_query_returns_integrated_segmentation_result() -> None:
    """The existing query route should no longer be a fixed not-ready stub."""
    segmentation_service = _FakeSegmentationService()
    service = FashionParserService(segmentation_service)

    response = service.answer_query(
        MultimodalQueryRequest(image_path="data/example.jpg", query="这是什么服饰？")
    )

    assert "1 instance" in response.answer
    assert response.regions[0].label == "top"
    assert response.regions[0].box.model_dump() == {
        "x_min": 10,
        "y_min": 20,
        "x_max": 111,
        "y_max": 221,
    }
    assert response.segmentation is not None
    assert response.segmentation.instances[0].mask
    assert segmentation_service.calls == [("data/example.jpg", None, True)]


def test_query_routes_known_part_language_through_localization() -> None:
    """The main query path should compose segmentation and 3.1.2 localization."""
    segmentation_service = _FakeSegmentationService()
    localization_service = _FakeLocalizationService()
    service = FashionParserService(
        segmentation_service,
        localization_service=localization_service,
    )

    response = service.answer_query(
        MultimodalQueryRequest(
            image_path="data/example.jpg",
            query="这件衣服的衣领在哪里？",
        )
    )

    assert "localization completed" in response.answer
    assert response.regions[0].label == "collar"
    assert response.localization is not None
    assert response.localization.regions[0].mask
    assert response.segmentation is not None
    assert localization_service.calls == [
        ("data/example.jpg", "这件衣服的衣领在哪里？", None, False)
    ]


def test_query_reuses_segmentation_for_garment_aware_localization() -> None:
    """Garment-derived regions should receive the current 3.1.1 prediction."""
    segmentation_service = _FakeSegmentationService()
    localization_service = _GarmentAwareFakeLocalizationService()
    service = FashionParserService(
        segmentation_service,
        localization_service=localization_service,
    )

    response = service.answer_query(
        MultimodalQueryRequest(
            image_path="data/example.jpg",
            query="这件衣服的下摆在哪里？",
        )
    )

    assert response.localization is not None
    assert len(localization_service.garment_predictions) == 1
    assert localization_service.garment_predictions[0] is response.segmentation
    assert segmentation_service.calls == [("data/example.jpg", None, True)]


def test_query_does_not_localize_general_garment_questions() -> None:
    """General garment queries should retain the accepted 3.1.1 response path."""
    localization_service = _FakeLocalizationService()
    service = FashionParserService(
        _FakeSegmentationService(),
        localization_service=localization_service,
    )

    response = service.answer_query(
        MultimodalQueryRequest(
            image_path="data/example.jpg",
            query="图中有哪些服饰？",
        )
    )

    assert response.localization is None
    assert localization_service.calls == []


def test_query_can_explicitly_disable_automatic_subject_roi() -> None:
    """Clients should retain whole-image inference for controlled comparisons."""
    segmentation_service = _FakeSegmentationService()
    service = FashionParserService(segmentation_service)

    service.answer_query(
        MultimodalQueryRequest(
            image_path="data/example.jpg",
            query="这是什么服饰？",
            auto_subject_roi=False,
        )
    )

    assert segmentation_service.calls == [("data/example.jpg", None, False)]


def test_query_manual_roi_overrides_configured_automatic_default() -> None:
    """A manual subject box should not require an extra false auto flag."""
    subject_roi = SegmentationSubjectROI(
        x_min=10.0,
        y_min=20.0,
        x_max=200.0,
        y_max=300.0,
    )
    segmentation_service = _FakeSegmentationService()
    service = FashionParserService(segmentation_service)

    service.answer_query(
        MultimodalQueryRequest(
            image_path="data/example.jpg",
            query="这是什么服饰？",
            subject_roi=subject_roi,
        )
    )

    assert segmentation_service.calls == [("data/example.jpg", subject_roi, False)]


def test_query_rejects_manual_and_automatic_roi_together() -> None:
    """Query requests should reject conflicting ROI modes before inference."""
    with pytest.raises(ValueError, match="cannot be used together"):
        MultimodalQueryRequest(
            image_path="data/example.jpg",
            query="这是什么服饰？",
            subject_roi=SegmentationSubjectROI(
                x_min=10.0,
                y_min=20.0,
                x_max=200.0,
                y_max=300.0,
            ),
            auto_subject_roi=True,
        )
