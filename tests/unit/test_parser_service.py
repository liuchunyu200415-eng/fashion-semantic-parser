"""Tests for multimodal parser orchestration."""

import pytest

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
