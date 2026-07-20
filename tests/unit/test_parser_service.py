"""Tests for multimodal parser orchestration."""

from fashion_semantic_parser.models.schemas import MultimodalQueryRequest
from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
    SegmentationSubjectROI,
)
from fashion_semantic_parser.service.parser_service import FashionParserService


class _FakeSegmentationService:
    def segment(
        self,
        image_path: str,
        subject_roi: SegmentationSubjectROI | None = None,
    ) -> SegmentationPrediction:
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
    service = FashionParserService(_FakeSegmentationService())

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
