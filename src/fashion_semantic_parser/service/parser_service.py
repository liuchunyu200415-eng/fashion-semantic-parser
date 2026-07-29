"""High-level orchestration service for multimodal fashion parsing."""

import math

from fashion_semantic_parser.models.schemas import (
    BoundingBox,
    MultimodalQueryRequest,
    MultimodalQueryResponse,
    RegionPrediction,
)
from fashion_semantic_parser.models.segmentation import SegmentationPrediction
from fashion_semantic_parser.service.segmentation_runtime import (
    GarmentSegmentationService,
    SegmentationRuntime,
)


class FashionParserService:
    """Coordinate visual parsing, RAG retrieval, and answer generation."""

    def __init__(
        self,
        segmentation_service: SegmentationRuntime | None = None,
        *,
        default_auto_subject_roi: bool = True,
    ) -> None:
        """Create the parser with a shared garment segmentation runtime."""
        self.segmentation_service = (
            segmentation_service
            if segmentation_service is not None
            else GarmentSegmentationService()
        )
        self.default_auto_subject_roi = default_auto_subject_roi

    def answer_query(
        self,
        request: MultimodalQueryRequest,
    ) -> MultimodalQueryResponse:
        """Answer a multimodal fashion query.

        Args:
            request: User image path and natural language query.

        Returns:
            Structured answer with localized regions and references.

        Notes:
            PRD 3.1.1 is integrated here. Later language grounding, attribute,
            RAG, and answer-generation adapters remain separate milestones.
        """
        auto_subject_roi = request.auto_subject_roi
        if auto_subject_roi is None:
            auto_subject_roi = (
                self.default_auto_subject_roi and request.subject_roi is None
            )
        segmentation = self.segmentation_service.segment(
            request.image_path,
            subject_roi=request.subject_roi,
            auto_subject_roi=auto_subject_roi,
        )
        return MultimodalQueryResponse(
            answer=_segmentation_summary(segmentation),
            regions=[
                RegionPrediction(
                    label=instance.category_label,
                    box=BoundingBox(
                        x_min=math.floor(instance.box.x_min),
                        y_min=math.floor(instance.box.y_min),
                        x_max=math.ceil(instance.box.x_max),
                        y_max=math.ceil(instance.box.y_max),
                    ),
                    confidence=instance.confidence,
                )
                for instance in segmentation.instances
            ],
            segmentation=segmentation,
        )


def _segmentation_summary(prediction: SegmentationPrediction) -> str:
    """Describe the completed 3.1.1 stage without claiming full QA support."""
    if not prediction.instances:
        return "Garment segmentation completed; no garment instances were detected."

    labels = ", ".join(instance.category_label for instance in prediction.instances)
    return (
        "Garment segmentation completed. "
        f"Detected {len(prediction.instances)} instance(s): {labels}."
    )
