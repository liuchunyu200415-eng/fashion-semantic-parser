"""High-level orchestration service for multimodal fashion parsing."""

import math
from collections.abc import Callable
from typing import cast

from fashion_semantic_parser.common.exceptions import InvalidImageInputError
from fashion_semantic_parser.dao.localization.taxonomy import (
    resolve_localization_prompt,
)
from fashion_semantic_parser.models.localization import RegionLocalizationPrediction
from fashion_semantic_parser.models.schemas import (
    BoundingBox,
    MultimodalQueryRequest,
    MultimodalQueryResponse,
    RegionPrediction,
)
from fashion_semantic_parser.models.segmentation import SegmentationPrediction
from fashion_semantic_parser.service.region_localization import (
    RegionLocalizationRuntime,
)
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
        localization_service: RegionLocalizationRuntime | None = None,
        default_auto_subject_roi: bool = True,
    ) -> None:
        """Create the parser with a shared garment segmentation runtime."""
        self.segmentation_service = (
            segmentation_service
            if segmentation_service is not None
            else GarmentSegmentationService()
        )
        self.localization_service = localization_service
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
        localization = self._localize_known_region_query(request, segmentation)
        if localization is not None:
            return MultimodalQueryResponse(
                answer=_localization_summary(localization),
                regions=[
                    RegionPrediction(
                        label=region.region_label,
                        box=BoundingBox(
                            x_min=math.floor(region.box.x_min),
                            y_min=math.floor(region.box.y_min),
                            x_max=math.ceil(region.box.x_max),
                            y_max=math.ceil(region.box.y_max),
                        ),
                        confidence=region.confidence,
                    )
                    for region in localization.regions
                ],
                segmentation=segmentation,
                localization=localization,
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

    def _localize_known_region_query(
        self,
        request: MultimodalQueryRequest,
        segmentation: SegmentationPrediction,
    ) -> RegionLocalizationPrediction | None:
        """Run 3.1.2 only when the query names a known local region."""
        if self.localization_service is None:
            return None
        try:
            prompt = resolve_localization_prompt(request.query)
        except ValueError as error:
            raise InvalidImageInputError(str(error)) from error
        if prompt.region_label == "custom":
            return None
        localize_with_garment_prediction = getattr(
            self.localization_service,
            "localize_with_garment_prediction",
            None,
        )
        if callable(localize_with_garment_prediction):
            localization_callable = cast(
                Callable[..., RegionLocalizationPrediction],
                localize_with_garment_prediction,
            )
            # getattr plus callable performs the runtime protocol narrowing.
            localization = localization_callable(  # pylint: disable=not-callable
                request.image_path,
                request.query,
                segmentation,
                subject_roi=segmentation.subject_roi,
                auto_subject_roi=False,
            )
        else:
            localization = self.localization_service.localize(
                request.image_path,
                request.query,
                subject_roi=segmentation.subject_roi,
                auto_subject_roi=False,
            )
        return localization.model_copy(
            update={
                "subject_roi": segmentation.subject_roi,
                "subject_roi_source": segmentation.subject_roi_source,
            }
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


def _localization_summary(prediction: RegionLocalizationPrediction) -> str:
    """Describe a completed 3.1.2 query without claiming semantic QA."""
    if not prediction.regions:
        return "Fashion-part localization completed; no matching regions were found."
    labels = ", ".join(region.region_label for region in prediction.regions)
    return (
        "Fashion-part localization completed. "
        f"Detected {len(prediction.regions)} matching region(s): {labels}."
    )
