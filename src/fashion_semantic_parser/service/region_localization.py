"""Runtime contract for PRD 3.1.2 language-guided region localization."""

from typing import Protocol

from fashion_semantic_parser.common.exceptions import ModelNotReadyError
from fashion_semantic_parser.models.localization import RegionLocalizationPrediction
from fashion_semantic_parser.models.segmentation import SegmentationSubjectROI


class RegionLocalizationRuntime(Protocol):
    """Minimal interface implemented by language-guided localization runtimes."""

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Localize the image region described by natural language."""
        ...


class UnavailableRegionLocalizationService:
    """Explicit placeholder until Grounding DINO and SAM-HQ are configured."""

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Reject inference instead of returning an invented localization."""
        raise ModelNotReadyError(
            "PRD 3.1.2 language-guided localization is not configured. "
            "Prepare the Fashionpedia part dataset and install the Grounding "
            "DINO + SAM-HQ runtime before calling /v1/localize."
        )
