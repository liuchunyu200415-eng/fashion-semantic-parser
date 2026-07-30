"""Domain data models for fashion semantic parsing."""

from fashion_semantic_parser.models.localization import (
    LocalizationBoundingBox,
    LocalizedRegion,
    RegionLocalizationPrediction,
    RegionLocalizationRequest,
)
from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
    SegmentationRequest,
    SegmentationSubjectROI,
)

__all__ = [
    "LocalizationBoundingBox",
    "LocalizedRegion",
    "RegionLocalizationPrediction",
    "RegionLocalizationRequest",
    "SegmentationBoundingBox",
    "SegmentationInstance",
    "SegmentationPrediction",
    "SegmentationRequest",
    "SegmentationSubjectROI",
]
