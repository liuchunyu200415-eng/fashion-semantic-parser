"""Domain data models for fashion semantic parsing."""

from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
    SegmentationSubjectROI,
)

__all__ = [
    "SegmentationBoundingBox",
    "SegmentationInstance",
    "SegmentationPrediction",
    "SegmentationSubjectROI",
]
