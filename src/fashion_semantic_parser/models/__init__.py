"""Domain data models for fashion semantic parsing."""

from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
)

__all__ = [
    "SegmentationBoundingBox",
    "SegmentationInstance",
    "SegmentationPrediction",
]
