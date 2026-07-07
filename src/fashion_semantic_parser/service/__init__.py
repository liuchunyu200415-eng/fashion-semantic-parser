"""Business services for the fashion semantic parser."""

from fashion_semantic_parser.service.segmentation_baseline import (
    Detectron2SegmentationBaseline,
    SegmentationBaselineSettings,
)

__all__ = [
    "Detectron2SegmentationBaseline",
    "SegmentationBaselineSettings",
]
