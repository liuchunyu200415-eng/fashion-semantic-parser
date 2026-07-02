"""Segmentation dataset preparation utilities."""

from fashion_semantic_parser.dao.segmentation.coco import (
    COCOCategory,
    COCOConversionSummary,
    convert_deepfashion2_to_coco,
)
from fashion_semantic_parser.dao.segmentation.taxonomy import (
    PRD_SEGMENTATION_CATEGORIES,
    map_deepfashion2_category,
)

__all__ = [
    "COCOCategory",
    "COCOConversionSummary",
    "PRD_SEGMENTATION_CATEGORIES",
    "convert_deepfashion2_to_coco",
    "map_deepfashion2_category",
]
