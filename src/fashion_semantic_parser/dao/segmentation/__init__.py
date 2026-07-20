"""Segmentation dataset preparation utilities."""

from fashion_semantic_parser.dao.segmentation.coco import (
    COCOCategory,
    COCOConversionSummary,
    coco_categories,
    convert_deepfashion2_to_coco,
)
from fashion_semantic_parser.dao.segmentation.fashionpedia import (
    FashionpediaPreparationSummary,
    audit_fashionpedia_annotations,
    convert_fashionpedia_to_coco,
)
from fashion_semantic_parser.dao.segmentation.taxonomy import (
    FASHIONPEDIA_AMBIGUOUS_CATEGORIES,
    FASHIONPEDIA_GARMENT_PART_CATEGORIES,
    PRD_SEGMENTATION_CATEGORIES,
    fashionpedia_category_exclusion_reason,
    map_deepfashion2_category,
    map_fashionpedia_category,
)

__all__ = [
    "COCOCategory",
    "COCOConversionSummary",
    "FASHIONPEDIA_AMBIGUOUS_CATEGORIES",
    "FASHIONPEDIA_GARMENT_PART_CATEGORIES",
    "FashionpediaPreparationSummary",
    "PRD_SEGMENTATION_CATEGORIES",
    "audit_fashionpedia_annotations",
    "coco_categories",
    "convert_fashionpedia_to_coco",
    "convert_deepfashion2_to_coco",
    "fashionpedia_category_exclusion_reason",
    "map_deepfashion2_category",
    "map_fashionpedia_category",
]
