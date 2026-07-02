"""Category taxonomy for PRD 3.1.1 garment instance segmentation."""

from pydantic import BaseModel


class SegmentationCategory(BaseModel):
    """One supported garment instance segmentation category."""

    id: int
    name: str
    english_name: str


PRD_SEGMENTATION_CATEGORIES = [
    SegmentationCategory(id=1, name="上衣", english_name="top"),
    SegmentationCategory(id=2, name="裤子", english_name="pants"),
    SegmentationCategory(id=3, name="裙子", english_name="skirt"),
    SegmentationCategory(id=4, name="外套", english_name="outerwear"),
    SegmentationCategory(id=5, name="连衣裙", english_name="dress"),
    SegmentationCategory(id=6, name="鞋子", english_name="shoes"),
    SegmentationCategory(id=7, name="包包", english_name="bag"),
    SegmentationCategory(id=8, name="配饰", english_name="accessory"),
]

_DEEPFASHION2_TO_PRD = {
    "short sleeve top": "top",
    "long sleeve top": "top",
    "vest": "top",
    "sling": "top",
    "trousers": "pants",
    "shorts": "pants",
    "skirt": "skirt",
    "short sleeve outwear": "outerwear",
    "long sleeve outwear": "outerwear",
    "short sleeve dress": "dress",
    "long sleeve dress": "dress",
    "vest dress": "dress",
    "sling dress": "dress",
}
_ENGLISH_NAME_TO_CATEGORY = {
    category.english_name: category for category in PRD_SEGMENTATION_CATEGORIES
}


def map_deepfashion2_category(category_name: str) -> SegmentationCategory | None:
    """Map a DeepFashion2 category name to a PRD segmentation category.

    Args:
        category_name: DeepFashion2 category name.

    Returns:
        PRD segmentation category if supported by DeepFashion2, otherwise ``None``.
    """
    english_name = _DEEPFASHION2_TO_PRD.get(category_name)
    if english_name is None:
        return None
    return _ENGLISH_NAME_TO_CATEGORY[english_name]
