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
_FASHIONPEDIA_TO_PRD = {
    "shirt, blouse": "top",
    "top, t-shirt, sweatshirt": "top",
    "sweater": "top",
    "cardigan": "outerwear",
    "jacket": "outerwear",
    "vest": "top",
    "pants": "pants",
    "shorts": "pants",
    "skirt": "skirt",
    "coat": "outerwear",
    "dress": "dress",
    "cape": "outerwear",
    "glasses": "accessory",
    "hat": "accessory",
    "headband, head covering, hair accessory": "accessory",
    "tie": "accessory",
    "glove": "accessory",
    "watch": "accessory",
    "belt": "accessory",
    "leg warmer": "accessory",
    "tights, stockings": "accessory",
    "sock": "accessory",
    "shoe": "shoes",
    "bag, wallet": "bag",
    "scarf": "accessory",
    "umbrella": "accessory",
}
FASHIONPEDIA_AMBIGUOUS_CATEGORIES = frozenset({"jumpsuit"})
FASHIONPEDIA_GARMENT_PART_CATEGORIES = frozenset(
    {
        "hood",
        "collar",
        "lapel",
        "epaulette",
        "sleeve",
        "pocket",
        "neckline",
        "buckle",
        "zipper",
        "applique",
        "bead",
        "bow",
        "flower",
        "fringe",
        "ribbon",
        "rivet",
        "ruffle",
        "sequin",
        "tassel",
    }
)
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


def map_fashionpedia_category(category_name: str) -> SegmentationCategory | None:
    """Map one Fashionpedia main-apparel category to the PRD taxonomy."""
    english_name = _FASHIONPEDIA_TO_PRD.get(_normalize_category_name(category_name))
    if english_name is None:
        return None
    return _ENGLISH_NAME_TO_CATEGORY[english_name]


def fashionpedia_category_exclusion_reason(category_name: str) -> str | None:
    """Explain why one Fashionpedia category is excluded from PRD training."""
    normalized_name = _normalize_category_name(category_name)
    if normalized_name in FASHIONPEDIA_AMBIGUOUS_CATEGORIES:
        return "ambiguous_main_apparel"
    if normalized_name in FASHIONPEDIA_GARMENT_PART_CATEGORIES:
        return "garment_part"
    if normalized_name not in _FASHIONPEDIA_TO_PRD:
        return "unknown"
    return None


def _normalize_category_name(category_name: str) -> str:
    """Normalize external taxonomy labels before exact mapping."""
    return " ".join(category_name.strip().lower().split())
