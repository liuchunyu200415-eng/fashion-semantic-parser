"""Fashionpedia part taxonomy and PRD 3.1.2 coverage boundaries."""

import re
from typing import Literal

from pydantic import BaseModel

CoverageStatus = Literal["exact", "partial", "missing"]
HorizontalConstraint = Literal["left", "right"]
VerticalConstraint = Literal["upper", "lower"]


class FashionpediaPartCategory(BaseModel):
    """One directly annotated Fashionpedia local-part category."""

    id: int
    source_id: int
    english_name: str
    chinese_name: str
    region_group: str
    prompt_terms: tuple[str, ...]


class PRDRegionCoverage(BaseModel):
    """Current annotation coverage for one example region in PRD 3.1.2."""

    english_name: str
    chinese_name: str
    status: CoverageStatus
    source_categories: tuple[str, ...] = ()
    note: str


class LocalizationPrompt(BaseModel):
    """Normalized English grounding prompt for one user query."""

    region_label: str
    grounding_prompt: str
    matched_term: str


class LocalizationQueryConstraints(BaseModel):
    """Composable image-coordinate constraints retained from the full query."""

    horizontal: HorizontalConstraint | None = None
    vertical: VerticalConstraint | None = None


def pluralize_english_part_name(value: str) -> str:
    """Pluralize one controlled Fashionpedia English part name."""
    normalized = value.strip()
    if re.search(r"[^aeiou]y$", normalized, re.IGNORECASE):
        return f"{normalized[:-1]}ies"
    if re.search(r"(?:s|x|z|ch|sh)$", normalized, re.IGNORECASE):
        return f"{normalized}es"
    return f"{normalized}s"


FASHIONPEDIA_PART_CATEGORIES = [
    FashionpediaPartCategory(
        id=1,
        source_id=27,
        english_name="hood",
        chinese_name="帽兜",
        region_group="hood",
        prompt_terms=("hood", "帽兜", "连帽"),
    ),
    FashionpediaPartCategory(
        id=2,
        source_id=28,
        english_name="collar",
        chinese_name="衣领",
        region_group="collar",
        prompt_terms=("collar", "衣领", "领子"),
    ),
    FashionpediaPartCategory(
        id=3,
        source_id=29,
        english_name="lapel",
        chinese_name="翻领",
        region_group="collar",
        prompt_terms=("lapel", "翻领", "驳领"),
    ),
    FashionpediaPartCategory(
        id=4,
        source_id=30,
        english_name="epaulette",
        chinese_name="肩章",
        region_group="shoulder",
        prompt_terms=("epaulette", "肩章", "肩部装饰"),
    ),
    FashionpediaPartCategory(
        id=5,
        source_id=31,
        english_name="sleeve",
        chinese_name="袖子",
        region_group="sleeve",
        prompt_terms=("sleeve", "袖子", "衣袖"),
    ),
    FashionpediaPartCategory(
        id=6,
        source_id=32,
        english_name="pocket",
        chinese_name="口袋",
        region_group="pocket",
        prompt_terms=("pocket", "口袋", "衣袋"),
    ),
    FashionpediaPartCategory(
        id=7,
        source_id=33,
        english_name="neckline",
        chinese_name="领口",
        region_group="collar",
        prompt_terms=("neckline", "领口", "领口线"),
    ),
    FashionpediaPartCategory(
        id=8,
        source_id=34,
        english_name="buckle",
        chinese_name="搭扣",
        region_group="decoration",
        prompt_terms=("buckle", "搭扣", "扣环"),
    ),
    FashionpediaPartCategory(
        id=9,
        source_id=35,
        english_name="zipper",
        chinese_name="拉链",
        region_group="decoration",
        prompt_terms=("zipper", "拉链"),
    ),
    FashionpediaPartCategory(
        id=10,
        source_id=36,
        english_name="applique",
        chinese_name="贴花",
        region_group="decoration",
        prompt_terms=("applique", "贴花", "贴布"),
    ),
    FashionpediaPartCategory(
        id=11,
        source_id=37,
        english_name="bead",
        chinese_name="珠饰",
        region_group="decoration",
        prompt_terms=("bead", "珠饰", "钉珠"),
    ),
    FashionpediaPartCategory(
        id=12,
        source_id=38,
        english_name="bow",
        chinese_name="蝴蝶结",
        region_group="decoration",
        prompt_terms=("bow", "蝴蝶结"),
    ),
    FashionpediaPartCategory(
        id=13,
        source_id=39,
        english_name="flower",
        chinese_name="花饰",
        region_group="decoration",
        prompt_terms=("flower decoration", "花饰"),
    ),
    FashionpediaPartCategory(
        id=14,
        source_id=40,
        english_name="fringe",
        chinese_name="流苏边",
        region_group="decoration",
        prompt_terms=("fringe", "流苏边"),
    ),
    FashionpediaPartCategory(
        id=15,
        source_id=41,
        english_name="ribbon",
        chinese_name="丝带",
        region_group="decoration",
        prompt_terms=("ribbon", "丝带"),
    ),
    FashionpediaPartCategory(
        id=16,
        source_id=42,
        english_name="rivet",
        chinese_name="铆钉",
        region_group="decoration",
        prompt_terms=("rivet", "铆钉"),
    ),
    FashionpediaPartCategory(
        id=17,
        source_id=43,
        english_name="ruffle",
        chinese_name="荷叶边",
        region_group="decoration",
        prompt_terms=("ruffle", "荷叶边", "褶边"),
    ),
    FashionpediaPartCategory(
        id=18,
        source_id=44,
        english_name="sequin",
        chinese_name="亮片",
        region_group="decoration",
        prompt_terms=("sequin", "亮片"),
    ),
    FashionpediaPartCategory(
        id=19,
        source_id=45,
        english_name="tassel",
        chinese_name="穗饰",
        region_group="decoration",
        prompt_terms=("tassel", "穗饰", "流苏"),
    ),
]

PRD_LOCALIZATION_REGION_COVERAGE = [
    PRDRegionCoverage(
        english_name="collar",
        chinese_name="领口",
        status="exact",
        source_categories=("collar", "lapel", "neckline"),
        note="Fashionpedia provides direct masks for collar-related regions.",
    ),
    PRDRegionCoverage(
        english_name="cuff",
        chinese_name="袖口",
        status="missing",
        note="Sleeve masks cover the full sleeve and must not be relabeled as cuffs.",
    ),
    PRDRegionCoverage(
        english_name="hem",
        chinese_name="下摆",
        status="missing",
        note="No direct Fashionpedia hem mask is available.",
    ),
    PRDRegionCoverage(
        english_name="pocket",
        chinese_name="口袋",
        status="exact",
        source_categories=("pocket",),
        note="Fashionpedia provides direct pocket masks.",
    ),
    PRDRegionCoverage(
        english_name="shoulder",
        chinese_name="肩部",
        status="partial",
        source_categories=("epaulette",),
        note="Epaulette masks cover shoulder decorations, not the full shoulder.",
    ),
    PRDRegionCoverage(
        english_name="waist",
        chinese_name="腰部",
        status="missing",
        note="Belts are object masks and are not equivalent to a waist region.",
    ),
    PRDRegionCoverage(
        english_name="pattern",
        chinese_name="图案",
        status="missing",
        note="Pattern attributes exist, but the dataset has no general pattern masks.",
    ),
    PRDRegionCoverage(
        english_name="decoration",
        chinese_name="装饰",
        status="exact",
        source_categories=(
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
        ),
        note="Fashionpedia provides direct masks for closures and decorations.",
    ),
]

_SOURCE_NAME_TO_CATEGORY = {
    category.english_name: category for category in FASHIONPEDIA_PART_CATEGORIES
}

_PRD_QUERY_PROMPTS = (
    (
        "collar",
        "neckline . collar . lapel",
        ("领口", "衣领", "领子", "neckline", "collar"),
    ),
    (
        "cuff",
        "cuff . sleeve cuff",
        ("袖口", "cuff", "sleeve cuff"),
    ),
    (
        "hem",
        "garment hem . lower hem",
        ("下摆", "衣摆", "hem", "lower hem"),
    ),
    (
        "pocket",
        "pocket",
        ("口袋", "衣袋", "pocket"),
    ),
    (
        "shoulder",
        "shoulder area . epaulette",
        ("肩部", "肩膀", "shoulder", "shoulder area"),
    ),
    (
        "waist",
        "waist area . waistline",
        ("腰部", "腰线", "waist", "waistline"),
    ),
    (
        "pattern",
        "fabric pattern . printed pattern",
        ("图案", "花纹", "纹样", "pattern", "printed pattern"),
    ),
    (
        "decoration",
        "clothing decoration . embellishment",
        ("装饰", "装饰物", "decoration", "embellishment"),
    ),
)


def map_fashionpedia_part_category(
    category_name: str,
) -> FashionpediaPartCategory | None:
    """Map one official Fashionpedia part name to the localization taxonomy."""
    normalized_name = " ".join(category_name.strip().lower().split())
    return _SOURCE_NAME_TO_CATEGORY.get(normalized_name)


def resolve_localization_prompt(query: str) -> LocalizationPrompt:
    """Map Chinese or English fashion-part language to an English prompt.

    Grounding DINO's official Swin-T checkpoint uses an English text encoder.
    Known PRD and Fashionpedia terms are therefore normalized before inference,
    while an unknown English free-form query is retained as a custom prompt.
    """
    normalized_query = " ".join(query.strip().lower().split())
    if not normalized_query:
        raise ValueError("Localization query cannot be empty.")

    candidates: list[tuple[int, int, LocalizationPrompt]] = []
    for region_label, grounding_prompt, terms in _PRD_QUERY_PROMPTS:
        for term in terms:
            normalized_term = term.lower()
            if normalized_term in normalized_query:
                candidates.append(
                    (
                        len(normalized_term),
                        1,
                        LocalizationPrompt(
                            region_label=region_label,
                            grounding_prompt=grounding_prompt,
                            matched_term=term,
                        ),
                    )
                )

    for category in FASHIONPEDIA_PART_CATEGORIES:
        english_prompt = category.prompt_terms[0]
        for term in category.prompt_terms:
            normalized_term = term.lower()
            if normalized_term in normalized_query:
                candidates.append(
                    (
                        len(normalized_term),
                        2,
                        LocalizationPrompt(
                            region_label=category.english_name,
                            grounding_prompt=english_prompt,
                            matched_term=term,
                        ),
                    )
                )

    if candidates:
        return max(candidates, key=lambda candidate: candidate[:2])[2]
    return LocalizationPrompt(
        region_label="custom",
        grounding_prompt=normalized_query,
        matched_term=query.strip(),
    )


def resolve_localization_query_constraints(
    query: str,
) -> LocalizationQueryConstraints:
    """Extract unambiguous image-coordinate modifiers without losing the query."""
    normalized_query = " ".join(query.strip().lower().split())
    if not normalized_query:
        raise ValueError("Localization query cannot be empty.")

    has_left = _contains_any(
        normalized_query,
        ("左边", "左侧", "画面左", "left", "leftmost"),
    )
    has_right = _contains_any(
        normalized_query,
        ("右边", "右侧", "画面右", "right", "rightmost"),
    )
    has_upper = _contains_any(
        normalized_query,
        ("最上面", "上方", "顶部", "upper", "topmost", "at the top"),
    )
    has_lower = _contains_any(
        normalized_query,
        ("最下面", "下面的", "下方", "底部", "lower", "bottommost"),
    )
    return LocalizationQueryConstraints(
        horizontal=(
            "left"
            if has_left and not has_right
            else "right" if has_right and not has_left else None
        ),
        vertical=(
            "upper"
            if has_upper and not has_lower
            else "lower" if has_lower and not has_upper else None
        ),
    )


def _contains_any(query: str, terms: tuple[str, ...]) -> bool:
    """Match Chinese substrings and whole English terms without false positives."""
    for term in terms:
        if term.isascii():
            if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", query):
                return True
        elif term in query:
            return True
    return False


def localization_coco_categories() -> list[dict[str, object]]:
    """Return COCO categories with prompt metadata for text grounding."""
    return [
        {
            "id": category.id,
            "name": category.english_name,
            "supercategory": category.region_group,
            "chinese_name": category.chinese_name,
            "source_category_id": category.source_id,
            "prompt_terms": list(category.prompt_terms),
        }
        for category in FASHIONPEDIA_PART_CATEGORIES
    ]
