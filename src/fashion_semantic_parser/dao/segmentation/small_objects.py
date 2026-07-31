"""Build COCO training subsets centered on small garment instances."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field


class SmallObjectSubsetSummary(BaseModel):
    """Audit information for one small-object COCO subset."""

    source_path: str
    output_path: str | None
    target_categories: list[str]
    maximum_area: float
    source_image_count: int
    source_annotation_count: int
    selected_image_count: int
    selected_annotation_count: int
    target_small_annotation_count: int
    target_small_category_counts: dict[str, int] = Field(default_factory=dict)


def build_small_object_coco_subset(
    *,
    source_path: Path,
    output_path: Path | None,
    target_categories: Sequence[str] = ("shoes", "bag", "accessory"),
    maximum_area: float = float(32**2),
) -> SmallObjectSubsetSummary:
    """Select images containing target annotations below a COCO area limit.

    Every annotation from a selected image is retained so the duplicate training
    record keeps its original garment context. Image files are referenced, not
    copied.
    """
    if maximum_area <= 0.0:
        raise ValueError("maximum_area must be positive.")
    normalized_targets = tuple(
        dict.fromkeys(name.strip() for name in target_categories)
    )
    if not normalized_targets or any(not name for name in normalized_targets):
        raise ValueError("At least one non-empty target category is required.")

    source = _read_coco_mapping(source_path)
    categories = source["categories"]
    category_names_by_id = {
        int(category["id"]): str(category["name"]) for category in categories
    }
    category_ids_by_name = {
        category_name: category_id
        for category_id, category_name in category_names_by_id.items()
    }
    unknown_categories = set(normalized_targets) - set(category_ids_by_name)
    if unknown_categories:
        raise ValueError(
            "Unknown COCO target categories: " + ", ".join(sorted(unknown_categories))
        )
    target_category_ids = {category_ids_by_name[name] for name in normalized_targets}

    selected_image_ids: set[int] = set()
    target_counts: Counter[str] = Counter()
    for annotation in source["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id not in target_category_ids:
            continue
        area = _annotation_area(annotation)
        if area is None or not 0.0 < area < maximum_area:
            continue
        selected_image_ids.add(int(annotation["image_id"]))
        target_counts[category_names_by_id[category_id]] += 1

    selected_images = [
        image for image in source["images"] if int(image["id"]) in selected_image_ids
    ]
    selected_annotations = [
        annotation
        for annotation in source["annotations"]
        if int(annotation["image_id"]) in selected_image_ids
    ]

    if output_path is not None:
        output = dict(source)
        output["images"] = selected_images
        output["annotations"] = selected_annotations
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return SmallObjectSubsetSummary(
        source_path=str(source_path),
        output_path=str(output_path) if output_path is not None else None,
        target_categories=list(normalized_targets),
        maximum_area=maximum_area,
        source_image_count=len(source["images"]),
        source_annotation_count=len(source["annotations"]),
        selected_image_count=len(selected_images),
        selected_annotation_count=len(selected_annotations),
        target_small_annotation_count=sum(target_counts.values()),
        target_small_category_counts={
            name: target_counts[name] for name in normalized_targets
        },
    )


def _read_coco_mapping(path: Path) -> dict[str, Any]:
    """Read and minimally validate a COCO mapping."""
    with path.open("r", encoding="utf-8") as file:
        source = json.load(file)
    if not isinstance(source, dict):
        raise ValueError(f"Expected a COCO JSON object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(source.get(key), list):
            raise ValueError(f"COCO field must be a list: {key}")
    return source


def _annotation_area(annotation: dict[str, Any]) -> float | None:
    """Read mask area, falling back to COCO xywh box area."""
    if annotation.get("area") is not None:
        return float(annotation["area"])
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    return float(bbox[2]) * float(bbox[3])
