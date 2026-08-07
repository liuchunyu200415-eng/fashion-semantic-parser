"""Build complete-image COCO subsets for localization class replay."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field


class LocalizationTargetSubsetSummary(BaseModel):
    """Audit information for one targeted localization training subset."""

    source_path: str
    output_path: str | None
    target_categories: list[str]
    source_image_count: int
    source_annotation_count: int
    selected_image_count: int
    selected_annotation_count: int
    target_annotation_count: int
    target_annotation_counts: dict[str, int] = Field(default_factory=dict)


def build_localization_target_coco_subset(
    *,
    source_path: Path,
    output_path: Path | None,
    target_categories: Sequence[str],
) -> LocalizationTargetSubsetSummary:
    """Select images containing target classes while retaining all annotations."""
    normalized_targets = tuple(
        dict.fromkeys(name.strip() for name in target_categories)
    )
    if not normalized_targets or any(not name for name in normalized_targets):
        raise ValueError("At least one non-empty target category is required.")

    source = _read_coco_mapping(source_path)
    names_by_id = {
        int(category["id"]): str(category["name"]) for category in source["categories"]
    }
    ids_by_name = {name: category_id for category_id, name in names_by_id.items()}
    unknown = set(normalized_targets) - set(ids_by_name)
    if unknown:
        raise ValueError(
            "Unknown localization target categories: " + ", ".join(sorted(unknown))
        )
    target_ids = {ids_by_name[name] for name in normalized_targets}

    selected_image_ids: set[int] = set()
    target_counts: Counter[str] = Counter()
    for annotation in source["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id not in target_ids or int(annotation.get("iscrowd", 0)) != 0:
            continue
        selected_image_ids.add(int(annotation["image_id"]))
        target_counts[names_by_id[category_id]] += 1

    selected_images = [
        image for image in source["images"] if int(image["id"]) in selected_image_ids
    ]
    selected_annotations = [
        annotation
        for annotation in source["annotations"]
        if int(annotation["image_id"]) in selected_image_ids
    ]
    if not selected_images:
        raise ValueError("No images contain the requested target categories.")

    if output_path is not None:
        output = dict(source)
        output["images"] = selected_images
        output["annotations"] = selected_annotations
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return LocalizationTargetSubsetSummary(
        source_path=str(source_path),
        output_path=str(output_path) if output_path is not None else None,
        target_categories=list(normalized_targets),
        source_image_count=len(source["images"]),
        source_annotation_count=len(source["annotations"]),
        selected_image_count=len(selected_images),
        selected_annotation_count=len(selected_annotations),
        target_annotation_count=sum(target_counts.values()),
        target_annotation_counts={
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
