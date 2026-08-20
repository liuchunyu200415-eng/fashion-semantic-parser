"""Build compact Fashionpedia query-region records for DINOv2 alignment."""

import json
import re
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from fashion_semantic_parser.common.paths import to_project_relative_path
from fashion_semantic_parser.dao.fashionpedia import (
    category_records_by_id,
    dict_records,
    image_sort_key,
    is_integer,
    is_positive_number,
    normalize_coco_bbox_xywh,
    normalize_coco_segmentation,
    read_fashionpedia_json,
    resolve_fashionpedia_split_paths,
    safe_image_path,
    source_category_name,
)
from fashion_semantic_parser.dao.localization.referring_smoke import (
    ReferringQueryDimension,
    ReferringReferenceFrame,
)
from fashion_semantic_parser.dao.localization.taxonomy import (
    FashionpediaPartCategory,
    map_fashionpedia_part_category,
)
from fashion_semantic_parser.models.localization import LocalizationBoundingBox

TrainingLanguage = Literal["zh", "en"]
TrainingSplit = Literal["train", "validation"]
TrainingAugmentationMethod = Literal["template", "llm_paraphrase"]


class ReferringTrainingTarget(BaseModel):
    """One official Fashionpedia Mask referenced by a language expression."""

    source_annotation_id: int = Field(ge=0)
    label: str = Field(min_length=1)
    box: LocalizationBoundingBox

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        """Keep labels compact and reject whitespace-only values."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Training target label cannot be empty.")
        return normalized


class ReferringTrainingSample(BaseModel):
    """One image-query-target set for DINOv2 region-text alignment."""

    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_dataset: Literal["fashionpedia"] = "fashionpedia"
    split: Literal["train", "validation"]
    image_path: str = Field(min_length=1)
    source_image_id: int = Field(ge=0)
    query: str = Field(min_length=1)
    language: TrainingLanguage
    dimensions: list[ReferringQueryDimension] = Field(min_length=1)
    reference_frame: ReferringReferenceFrame | None = None
    target_label: str = Field(min_length=1)
    targets: list[ReferringTrainingTarget] = Field(min_length=1)
    template_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_attribute_ids: list[int] = Field(default_factory=list)
    reference_category: str | None = None
    augmentation_method: TrainingAugmentationMethod = "template"
    source_sample_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    generator_model: str | None = None

    @field_validator(
        "image_path",
        "query",
        "target_label",
        "reference_category",
        "generator_model",
    )
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        """Normalize required and optional human-readable fields."""
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Referring training text fields cannot be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_query_contract(self) -> "ReferringTrainingSample":
        """Keep dimensions, targets, and modifiers unambiguous."""
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("Training dimensions cannot contain duplicates.")
        if "basic" not in self.dimensions:
            raise ValueError(
                "Every referring training sample requires basic target text."
            )
        if "spatial" in self.dimensions and self.reference_frame is None:
            raise ValueError("Spatial training samples require a reference frame.")
        if "spatial" not in self.dimensions and self.reference_frame is not None:
            raise ValueError("Non-spatial training samples cannot set reference_frame.")
        if any(target.label != self.target_label for target in self.targets):
            raise ValueError("Every target label must match target_label.")
        if len(self.source_attribute_ids) != len(set(self.source_attribute_ids)):
            raise ValueError("source_attribute_ids cannot contain duplicates.")
        if self.augmentation_method == "llm_paraphrase":
            if self.source_sample_id is None or self.generator_model is None:
                raise ValueError(
                    "LLM paraphrases require source_sample_id and generator_model."
                )
        elif self.source_sample_id is not None or self.generator_model is not None:
            raise ValueError(
                "Template samples cannot define LLM paraphrase provenance."
            )
        return self


class ReferringTrainingPreparationSummary(BaseModel):
    """Counts and provenance for one compact Fashionpedia JSONL index."""

    split: str
    source_image_count: int
    selected_image_count: int
    source_annotation_count: int
    selected_annotation_count: int
    valid_part_annotation_count: int
    invalid_part_annotation_count: int
    missing_image_count: int
    output_sample_count: int
    target_reference_count: int
    dimension_counts: dict[str, int]
    language_counts: dict[str, int]
    category_counts: dict[str, int]
    template_counts: dict[str, int]
    relation_association_count: int
    unmatched_relation_part_count: int
    spatial_ambiguous_group_count: int
    incompatible_attribute_group_count: int
    source_annotation_path: str
    output_path: str
    summary_output_path: str
    mask_storage: Literal["source_annotation_reference"] = "source_annotation_reference"


class _PartRow(BaseModel):
    """Validated source annotation used while generating language records."""

    annotation_id: int
    category: FashionpediaPartCategory
    bbox: tuple[float, float, float, float]
    attribute_ids: tuple[int, ...]


class _GarmentRow(BaseModel):
    """One source garment box eligible for reliable part-on-garment relations."""

    annotation_id: int
    source_name: str
    english_name: str
    chinese_name: str
    bbox: tuple[float, float, float, float]


_GARMENT_NAMES: dict[str, tuple[str, str]] = {
    "shirt, blouse": ("shirt", "衬衫"),
    "top, t-shirt, sweatshirt": ("top", "上衣"),
    "sweater": ("sweater", "毛衣"),
    "cardigan": ("cardigan", "开衫"),
    "jacket": ("jacket", "夹克"),
    "vest": ("vest", "马甲"),
    "pants": ("pants", "裤子"),
    "shorts": ("shorts", "短裤"),
    "skirt": ("skirt", "半身裙"),
    "coat": ("coat", "大衣"),
    "dress": ("dress", "连衣裙"),
    "jumpsuit": ("jumpsuit", "连体裤"),
    "cape": ("cape", "披肩"),
}


# Full-dataset ETL keeps source validation, atomic writes, and audit counters in
# one transaction boundary so a partial index can never be published.
# pylint: disable-next=R0913,R0914,R0912,R0915
def prepare_fashionpedia_referring_training_data(
    *,
    root: Path,
    split: TrainingSplit,
    output_path: Path,
    summary_output_path: Path,
    limit: int | None = None,
    min_spatial_separation: float = 0.05,
    max_attributes_per_annotation: int = 2,
    progress_every: int = 1000,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> ReferringTrainingPreparationSummary:
    """Write compact language records that reference official source Masks."""
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to zero")
    if not 0.0 <= min_spatial_separation <= 1.0:
        raise ValueError("min_spatial_separation must be between zero and one")
    if max_attributes_per_annotation < 0:
        raise ValueError("max_attributes_per_annotation cannot be negative")
    if progress_every < 1:
        raise ValueError("progress_every must be at least one")

    annotation_path, image_root = resolve_fashionpedia_split_paths(root, split)
    source = read_fashionpedia_json(annotation_path)
    source_images = dict_records(source.get("images"))
    source_annotations = dict_records(source.get("annotations"))
    category_by_id = category_records_by_id(dict_records(source.get("categories")))
    attribute_names = _attribute_names_by_id(dict_records(source.get("attributes")))

    selected_images = sorted(source_images, key=image_sort_key)
    if limit is not None:
        selected_images = selected_images[:limit]
    selected_ids = {
        image["id"] for image in selected_images if is_integer(image.get("id"))
    }
    selected_annotations = [
        annotation
        for annotation in source_annotations
        if annotation.get("image_id") in selected_ids
    ]
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in selected_annotations:
        image_id = annotation.get("image_id")
        if is_integer(image_id):
            annotations_by_image[image_id].append(annotation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_suffix(f"{output_path.suffix}.tmp")
    dimension_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    valid_part_annotation_count = 0
    invalid_part_annotation_count = 0
    missing_image_count = 0
    output_sample_count = 0
    target_reference_count = 0
    relation_association_count = 0
    unmatched_relation_part_count = 0
    spatial_ambiguous_group_count = 0
    incompatible_attribute_group_count = 0
    last_progress_index = 0

    try:
        with tmp_output.open("w", encoding="utf-8") as output_file:
            for image_index, image in enumerate(selected_images, start=1):
                if progress_callback is not None and (
                    image_index == 1 or image_index % progress_every == 0
                ):
                    progress_callback(
                        image_index,
                        len(selected_images),
                        output_sample_count,
                    )
                    last_progress_index = image_index
                image_id = image.get("id")
                file_name = image.get("file_name")
                width = image.get("width")
                height = image.get("height")
                if not (
                    is_integer(image_id)
                    and isinstance(file_name, str)
                    and is_positive_number(width)
                    and is_positive_number(height)
                ):
                    continue
                part_rows, garment_rows, invalid_count = _source_rows_for_image(
                    annotations=annotations_by_image.get(image_id, []),
                    category_by_id=category_by_id,
                    max_attributes_per_annotation=max_attributes_per_annotation,
                )
                invalid_part_annotation_count += invalid_count
                valid_part_annotation_count += len(part_rows)
                if not part_rows:
                    continue
                image_path = safe_image_path(image_root, file_name)
                if not image_path.is_file():
                    missing_image_count += 1
                    continue
                samples, audit = build_referring_samples_for_image(
                    split=split,
                    image_path=to_project_relative_path(image_path),
                    source_image_id=image_id,
                    width=int(width),
                    height=int(height),
                    part_rows=part_rows,
                    garment_rows=garment_rows,
                    attribute_names=attribute_names,
                    min_spatial_separation=min_spatial_separation,
                )
                relation_association_count += audit["relation_association_count"]
                unmatched_relation_part_count += audit["unmatched_relation_part_count"]
                spatial_ambiguous_group_count += audit["spatial_ambiguous_group_count"]
                incompatible_attribute_group_count += audit[
                    "incompatible_attribute_group_count"
                ]
                for sample in samples:
                    output_file.write(
                        json.dumps(sample.model_dump(mode="json"), ensure_ascii=False)
                    )
                    output_file.write("\n")
                    output_sample_count += 1
                    target_reference_count += len(sample.targets)
                    language_counts[sample.language] += 1
                    category_counts[sample.target_label] += 1
                    template_counts[sample.template_id] += 1
                    for dimension in sample.dimensions:
                        dimension_counts[dimension] += 1
            if progress_callback is not None and last_progress_index != len(
                selected_images
            ):
                progress_callback(
                    len(selected_images),
                    len(selected_images),
                    output_sample_count,
                )
        if missing_image_count:
            raise FileNotFoundError(
                f"Fashionpedia {split} is missing {missing_image_count} image(s) "
                f"with valid local-part Masks under {image_root}."
            )
        tmp_output.replace(output_path)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise

    summary = ReferringTrainingPreparationSummary(
        split=split,
        source_image_count=len(source_images),
        selected_image_count=len(selected_images),
        source_annotation_count=len(source_annotations),
        selected_annotation_count=len(selected_annotations),
        valid_part_annotation_count=valid_part_annotation_count,
        invalid_part_annotation_count=invalid_part_annotation_count,
        missing_image_count=missing_image_count,
        output_sample_count=output_sample_count,
        target_reference_count=target_reference_count,
        dimension_counts=dict(sorted(dimension_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        category_counts=dict(sorted(category_counts.items())),
        template_counts=dict(sorted(template_counts.items())),
        relation_association_count=relation_association_count,
        unmatched_relation_part_count=unmatched_relation_part_count,
        spatial_ambiguous_group_count=spatial_ambiguous_group_count,
        incompatible_attribute_group_count=incompatible_attribute_group_count,
        source_annotation_path=to_project_relative_path(annotation_path),
        output_path=to_project_relative_path(output_path),
        summary_output_path=to_project_relative_path(summary_output_path),
    )
    summary_output_path.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return summary


# One image is the semantic grouping boundary for basic, spatial, attribute,
# and garment-relation targets.
# pylint: disable-next=too-many-arguments,too-many-locals
def build_referring_samples_for_image(
    *,
    split: TrainingSplit,
    image_path: str,
    source_image_id: int,
    width: int,
    height: int,
    part_rows: list[_PartRow],
    garment_rows: list[_GarmentRow],
    attribute_names: dict[int, str],
    min_spatial_separation: float,
) -> tuple[list[ReferringTrainingSample], dict[str, int]]:
    """Generate only expressions whose targets are deterministic in one image."""
    samples: list[ReferringTrainingSample] = []
    part_groups: dict[str, list[_PartRow]] = defaultdict(list)
    for row in part_rows:
        part_groups[row.category.english_name].append(row)

    spatial_ambiguous_group_count = 0
    for _, rows in sorted(part_groups.items()):
        category = rows[0].category
        rows = sorted(rows, key=lambda row: row.annotation_id)
        samples.extend(
            _bilingual_samples(
                split=split,
                image_path=image_path,
                source_image_id=source_image_id,
                category=category,
                rows=rows,
                dimensions=["basic"],
                template="basic",
                zh_query=f"这件衣服的{category.chinese_name}",
                en_query=f"the {category.english_name} on the garment",
            )
        )
        spatial_samples, ambiguous = _spatial_samples(
            split=split,
            image_path=image_path,
            source_image_id=source_image_id,
            category=category,
            rows=rows,
            width=width,
            height=height,
            min_separation=min_spatial_separation,
        )
        samples.extend(spatial_samples)
        spatial_ambiguous_group_count += ambiguous

    attribute_groups: dict[tuple[str, int], list[_PartRow]] = defaultdict(list)
    for row in part_rows:
        for attribute_id in row.attribute_ids:
            if attribute_id in attribute_names:
                attribute_groups[(row.category.english_name, attribute_id)].append(row)
    incompatible_attribute_group_count = 0
    for (_, attribute_id), rows in sorted(attribute_groups.items()):
        category = rows[0].category
        attribute_name = attribute_names[attribute_id]
        attribute_query = _attribute_query(category, attribute_name)
        if attribute_query is None:
            incompatible_attribute_group_count += 1
            continue
        samples.append(
            _sample(
                split=split,
                image_path=image_path,
                source_image_id=source_image_id,
                category=category,
                rows=rows,
                dimensions=["basic", "attribute"],
                template_id=f"attribute-{attribute_id}-en",
                query=attribute_query,
                language="en",
                source_attribute_ids=[attribute_id],
            )
        )

    relation_groups: dict[tuple[str, str], list[_PartRow]] = defaultdict(list)
    relation_garments: dict[tuple[str, str], _GarmentRow] = {}
    relation_association_count = 0
    unmatched_relation_part_count = 0
    for row in part_rows:
        containers = [
            garment
            for garment in garment_rows
            if _box_containment_ratio(row.bbox, garment.bbox) >= 0.80
        ]
        if len(containers) != 1:
            unmatched_relation_part_count += 1
            continue
        garment = containers[0]
        key = (row.category.english_name, garment.source_name)
        relation_groups[key].append(row)
        relation_garments[key] = garment
        relation_association_count += 1
    for key, rows in sorted(relation_groups.items()):
        garment = relation_garments[key]
        category = rows[0].category
        english_garment = garment.english_name
        if english_garment == "top":
            english_garment = "top garment"
        samples.extend(
            _bilingual_samples(
                split=split,
                image_path=image_path,
                source_image_id=source_image_id,
                category=category,
                rows=rows,
                dimensions=["basic", "relation"],
                template=f"relation-{garment.english_name}",
                zh_query=(f"这件{garment.chinese_name}上的{category.chinese_name}"),
                en_query=(f"the {category.english_name} on the {english_garment}"),
                reference_category=garment.english_name,
            )
        )

    return samples, {
        "relation_association_count": relation_association_count,
        "unmatched_relation_part_count": unmatched_relation_part_count,
        "spatial_ambiguous_group_count": spatial_ambiguous_group_count,
        "incompatible_attribute_group_count": incompatible_attribute_group_count,
    }


def _attribute_query(
    category: FashionpediaPartCategory,
    attribute_name: str,
) -> str | None:
    """Create natural attribute text and reject cross-part attribute hints."""
    match = re.fullmatch(r"(.+?)\s*\(([^()]+)\)", attribute_name)
    if match is None:
        return f"the {category.english_name} with {attribute_name}"
    value = match.group(1).strip()
    hint = match.group(2).strip().casefold()
    compatible_targets = {
        "neck": {"neckline", "collar"},
        "neckline": {"neckline", "collar"},
        "pocket": {"pocket"},
        "sleeve": {"sleeve"},
    }.get(hint)
    if (
        compatible_targets is not None
        and category.english_name not in compatible_targets
    ):
        return None
    return f"the {value} {category.english_name}"


def _source_rows_for_image(
    *,
    annotations: list[dict[str, Any]],
    category_by_id: dict[int, dict[str, Any]],
    max_attributes_per_annotation: int,
) -> tuple[list[_PartRow], list[_GarmentRow], int]:
    """Validate local parts and relation garments from official annotations."""
    parts: list[_PartRow] = []
    garments: list[_GarmentRow] = []
    invalid_part_count = 0
    for annotation in annotations:
        annotation_id = annotation.get("id")
        category_name = source_category_name(annotation, category_by_id)
        bbox = normalize_coco_bbox_xywh(annotation.get("bbox"))
        category = map_fashionpedia_part_category(category_name)
        if category is not None:
            segmentation = normalize_coco_segmentation(annotation.get("segmentation"))
            if (
                not is_integer(annotation_id)
                or bbox is None
                or segmentation is None
                or annotation.get("iscrowd") == 1
            ):
                invalid_part_count += 1
                continue
            assert isinstance(annotation_id, int)
            raw_attributes = annotation.get("attribute_ids")
            if not isinstance(raw_attributes, list):
                raw_attributes = []
            attributes = sorted(
                {value for value in raw_attributes if is_integer(value)}
            )[:max_attributes_per_annotation]
            parts.append(
                _PartRow(
                    annotation_id=int(annotation_id),
                    category=category,
                    bbox=tuple(bbox),
                    attribute_ids=tuple(attributes),
                )
            )
            continue

        garment_names = _GARMENT_NAMES.get(category_name)
        if garment_names is None or not is_integer(annotation_id) or bbox is None:
            continue
        assert isinstance(annotation_id, int)
        garments.append(
            _GarmentRow(
                annotation_id=int(annotation_id),
                source_name=category_name,
                english_name=garment_names[0],
                chinese_name=garment_names[1],
                bbox=tuple(bbox),
            )
        )
    return parts, garments, invalid_part_count


# Spatial extrema need the image geometry and complete same-part instance set.
# pylint: disable-next=too-many-arguments,too-many-locals
def _spatial_samples(
    *,
    split: TrainingSplit,
    image_path: str,
    source_image_id: int,
    category: FashionpediaPartCategory,
    rows: list[_PartRow],
    width: int,
    height: int,
    min_separation: float,
) -> tuple[list[ReferringTrainingSample], int]:
    """Generate unique image-frame extrema and skip near-tied expressions."""
    if len(rows) < 2:
        return [], 0
    samples: list[ReferringTrainingSample] = []
    ambiguous = 0
    specifications = (
        (
            "left",
            0,
            width,
            f"衣服左侧的{category.chinese_name}",
            f"the {category.english_name} on the left side of the garment",
        ),
        (
            "right",
            0,
            width,
            f"衣服右侧的{category.chinese_name}",
            f"the {category.english_name} on the right side of the garment",
        ),
        (
            "upper",
            1,
            height,
            f"衣服上部的{category.chinese_name}",
            f"the {category.english_name} in the upper part of the garment",
        ),
        (
            "lower",
            1,
            height,
            f"衣服下部的{category.chinese_name}",
            f"the {category.english_name} in the lower part of the garment",
        ),
    )
    for modifier, axis, axis_size, zh_query, en_query in specifications:
        reverse = modifier in ("right", "lower")
        ranked = sorted(
            rows,
            key=partial(_part_center_coordinate, axis=axis),
            reverse=reverse,
        )
        separation = abs(
            _box_center(ranked[0].bbox)[axis] - _box_center(ranked[1].bbox)[axis]
        )
        if separation < min_separation * axis_size:
            ambiguous += 1
            continue
        samples.extend(
            _bilingual_samples(
                split=split,
                image_path=image_path,
                source_image_id=source_image_id,
                category=category,
                rows=[ranked[0]],
                dimensions=["basic", "spatial"],
                template=f"spatial-{modifier}",
                zh_query=zh_query,
                en_query=en_query,
                reference_frame="image",
            )
        )
    return samples, ambiguous


# Shared sample construction deliberately keeps both language variants aligned.
# pylint: disable-next=too-many-arguments
def _bilingual_samples(
    *,
    split: TrainingSplit,
    image_path: str,
    source_image_id: int,
    category: FashionpediaPartCategory,
    rows: list[_PartRow],
    dimensions: list[ReferringQueryDimension],
    template: str,
    zh_query: str,
    en_query: str,
    reference_frame: ReferringReferenceFrame | None = None,
    reference_category: str | None = None,
) -> list[ReferringTrainingSample]:
    """Create paired Chinese and English expressions for the same targets."""
    language_queries: tuple[tuple[TrainingLanguage, str], ...] = (
        ("zh", zh_query),
        ("en", en_query),
    )
    return [
        _sample(
            split=split,
            image_path=image_path,
            source_image_id=source_image_id,
            category=category,
            rows=rows,
            dimensions=dimensions,
            template_id=f"{template}-{language}",
            query=query,
            language=language,
            reference_frame=reference_frame,
            reference_category=reference_category,
        )
        for language, query in language_queries
    ]


# Stable training provenance requires all source identifiers at construction.
# pylint: disable-next=too-many-arguments
def _sample(
    *,
    split: TrainingSplit,
    image_path: str,
    source_image_id: int,
    category: FashionpediaPartCategory,
    rows: list[_PartRow],
    dimensions: list[ReferringQueryDimension],
    template_id: str,
    query: str,
    language: TrainingLanguage,
    reference_frame: ReferringReferenceFrame | None = None,
    source_attribute_ids: list[int] | None = None,
    reference_category: str | None = None,
) -> ReferringTrainingSample:
    """Build one stable record with source-Mask references and xyxy boxes."""
    rows = sorted(rows, key=lambda row: row.annotation_id)
    annotation_suffix = "-".join(str(row.annotation_id) for row in rows)
    sample_id = (
        f"fashionpedia-{split}-{source_image_id}-{category.english_name}-"
        f"{template_id}-{annotation_suffix}"
    )
    return ReferringTrainingSample(
        id=sample_id,
        split=split,
        image_path=image_path,
        source_image_id=source_image_id,
        query=query,
        language=language,
        dimensions=dimensions,
        reference_frame=reference_frame,
        target_label=category.english_name,
        targets=[_target(row) for row in rows],
        template_id=template_id,
        source_attribute_ids=source_attribute_ids or [],
        reference_category=reference_category,
    )


def _target(row: _PartRow) -> ReferringTrainingTarget:
    """Convert one source xywh box to the external xyxy contract."""
    x, y, width, height = row.bbox
    return ReferringTrainingTarget(
        source_annotation_id=row.annotation_id,
        label=row.category.english_name,
        box=LocalizationBoundingBox(
            x_min=x,
            y_min=y,
            x_max=x + width,
            y_max=y + height,
        ),
    )


def _attribute_names_by_id(records: list[dict[str, Any]]) -> dict[int, str]:
    """Index official non-empty attribute names without inventing translations."""
    values: dict[int, str] = {}
    for record in records:
        attribute_id = record.get("id")
        name = record.get("name")
        if not is_integer(attribute_id) or not isinstance(name, str):
            continue
        assert isinstance(attribute_id, int)
        normalized = " ".join(name.strip().split())
        if normalized:
            values[int(attribute_id)] = normalized
    return values


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the center of one xywh box."""
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _part_center_coordinate(row: _PartRow, *, axis: int) -> float:
    """Return one typed center coordinate for deterministic spatial ranking."""
    return _box_center(row.bbox)[axis]


def _box_containment_ratio(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    """Measure the fraction of one part box inside one garment box."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    x_overlap = max(0.0, min(ix + iw, ox + ow) - max(ix, ox))
    y_overlap = max(0.0, min(iy + ih, oy + oh) - max(iy, oy))
    area = iw * ih
    return x_overlap * y_overlap / area if area > 0.0 else 0.0
