"""Schemas for a bounded open-language localization feasibility benchmark."""

import json
import math
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, field_validator, model_validator

from fashion_semantic_parser.models.localization import LocalizationBoundingBox

ReferringQueryDimension = Literal["basic", "spatial", "attribute", "relation"]
ReferringQueryNovelty = Literal[
    "seen",
    "novel_paraphrase",
    "novel_composition",
    "novel_part",
]
ReferringReferenceFrame = Literal["image", "person", "garment"]
ReferringAnnotationStatus = Literal["mask", "box", "negative", "unlabelled"]


class ReferringSmokeTarget(BaseModel):
    """One target instance referred to by a benchmark expression."""

    label: str | None = None
    box: LocalizationBoundingBox | None = None
    segmentation: Any | None = None

    @model_validator(mode="after")
    def validate_spatial_annotation(self) -> "ReferringSmokeTarget":
        """Require a usable box or a COCO-compatible segmentation."""
        if self.box is None and self.segmentation is None:
            raise ValueError("A referring target requires a box or segmentation.")
        if self.segmentation is not None and not isinstance(
            self.segmentation,
            (dict, list),
        ):
            raise ValueError("Target segmentation must be COCO polygon or RLE data.")
        if isinstance(self.segmentation, (dict, list)) and not self.segmentation:
            raise ValueError("Target segmentation cannot be empty.")
        if isinstance(self.segmentation, list):
            polygons = cast(list[Any], self.segmentation)
            # Pydantic narrows this runtime union, but Pylint cannot infer the cast.
            for polygon in polygons:  # pylint: disable=not-an-iterable
                if (
                    not isinstance(polygon, list)
                    or len(polygon) < 6
                    or len(polygon) % 2
                    or not all(
                        isinstance(value, (int, float)) and math.isfinite(float(value))
                        for value in polygon
                    )
                ):
                    raise ValueError(
                        "Target polygons require at least three numeric xy points."
                    )
        if isinstance(self.segmentation, dict):
            size = self.segmentation.get("size")
            counts = self.segmentation.get("counts")
            if (
                not isinstance(size, list)
                or len(size) != 2
                or not all(isinstance(value, int) and value > 0 for value in size)
                or not isinstance(counts, (str, list))
                or (
                    isinstance(counts, list)
                    and not all(
                        isinstance(value, int) and value >= 0 for value in counts
                    )
                )
            ):
                raise ValueError("Target RLE requires positive size and counts.")
        return self


class ReferringSmokeCase(BaseModel):
    """One image-expression-target record in the feasibility benchmark."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    image_path: str = Field(min_length=1)
    query: str = Field(min_length=1)
    grounding_prompt: str = Field(min_length=1)
    dimensions: list[ReferringQueryDimension] = Field(min_length=1)
    novelty: ReferringQueryNovelty = "seen"
    reference_frame: ReferringReferenceFrame | None = None
    annotation_status: ReferringAnnotationStatus
    expected_count: int | None = Field(default=None, ge=0)
    targets: list[ReferringSmokeTarget] = Field(default_factory=list)
    contrast_set_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    notes: str | None = None

    @field_validator("image_path", "query", "grounding_prompt")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        """Reject whitespace-only fields and store stable compact text."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Referring smoke text fields cannot be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_annotation_contract(self) -> "ReferringSmokeCase":
        """Keep scored, negative, and unlabelled cases unambiguous."""
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("Referring query dimensions cannot contain duplicates.")
        if self.annotation_status == "mask":
            if not self.targets or any(
                target.segmentation is None for target in self.targets
            ):
                raise ValueError("Mask-labelled cases require masks for all targets.")
        elif self.annotation_status == "box":
            if not self.targets or any(target.box is None for target in self.targets):
                raise ValueError("Box-labelled cases require boxes for all targets.")
        elif self.annotation_status == "negative":
            if self.targets:
                raise ValueError("Negative cases cannot contain targets.")
            if self.expected_count not in (None, 0):
                raise ValueError("Negative cases must expect zero regions.")
            self.expected_count = 0
        else:
            if self.targets:
                raise ValueError("Unlabelled cases cannot contain scored targets.")
            if self.expected_count is not None:
                raise ValueError("Unlabelled cases cannot define expected_count.")

        if self.targets and self.expected_count is None:
            self.expected_count = len(self.targets)
        if self.targets and self.expected_count != len(self.targets):
            raise ValueError("expected_count must equal the number of targets.")
        if "spatial" in self.dimensions and self.reference_frame is None:
            raise ValueError("Spatial cases require an explicit reference_frame.")
        return self


class ReferringSmokeManifest(BaseModel):
    """Versioned collection of language-guided localization cases."""

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str | None = None
    cases: list[ReferringSmokeCase] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Reject whitespace-only benchmark names."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("Referring smoke manifest name cannot be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> "ReferringSmokeManifest":
        """Require stable unique IDs for saved per-case responses."""
        case_ids = [case.id for case in self.cases]
        duplicates = sorted(
            case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate referring smoke case IDs: {duplicates}")
        return self


def load_referring_smoke_manifest(path: Path) -> ReferringSmokeManifest:
    """Read and validate one UTF-8 referring-expression manifest."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return ReferringSmokeManifest.model_validate(payload)
