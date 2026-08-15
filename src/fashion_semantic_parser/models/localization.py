"""Typed schemas for PRD 3.1.2 language-guided region localization."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from fashion_semantic_parser.models.segmentation import (
    SegmentationSubjectROI,
    SubjectROISource,
)


class LocalizationBoundingBox(BaseModel):
    """Axis-aligned localized-region box in xyxy image pixel coordinates."""

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(ge=0.0)
    y_max: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_coordinate_order(self) -> "LocalizationBoundingBox":
        """Require a positive-area localization box."""
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(
                "localized region max coordinates must exceed min coordinates"
            )
        return self


class LocalizedRegion(BaseModel):
    """One text-matched fashion part with a refined segmentation mask."""

    region_label: str
    matched_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: LocalizationBoundingBox
    mask: list[list[float]] = Field(default_factory=list)
    mask_source: (
        Literal[
            "dense_local_reencoding",
            "mask2former_box_guided",
        ]
        | None
    ) = None
    box_source: Literal["dense_coarse_localization"] | None = None


class RegionLocalizationPrediction(BaseModel):
    """Language-guided local-region result for one RGB product image."""

    image_path: str
    query: str
    regions: list[LocalizedRegion] = Field(default_factory=list)
    subject_roi: SegmentationSubjectROI | None = None
    subject_roi_source: SubjectROISource | None = None


class RegionLocalizationRequest(BaseModel):
    """API request for language-guided fashion-part localization."""

    image_path: str = Field(min_length=1)
    query: str = Field(min_length=1)
    subject_roi: SegmentationSubjectROI | None = None
    auto_subject_roi: bool | None = None

    @model_validator(mode="after")
    def validate_roi_mode(self) -> "RegionLocalizationRequest":
        """Keep manual and automatic ROI selection mutually exclusive."""
        if self.subject_roi is not None and self.auto_subject_roi is True:
            raise ValueError("subject_roi and auto_subject_roi cannot be used together")
        return self
