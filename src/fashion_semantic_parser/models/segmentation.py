"""Typed schemas for garment instance segmentation outputs."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

SubjectROISource = Literal["manual", "detected", "full_image_fallback"]


class SegmentationBoundingBox(BaseModel):
    """Axis-aligned bounding box in xyxy image pixel coordinates."""

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(ge=0.0)
    y_max: float = Field(ge=0.0)


class SegmentationInstance(BaseModel):
    """One predicted garment instance."""

    category_id: int = Field(ge=1)
    category_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: SegmentationBoundingBox
    mask: list[list[float]] = Field(default_factory=list)


class SegmentationSubjectROI(BaseModel):
    """Subject/person region of interest in xyxy image pixel coordinates."""

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(ge=0.0)
    y_max: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_coordinate_order(self) -> "SegmentationSubjectROI":
        """Require a non-empty ROI with ordered xyxy coordinates."""
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("subject ROI max coordinates must exceed min coordinates")
        return self


class SegmentationPrediction(BaseModel):
    """Instance segmentation result for one RGB product image."""

    image_path: str
    instances: list[SegmentationInstance] = Field(default_factory=list)
    subject_roi: SegmentationSubjectROI | None = None
    subject_roi_source: SubjectROISource | None = None


class SegmentationRequest(BaseModel):
    """API request for garment instance segmentation."""

    image_path: str = Field(min_length=1)
    subject_roi: SegmentationSubjectROI | None = None
    auto_subject_roi: bool = False

    @model_validator(mode="after")
    def validate_roi_mode(self) -> "SegmentationRequest":
        """Keep manual and automatic ROI selection mutually exclusive."""
        if self.subject_roi is not None and self.auto_subject_roi:
            raise ValueError("subject_roi and auto_subject_roi cannot be used together")
        return self
