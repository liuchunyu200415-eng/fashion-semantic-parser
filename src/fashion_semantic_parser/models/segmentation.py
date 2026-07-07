"""Typed schemas for garment instance segmentation outputs."""

from pydantic import BaseModel, Field


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


class SegmentationPrediction(BaseModel):
    """Instance segmentation result for one RGB product image."""

    image_path: str
    instances: list[SegmentationInstance] = Field(default_factory=list)
