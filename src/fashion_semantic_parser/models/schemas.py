"""Typed schemas for image parsing and multimodal question answering."""

from pydantic import BaseModel, Field

from fashion_semantic_parser.models.segmentation import SegmentationPrediction


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in image pixel coordinates."""

    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=0)
    y_max: int = Field(ge=0)


class AttributePrediction(BaseModel):
    """Fine-grained fashion attribute prediction."""

    name: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class RegionPrediction(BaseModel):
    """Localized fashion region prediction."""

    label: str
    box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    attributes: list[AttributePrediction] = Field(default_factory=list)


class MultimodalQueryRequest(BaseModel):
    """Request body for image and natural language driven parsing."""

    image_path: str
    query: str


class MultimodalQueryResponse(BaseModel):
    """Response body for multimodal parsing and answering."""

    answer: str
    regions: list[RegionPrediction] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    segmentation: SegmentationPrediction | None = None
