"""Typed schemas for normalized fashion dataset samples."""

from typing import Any

from pydantic import BaseModel, Field


class FashionItemAnnotation(BaseModel):
    """Normalized annotation for one garment item in an image."""

    item_id: str
    category_name: str | None = None
    category_id: int | None = None
    style: int | None = None
    bounding_box: list[int] = Field(default_factory=list)
    raw_attributes: dict[str, Any] = Field(default_factory=dict)


class FashionSample(BaseModel):
    """Normalized dataset sample used by training and analysis pipelines."""

    dataset_name: str
    split: str
    image_path: str
    annotation_path: str | None = None
    items: list[FashionItemAnnotation] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
