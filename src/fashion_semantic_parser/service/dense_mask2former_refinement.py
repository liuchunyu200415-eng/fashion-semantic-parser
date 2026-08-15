"""Refine full-query DINOv2 boxes with supervised Fashionpedia part Masks."""

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from pydantic import BaseModel, Field

from fashion_semantic_parser.models.localization import (
    LocalizedRegion,
    RegionLocalizationPrediction,
)
from fashion_semantic_parser.models.segmentation import SegmentationSubjectROI
from fashion_semantic_parser.service.dense_region_localization import box_iou
from fashion_semantic_parser.service.region_localization import (
    Mask2FormerPartLocalizationService,
    RegionLocalizationRuntime,
)


class DenseMask2FormerRefinementSettings(BaseModel):
    """Validated geometric gate for supervised Mask replacement."""

    minimum_box_iou: float = Field(default=0.05, ge=0.0, le=1.0)


@dataclass(frozen=True)
class Mask2FormerRefinementMatch:
    """One query-compatible Mask2Former candidate and its DINO Box overlap."""

    region: LocalizedRegion
    box_iou: float


class DenseMask2FormerRefinementRegionLocalizationService:
    """Keep full-query DINOv2 selection and replace only its known-part Mask."""

    supports_open_queries = True
    requires_full_image = True

    def __init__(
        self,
        dense_service: RegionLocalizationRuntime,
        part_service: Mask2FormerPartLocalizationService,
        *,
        settings: DenseMask2FormerRefinementSettings | None = None,
    ) -> None:
        """Compose open-query selection with domain-adapted Mask refinement."""
        self.dense_service = dense_service
        self.part_service = part_service
        self.settings = settings or DenseMask2FormerRefinementSettings()

    def accepts_query(self, query: str) -> bool:
        """Accept the same open local-region query scope as the dense service."""
        accepts_query = getattr(self.dense_service, "accepts_query", None)
        return (
            bool(accepts_query(query))
            if accepts_query is not None
            else bool(query.strip())
        )

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Return the dense result or its geometrically matched supervised Mask."""
        dense_prediction = self.dense_service.localize(
            image_path,
            query,
            subject_roi=subject_roi,
            auto_subject_roi=auto_subject_roi,
        )
        if not dense_prediction.regions or not self.part_service.supports_query(query):
            return dense_prediction
        part_prediction = self.part_service.localize(
            image_path,
            query,
            subject_roi=subject_roi,
            auto_subject_roi=auto_subject_roi,
        )
        return refine_dense_prediction_with_mask2former(
            dense_prediction,
            part_prediction,
            minimum_box_iou=self.settings.minimum_box_iou,
        )


def select_mask2former_refinement(
    dense_region: LocalizedRegion,
    candidates: list[LocalizedRegion],
    *,
    minimum_box_iou: float,
) -> Mask2FormerRefinementMatch | None:
    """Select the query-compatible part Mask best overlapping the DINOv2 Box."""
    matches = select_mask2former_refinements(
        dense_region,
        candidates,
        minimum_box_iou=minimum_box_iou,
    )
    return matches[0] if matches else None


def select_mask2former_refinements(
    dense_region: LocalizedRegion,
    candidates: list[LocalizedRegion],
    *,
    minimum_box_iou: float,
) -> list[Mask2FormerRefinementMatch]:
    """Return every query-compatible part Mask overlapping the DINOv2 Box."""
    if not 0.0 <= minimum_box_iou <= 1.0:
        raise ValueError("minimum_box_iou must be in [0, 1].")
    matches = [
        Mask2FormerRefinementMatch(
            region=candidate,
            box_iou=box_iou(
                _box_tuple(dense_region),
                _box_tuple(candidate),
            ),
        )
        for candidate in candidates
        if candidate.mask
    ]
    return sorted(
        (match for match in matches if match.box_iou >= minimum_box_iou),
        key=lambda match: (match.box_iou, match.region.confidence),
        reverse=True,
    )


def refine_dense_prediction_with_mask2former(
    dense_prediction: RegionLocalizationPrediction,
    part_prediction: RegionLocalizationPrediction,
    *,
    minimum_box_iou: float,
) -> RegionLocalizationPrediction:
    """Replace only the top-1 Mask while preserving query and DINOv2 Box."""
    if not dense_prediction.regions:
        return dense_prediction
    dense_region = dense_prediction.regions[0]
    matches = select_mask2former_refinements(
        dense_region,
        part_prediction.regions,
        minimum_box_iou=minimum_box_iou,
    )
    if not matches:
        return dense_prediction
    refined_region = dense_region.model_copy(
        update={
            "mask": [polygon for match in matches for polygon in match.region.mask],
            "mask_source": "mask2former_box_guided",
            "box_source": "dense_coarse_localization",
        }
    )
    return dense_prediction.model_copy(update={"regions": [refined_region]})


def localization_polygons_to_mask(
    polygons: list[list[float]],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize API polygons without using target annotations."""
    height, width = image_shape
    if height < 1 or width < 1:
        raise ValueError("Mask image dimensions must be positive.")
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        coordinates = np.asarray(polygon, dtype=np.float64)
        if (
            coordinates.ndim != 1
            or len(coordinates) < 6
            or len(coordinates) % 2
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError("Localization polygons must contain finite xy pairs.")
        points = np.rint(coordinates.reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [points], (1,))
    return cast(np.ndarray, np.asarray(mask, dtype=bool))


def _box_tuple(region: LocalizedRegion) -> tuple[float, float, float, float]:
    """Return one typed localization Box as an xyxy tuple."""
    return (
        region.box.x_min,
        region.box.y_min,
        region.box.x_max,
        region.box.y_max,
    )
