"""Production coarse-to-fine localization for complete language queries."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Any, cast

import cv2
import numpy as np

from fashion_semantic_parser.common.exceptions import (
    InvalidImageInputError,
    ModelNotReadyError,
)
from fashion_semantic_parser.common.paths import resolve_project_path
from fashion_semantic_parser.models.localization import (
    LocalizationBoundingBox,
    LocalizedRegion,
    RegionLocalizationPrediction,
)
from fashion_semantic_parser.models.segmentation import (
    SegmentationSubjectROI,
    SubjectROISource,
)
from fashion_semantic_parser.service.dense_crop_audit import (
    extract_crop_image,
    fuse_crop_score_maps,
    restore_crop_score_map,
    select_query_peak_crops,
)
from fashion_semantic_parser.service.dense_local_runtime import (
    DenseLocalReencodingSettings,
    DenseLocalRuntimeBundle,
    build_dense_local_runtime,
    load_dense_local_reencoding_settings,
)
from fashion_semantic_parser.service.dense_region_localization import mask_box
from fashion_semantic_parser.service.dinov2_region_encoder import patch_scores_to_image
from fashion_semantic_parser.service.subject_roi import Detectron2PersonROIDetector


@dataclass(frozen=True)
class DenseLocalMaskResult:
    """Query-conditioned local Mask with an independent coarse Box."""

    local_mask: np.ndarray
    coarse_box: tuple[float, float, float, float] | None
    confidence: float


class DenseLocalReencodingEngine:
    """Reuse frozen PRD encoders for category-free local re-encoding."""

    def __init__(
        self,
        settings: DenseLocalReencodingSettings,
        runtime: DenseLocalRuntimeBundle | None = None,
    ) -> None:
        self.settings = settings
        self._runtime = runtime
        self._runtime_lock = Lock()
        self._inference_lock = Lock()

    def predict(self, image_rgb: np.ndarray, query: str) -> DenseLocalMaskResult:
        """Predict a local-only Mask while retaining the coarse Box.

        Args:
            image_rgb: Complete or subject-cropped uint8 RGB image.
            query: Unmodified complete language expression.

        Returns:
            Local Mask, independent coarse Box, and foreground confidence.

        Raises:
            ValueError: If query, image, or model output geometry is invalid.
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Complete localization query cannot be empty.")
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Dense localization expects an HxWx3 uint8 RGB image.")
        runtime = self._get_runtime()
        with self._inference_lock:
            projected_query = runtime.projector.project(query)
            coarse_dense = runtime.image_encoder.encode_dense(image)
            coarse_probabilities = runtime.scorer.score(
                coarse_dense.features,
                projected_query,
            )
            coarse_scores = patch_scores_to_image(
                coarse_probabilities,
                coarse_dense.geometry,
            )
            crops = select_query_peak_crops(
                coarse_probabilities,
                coarse_dense.geometry,
                crop_fraction=self.settings.crop_fraction,
                max_crops=self.settings.max_crops,
            )
            restored_maps = []
            for crop in crops:
                crop_image = extract_crop_image(image, crop)
                crop_dense = runtime.image_encoder.encode_dense(crop_image)
                crop_probabilities = runtime.scorer.score(
                    crop_dense.features,
                    projected_query,
                )
                crop_scores = patch_scores_to_image(
                    crop_probabilities,
                    crop_dense.geometry,
                )
                restored_maps.append(
                    restore_crop_score_map(
                        crop_scores,
                        crop,
                        image.shape[:2],
                    )
                )
        local_scores = fuse_crop_score_maps(restored_maps)
        local_mask = np.asarray(
            local_scores >= runtime.scorer.threshold,
            dtype=bool,
        )
        if int(local_mask.sum()) < self.settings.min_mask_area:
            local_mask = np.zeros_like(local_mask, dtype=bool)
        coarse_mask = np.asarray(
            coarse_scores >= runtime.scorer.threshold,
            dtype=bool,
        )
        confidence = (
            float(np.mean(local_scores[local_mask])) if local_mask.any() else 0.0
        )
        return DenseLocalMaskResult(
            local_mask=local_mask,
            coarse_box=mask_box(coarse_mask) or mask_box(local_mask),
            confidence=max(0.0, min(1.0, confidence)),
        )

    def _get_runtime(self) -> DenseLocalRuntimeBundle:
        """Build and retain the heavy production runtime on first use."""
        if self._runtime is not None:
            return self._runtime
        with self._runtime_lock:
            if self._runtime is None:
                self._runtime = build_dense_local_runtime(self.settings)
        return self._runtime


class DenseLocalReencodingRegionLocalizationService:
    """Expose the frozen open-query local re-encoding path through the API."""

    supports_open_queries = True
    requires_full_image = True

    _GENERAL_GARMENT_QUESTIONS = (
        "图中有哪些服饰",
        "图中有什么服饰",
        "有哪些衣服",
        "这是什么服饰",
        "这是什么衣服",
        "what clothes are in",
        "what garments are in",
        "what is this garment",
    )

    def __init__(
        self,
        config_path: str = "configs/localization_dense_local_reencoding.yaml",
        *,
        engine: DenseLocalReencodingEngine | None = None,
        subject_roi_detector: Any | None = None,
        settings: DenseLocalReencodingSettings | None = None,
    ) -> None:
        self.config_path = config_path
        self._settings = settings
        self._engine = engine
        self._subject_roi_detector = subject_roi_detector
        self._dependency_lock = RLock()

    def accepts_query(self, query: str) -> bool:
        """Reject only explicit whole-image inventory or classification.

        Args:
            query: Complete user query.

        Returns:
            Whether the query belongs to open local-region localization.
        """
        normalized = " ".join(query.strip().casefold().split())
        if not normalized:
            return False
        return not any(
            phrase in normalized for phrase in self._GENERAL_GARMENT_QUESTIONS
        )

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Localize an arbitrary complete query without fixed-part mapping.

        Args:
            image_path: Project-relative source image path.
            query: Complete language expression.
            subject_roi: Optional manually supplied subject rectangle.
            auto_subject_roi: Whether to detect the primary subject first.

        Returns:
            API-ready local Mask and independent coarse Box.

        Raises:
            InvalidImageInputError: If input or requested geometry is invalid.
            ModelNotReadyError: If pinned runtime assets cannot be loaded.
        """
        if subject_roi is not None and auto_subject_roi:
            raise InvalidImageInputError(
                "subject_roi and auto_subject_roi cannot be used together"
            )
        resolved_path = _resolve_image_path(image_path)
        image_bgr = cv2.imread(str(resolved_path))
        if image_bgr is None:
            raise InvalidImageInputError(f"Unable to read image: {image_path}")
        effective_roi = subject_roi
        roi_source: SubjectROISource | None = "manual" if subject_roi else None
        try:
            if auto_subject_roi and not self.requires_full_image:
                effective_roi = self._get_subject_roi_detector().detect(resolved_path)
                roi_source = "detected" if effective_roi else "full_image_fallback"
            elif auto_subject_roi:
                roi_source = "full_image_fallback"
            crop_bgr, offset = _crop_subject_roi(
                image_bgr,
                effective_roi,
                margin=self._get_settings().subject_roi_margin,
            )
            result = self._get_engine().predict(
                cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                query,
            )
        except InvalidImageInputError:
            raise
        except (FileNotFoundError, OSError, RuntimeError, TypeError) as error:
            raise ModelNotReadyError(
                f"Dense local re-encoding runtime is unavailable: {error}"
            ) from error
        except ValueError as error:
            raise InvalidImageInputError(str(error)) from error
        regions = _result_to_regions(result, query=query, offset=offset)
        return RegionLocalizationPrediction(
            image_path=image_path,
            query=query,
            regions=regions,
            subject_roi=effective_roi,
            subject_roi_source=roi_source,
        )

    def _get_settings(self) -> DenseLocalReencodingSettings:
        """Load and cache validated deployment settings."""
        if self._settings is None:
            with self._dependency_lock:
                if self._settings is None:
                    self._settings = load_dense_local_reencoding_settings(
                        self.config_path
                    )
        return self._settings

    def _get_engine(self) -> DenseLocalReencodingEngine:
        """Build and cache the heavy engine."""
        if self._engine is None:
            with self._dependency_lock:
                if self._engine is None:
                    self._engine = DenseLocalReencodingEngine(self._get_settings())
        return self._engine

    def _get_subject_roi_detector(self) -> Any:
        """Build and cache the configured primary-person detector."""
        if self._subject_roi_detector is None:
            with self._dependency_lock:
                if self._subject_roi_detector is None:
                    self._subject_roi_detector = Detectron2PersonROIDetector(
                        self._get_settings().person_detector
                    )
        return self._subject_roi_detector


def _resolve_image_path(image_path: str) -> Path:
    """Resolve and validate one local API image path."""
    try:
        path = resolve_project_path(image_path)
    except ValueError as error:
        raise InvalidImageInputError(str(error)) from error
    if not path.is_file():
        raise InvalidImageInputError(f"Input image not found: {image_path}")
    return cast(Path, path)


def _crop_subject_roi(
    image: np.ndarray,
    subject_roi: SegmentationSubjectROI | None,
    *,
    margin: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Crop one optional ROI and retain its source-coordinate offset."""
    if subject_roi is None:
        return image, (0.0, 0.0)
    height, width = image.shape[:2]
    roi_width = subject_roi.x_max - subject_roi.x_min
    roi_height = subject_roi.y_max - subject_roi.y_min
    x_min = max(0, int(np.floor(subject_roi.x_min - roi_width * margin)))
    y_min = max(0, int(np.floor(subject_roi.y_min - roi_height * margin)))
    x_max = min(width, int(np.ceil(subject_roi.x_max + roi_width * margin)))
    y_max = min(height, int(np.ceil(subject_roi.y_max + roi_height * margin)))
    if x_max <= x_min or y_max <= y_min:
        raise InvalidImageInputError("Subject ROI does not overlap the input image.")
    return image[y_min:y_max, x_min:x_max], (float(x_min), float(y_min))


def _result_to_regions(
    result: DenseLocalMaskResult,
    *,
    query: str,
    offset: tuple[float, float],
) -> list[LocalizedRegion]:
    """Convert the local Mask and independent coarse Box to API polygons."""
    if not result.local_mask.any() or result.coarse_box is None:
        return []
    polygons = _mask_to_polygons(result.local_mask, offset=offset)
    if not polygons:
        return []
    x_offset, y_offset = offset
    return [
        LocalizedRegion(
            region_label="open_query_region",
            matched_text=query,
            confidence=result.confidence,
            box=LocalizationBoundingBox(
                x_min=result.coarse_box[0] + x_offset,
                y_min=result.coarse_box[1] + y_offset,
                x_max=result.coarse_box[2] + x_offset,
                y_max=result.coarse_box[3] + y_offset,
            ),
            mask=polygons,
            mask_source="dense_local_reencoding",
            box_source="dense_coarse_localization",
        )
    ]


def _mask_to_polygons(
    mask: np.ndarray,
    *,
    offset: tuple[float, float],
) -> list[list[float]]:
    """Convert all local connected components to source-coordinate polygons."""
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    x_offset, y_offset = offset
    polygons = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        polygons.append(
            [
                coordinate
                for x_value, y_value in points
                for coordinate in (
                    float(x_value) + x_offset,
                    float(y_value) + y_offset,
                )
            ]
        )
    return polygons
