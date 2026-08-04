"""Runtime service for PRD 3.1.2 language-guided region localization."""

import math
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping, Protocol

import cv2
import numpy as np

from fashion_semantic_parser.common.exceptions import (
    ConfigurationError,
    InvalidImageInputError,
    ModelNotReadyError,
)
from fashion_semantic_parser.common.paths import resolve_project_path
from fashion_semantic_parser.dao.localization.taxonomy import (
    FASHIONPEDIA_PART_CATEGORIES,
    LocalizationPrompt,
    resolve_localization_prompt,
)
from fashion_semantic_parser.models.localization import (
    LocalizationBoundingBox,
    LocalizedRegion,
    RegionLocalizationPrediction,
)
from fashion_semantic_parser.models.segmentation import (
    SegmentationInstance,
    SegmentationPrediction,
    SegmentationSubjectROI,
    SubjectROISource,
)
from fashion_semantic_parser.service.grounded_sam_hq import (
    GroundedMaskCandidate,
    GroundedMaskPredictor,
    GroundedSAMHQPredictor,
    GroundedSAMHQSettings,
    load_grounded_sam_hq_settings,
    validate_grounded_sam_hq_assets,
)
from fashion_semantic_parser.service.segmentation_runtime import (
    GarmentSegmentationService,
    SegmentationRuntime,
)
from fashion_semantic_parser.service.subject_roi import (
    Detectron2PersonROIDetector,
    PersonROIDetectorSettings,
)


class RegionLocalizationRuntime(Protocol):
    """Minimal interface implemented by language-guided localization runtimes."""

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Localize the image region described by natural language."""
        ...


class SubjectROIDetector(Protocol):
    """Person ROI dependency used before local-part grounding."""

    def detect(self, image_path: Path) -> SegmentationSubjectROI | None:
        """Return the primary subject ROI or None."""
        ...


PredictorFactory = Callable[[GroundedSAMHQSettings], GroundedMaskPredictor]
SubjectROIDetectorFactory = Callable[
    [GroundedSAMHQSettings],
    SubjectROIDetector,
]


class GroundedSAMHQRegionLocalizationService:
    """Localize a text-described fashion part with person ROI and Grounded SAM."""

    def __init__(
        self,
        config_path: str = "configs/localization_grounded_sam_hq.yaml",
        *,
        predictor: GroundedMaskPredictor | None = None,
        predictor_factory: PredictorFactory = GroundedSAMHQPredictor,
        subject_roi_detector: SubjectROIDetector | None = None,
        subject_roi_detector_factory: SubjectROIDetectorFactory | None = None,
        settings: GroundedSAMHQSettings | None = None,
        settings_overrides: Mapping[str, Any] | None = None,
        grounding_prompt_override: str | None = None,
    ) -> None:
        """Create a service that loads both foundation models on first use."""
        if grounding_prompt_override is not None:
            grounding_prompt_override = " ".join(
                grounding_prompt_override.strip().split()
            )
            if not grounding_prompt_override:
                raise ValueError("Grounding prompt override cannot be empty.")
        self.config_path = config_path
        self._predictor = predictor
        self._predictor_factory = predictor_factory
        self._predictor_lock = Lock()
        self._subject_roi_detector = subject_roi_detector
        self._subject_roi_detector_factory = (
            subject_roi_detector_factory or _build_default_subject_roi_detector
        )
        self._subject_roi_detector_lock = Lock()
        self._settings = settings
        self._settings_lock = Lock()
        self._settings_overrides = dict(settings_overrides or {})
        self._grounding_prompt_override = grounding_prompt_override

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Return query-aligned masks and mask-derived boxes for one image."""
        if subject_roi is not None and auto_subject_roi:
            raise InvalidImageInputError(
                "subject_roi and auto_subject_roi cannot be used together"
            )
        resolved_image_path = self._resolve_image_path(image_path)
        image = cv2.imread(str(resolved_image_path))
        if image is None:
            raise InvalidImageInputError(f"Unable to read image: {image_path}")

        settings = self._get_settings()
        if (
            self._predictor is None
            and self._predictor_factory is GroundedSAMHQPredictor
        ):
            validate_grounded_sam_hq_assets(settings)

        effective_roi = subject_roi
        roi_source: SubjectROISource | None = (
            "manual" if subject_roi is not None else None
        )
        try:
            if auto_subject_roi:
                effective_roi = self._get_subject_roi_detector().detect(
                    resolved_image_path
                )
                roi_source = (
                    "detected" if effective_roi is not None else "full_image_fallback"
                )
            crop, coordinate_offset = _crop_to_subject_roi(
                image,
                effective_roi,
                margin=settings.subject_roi_margin,
            )
            prompt = resolve_localization_prompt(query)
            if self._grounding_prompt_override is not None:
                prompt = prompt.model_copy(
                    update={
                        "grounding_prompt": self._grounding_prompt_override,
                    }
                )
            candidates = self._get_predictor().predict(
                crop,
                prompt.grounding_prompt,
            )
        except (ConfigurationError, ModelNotReadyError, InvalidImageInputError):
            raise
        except OSError as error:
            raise ModelNotReadyError(
                f"Localization runtime asset could not be loaded: {error}"
            ) from error
        except ValueError as error:
            raise InvalidImageInputError(str(error)) from error
        except (AttributeError, RuntimeError, TypeError) as error:
            raise ModelNotReadyError(
                f"Grounding DINO + SAM-HQ inference failed: {error}"
            ) from error

        regions = [
            region
            for candidate in candidates
            if (
                region := _candidate_to_localized_region(
                    candidate,
                    prompt=prompt,
                    coordinate_offset=coordinate_offset,
                    expected_shape=(int(crop.shape[0]), int(crop.shape[1])),
                    min_mask_area=settings.min_mask_area,
                )
            )
            is not None
        ]
        return RegionLocalizationPrediction(
            image_path=image_path,
            query=query,
            regions=regions,
            subject_roi=effective_roi,
            subject_roi_source=roi_source,
        )

    def _get_settings(self) -> GroundedSAMHQSettings:
        """Load the deployment YAML once."""
        if self._settings is not None:
            return self._settings
        with self._settings_lock:
            if self._settings is None:
                self._settings = load_grounded_sam_hq_settings(
                    self.config_path,
                    overrides=self._settings_overrides,
                )
        return self._settings

    def _get_predictor(self) -> GroundedMaskPredictor:
        """Initialize and reuse the Grounding DINO + SAM-HQ bundle."""
        if self._predictor is not None:
            return self._predictor
        with self._predictor_lock:
            if self._predictor is None:
                self._predictor = self._predictor_factory(self._get_settings())
        return self._predictor

    def _get_subject_roi_detector(self) -> SubjectROIDetector:
        """Initialize and reuse the accepted primary-person detector."""
        if self._subject_roi_detector is not None:
            return self._subject_roi_detector
        with self._subject_roi_detector_lock:
            if self._subject_roi_detector is None:
                self._subject_roi_detector = self._subject_roi_detector_factory(
                    self._get_settings()
                )
        return self._subject_roi_detector

    @staticmethod
    def _resolve_image_path(image_path: str) -> Path:
        """Validate that the API image path stays inside the project."""
        try:
            resolved_path = resolve_project_path(image_path)
        except ValueError as error:
            raise InvalidImageInputError(str(error)) from error
        if not resolved_path.is_file():
            raise InvalidImageInputError(f"Input image not found: {image_path}")
        return resolved_path


_SUPERVISED_PART_LABELS = frozenset(
    category.english_name for category in FASHIONPEDIA_PART_CATEGORIES
)
_SUPERVISED_DECORATION_LABELS = frozenset(
    category.english_name
    for category in FASHIONPEDIA_PART_CATEGORIES
    if category.region_group == "decoration"
)
_HEM_GARMENT_LABELS = frozenset(("top", "skirt", "outerwear", "dress"))
_HEM_GARMENT_QUERY_TERMS = (
    ("dress", ("连衣裙", "dress")),
    ("outerwear", ("外套", "大衣", "夹克", "coat", "jacket", "outerwear")),
    ("skirt", ("半身裙", "裙子", "skirt")),
    ("top", ("上衣", "衬衫", "毛衣", "shirt", "sweater", "t-shirt", "top")),
)
_WAIST_GARMENT_LABELS = frozenset(("top", "pants", "skirt", "outerwear", "dress"))
_WAIST_GARMENT_QUERY_TERMS = _HEM_GARMENT_QUERY_TERMS + (
    ("pants", ("裤子", "长裤", "短裤", "pants", "trousers", "shorts")),
)
_PATTERN_GARMENT_LABELS = _WAIST_GARMENT_LABELS
_PATTERN_GARMENT_QUERY_TERMS = _WAIST_GARMENT_QUERY_TERMS


class Mask2FormerPartLocalizationService:
    """Use the supervised 19-class Mask2Former for known Fashionpedia parts."""

    def __init__(
        self,
        config_path: str = "configs/localization_mask2former_parts_deployment.yaml",
        *,
        segmentation_service: SegmentationRuntime | None = None,
    ) -> None:
        """Create a lazy supervised part-localization runtime."""
        self.config_path = config_path
        self.segmentation_service = segmentation_service or GarmentSegmentationService(
            config_path
        )

    def supports_query(self, query: str) -> bool:
        """Return whether the query maps to directly supervised part labels."""
        prompt = _resolve_prompt_or_error(query)
        return bool(_supervised_labels_for_prompt(prompt))

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Predict and retain only classes aligned with the natural-language query."""
        prompt = _resolve_prompt_or_error(query)
        target_labels = _supervised_labels_for_prompt(prompt)
        if not target_labels:
            raise ModelNotReadyError(
                f"No directly supervised Mask2Former category covers query: {query}"
            )

        prediction = self.segmentation_service.segment(
            image_path,
            subject_roi=subject_roi,
            auto_subject_roi=auto_subject_roi,
        )
        regions = [
            LocalizedRegion(
                region_label=instance.category_label,
                matched_text=prompt.matched_term,
                confidence=instance.confidence,
                box=LocalizationBoundingBox(
                    x_min=instance.box.x_min,
                    y_min=instance.box.y_min,
                    x_max=instance.box.x_max,
                    y_max=instance.box.y_max,
                ),
                mask=instance.mask,
            )
            for instance in prediction.instances
            if instance.category_label in target_labels
        ]
        return RegionLocalizationPrediction(
            image_path=image_path,
            query=query,
            regions=regions,
            subject_roi=prediction.subject_roi,
            subject_roi_source=prediction.subject_roi_source,
        )


class HybridRegionLocalizationService:
    """Prefer supervised part masks and retain Grounded SAM for uncovered queries."""

    def __init__(
        self,
        supervised_service: Mask2FormerPartLocalizationService,
        fallback_service: RegionLocalizationRuntime,
        *,
        garment_segmentation_service: SegmentationRuntime | None = None,
    ) -> None:
        """Combine an exact-category runtime with an open-vocabulary fallback."""
        self.supervised_service = supervised_service
        self.fallback_service = fallback_service
        self.garment_segmentation_service = garment_segmentation_service

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Route directly supervised queries without loading both heavy paths."""
        return self._localize(
            image_path,
            query,
            subject_roi=subject_roi,
            auto_subject_roi=auto_subject_roi,
            garment_prediction=None,
        )

    def localize_with_garment_prediction(
        self,
        image_path: str,
        query: str,
        garment_prediction: SegmentationPrediction,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Reuse an existing 3.1.1 result for garment-derived regions."""
        return self._localize(
            image_path,
            query,
            subject_roi=subject_roi,
            auto_subject_roi=auto_subject_roi,
            garment_prediction=garment_prediction,
        )

    def _localize(
        self,
        image_path: str,
        query: str,
        *,
        subject_roi: SegmentationSubjectROI | None,
        auto_subject_roi: bool,
        garment_prediction: SegmentationPrediction | None,
    ) -> RegionLocalizationPrediction:
        """Route one query through supervised, derived, or fallback masks."""
        prompt = _resolve_prompt_or_error(query)
        if self.supervised_service.supports_query(query):
            return self.supervised_service.localize(
                image_path,
                query,
                subject_roi=subject_roi,
                auto_subject_roi=auto_subject_roi,
            )
        if prompt.region_label == "cuff":
            sleeve_prediction = self.supervised_service.localize(
                image_path,
                "sleeve",
                subject_roi=subject_roi,
                auto_subject_roi=auto_subject_roi,
            )
            cuff_regions = _derive_cuffs_from_sleeves(
                sleeve_prediction.regions,
                prompt=prompt,
                subject_roi=sleeve_prediction.subject_roi,
            )
            if cuff_regions:
                return RegionLocalizationPrediction(
                    image_path=image_path,
                    query=query,
                    regions=cuff_regions,
                    subject_roi=sleeve_prediction.subject_roi,
                    subject_roi_source=sleeve_prediction.subject_roi_source,
                )
        if prompt.region_label == "hem":
            effective_garment_prediction = garment_prediction
            if (
                effective_garment_prediction is None
                and self.garment_segmentation_service is not None
            ):
                effective_garment_prediction = (
                    self.garment_segmentation_service.segment(
                        image_path,
                        subject_roi=subject_roi,
                        auto_subject_roi=auto_subject_roi,
                    )
                )
            if effective_garment_prediction is not None:
                hem_regions = _derive_hems_from_garments(
                    effective_garment_prediction.instances,
                    prompt=prompt,
                    query=query,
                )
                if hem_regions:
                    return RegionLocalizationPrediction(
                        image_path=image_path,
                        query=query,
                        regions=hem_regions,
                        subject_roi=effective_garment_prediction.subject_roi,
                        subject_roi_source=(
                            effective_garment_prediction.subject_roi_source
                        ),
                    )
        if prompt.region_label == "waist":
            effective_garment_prediction = garment_prediction
            if (
                effective_garment_prediction is None
                and self.garment_segmentation_service is not None
            ):
                effective_garment_prediction = (
                    self.garment_segmentation_service.segment(
                        image_path,
                        subject_roi=subject_roi,
                        auto_subject_roi=auto_subject_roi,
                    )
                )
            if effective_garment_prediction is not None:
                waist_regions = _derive_waists_from_garments(
                    effective_garment_prediction.instances,
                    prompt=prompt,
                    query=query,
                )
                if waist_regions:
                    return RegionLocalizationPrediction(
                        image_path=image_path,
                        query=query,
                        regions=waist_regions,
                        subject_roi=effective_garment_prediction.subject_roi,
                        subject_roi_source=(
                            effective_garment_prediction.subject_roi_source
                        ),
                    )
        if prompt.region_label == "pattern":
            effective_garment_prediction = garment_prediction
            if (
                effective_garment_prediction is None
                and self.garment_segmentation_service is not None
            ):
                effective_garment_prediction = (
                    self.garment_segmentation_service.segment(
                        image_path,
                        subject_roi=subject_roi,
                        auto_subject_roi=auto_subject_roi,
                    )
                )
            if effective_garment_prediction is not None:
                pattern_regions = _derive_patterns_from_garments(
                    image_path,
                    effective_garment_prediction.instances,
                    prompt=prompt,
                    query=query,
                )
                if pattern_regions:
                    return RegionLocalizationPrediction(
                        image_path=image_path,
                        query=query,
                        regions=pattern_regions,
                        subject_roi=effective_garment_prediction.subject_roi,
                        subject_roi_source=(
                            effective_garment_prediction.subject_roi_source
                        ),
                    )
        return self.fallback_service.localize(
            image_path,
            query,
            subject_roi=subject_roi,
            auto_subject_roi=auto_subject_roi,
        )


class UnavailableRegionLocalizationService:
    """Explicit fallback for deployments that disable localization."""

    def localize(
        self,
        image_path: str,
        query: str,
        subject_roi: SegmentationSubjectROI | None = None,
        auto_subject_roi: bool = True,
    ) -> RegionLocalizationPrediction:
        """Reject inference instead of returning an invented localization."""
        raise ModelNotReadyError(
            "PRD 3.1.2 language-guided localization is not configured. "
            "Install a supervised or Grounding DINO + SAM-HQ runtime and weights "
            "before calling /v1/localize."
        )


def _resolve_prompt_or_error(query: str) -> LocalizationPrompt:
    """Resolve a query while keeping input errors in the service domain."""
    try:
        return resolve_localization_prompt(query)
    except ValueError as error:
        raise InvalidImageInputError(str(error)) from error


def _supervised_labels_for_prompt(
    prompt: LocalizationPrompt,
) -> frozenset[str]:
    """Map one normalized query to the labels with direct Mask supervision."""
    if prompt.region_label in _SUPERVISED_PART_LABELS:
        return frozenset((prompt.region_label,))
    if prompt.region_label == "shoulder":
        return frozenset(("epaulette",))
    if prompt.region_label == "decoration":
        return _SUPERVISED_DECORATION_LABELS
    return frozenset()


def _build_default_subject_roi_detector(
    settings: GroundedSAMHQSettings,
) -> SubjectROIDetector:
    """Build the same COCO-person ROI detector used by PRD 3.1.1."""
    return Detectron2PersonROIDetector(
        PersonROIDetectorSettings(
            device=settings.device,
            precision=settings.precision,
        )
    )


def _derive_cuffs_from_sleeves(
    sleeve_regions: list[LocalizedRegion],
    *,
    prompt: LocalizationPrompt,
    subject_roi: SegmentationSubjectROI | None,
    distal_fraction: float = 0.08,
    min_sleeve_confidence: float = 0.5,
    max_cuffs: int = 2,
) -> list[LocalizedRegion]:
    """Approximate each cuff from the distal end of a supervised sleeve mask."""
    if not 0.0 < distal_fraction < 0.5:
        raise ValueError("distal_fraction must be between 0 and 0.5")
    if not 0.0 <= min_sleeve_confidence <= 1.0:
        raise ValueError("min_sleeve_confidence must be between 0 and 1")
    if max_cuffs < 1:
        raise ValueError("max_cuffs must be positive")
    sleeve_regions = sorted(
        (
            region
            for region in sleeve_regions
            if region.confidence >= min_sleeve_confidence
        ),
        key=lambda region: region.confidence,
        reverse=True,
    )[:max_cuffs]
    if not sleeve_regions:
        return []

    sleeve_centers = np.asarray(
        [
            (
                (region.box.x_min + region.box.x_max) / 2.0,
                (region.box.y_min + region.box.y_max) / 2.0,
            )
            for region in sleeve_regions
        ],
        dtype=np.float64,
    )
    if subject_roi is not None:
        body_center = np.asarray(
            (
                (subject_roi.x_min + subject_roi.x_max) / 2.0,
                (subject_roi.y_min + subject_roi.y_max) / 2.0,
            ),
            dtype=np.float64,
        )
    else:
        body_center = sleeve_centers.mean(axis=0)

    cuffs = []
    for region in sleeve_regions:
        sleeve_mask, coordinate_offset = _localized_region_to_local_mask(region)
        if sleeve_mask is None:
            continue
        y_values, x_values = np.nonzero(sleeve_mask)
        global_points = np.column_stack(
            (
                x_values.astype(np.float64) + coordinate_offset[0],
                y_values.astype(np.float64) + coordinate_offset[1],
            )
        )
        point_center = global_points.mean(axis=0)
        centered = global_points - point_center
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        direction = axes[0]
        axis_projections = centered @ direction
        negative_end = point_center + direction * float(axis_projections.min())
        positive_end = point_center + direction * float(axis_projections.max())
        if subject_roi is None and len(sleeve_regions) == 1:
            if negative_end[1] > positive_end[1]:
                direction = -direction
        else:
            negative_distance = float(np.linalg.norm(negative_end - body_center))
            positive_distance = float(np.linalg.norm(positive_end - body_center))
            if negative_distance > positive_distance:
                direction = -direction
            elif (
                math.isclose(negative_distance, positive_distance) and direction[1] < 0
            ):
                direction = -direction
        projections = global_points @ direction
        threshold = float(np.quantile(projections, 1.0 - distal_fraction))
        distal_points = global_points[projections >= threshold]
        if len(distal_points) < 3:
            continue

        cuff_mask = np.zeros_like(sleeve_mask, dtype=bool)
        local_x = np.rint(distal_points[:, 0] - coordinate_offset[0]).astype(int)
        local_y = np.rint(distal_points[:, 1] - coordinate_offset[1]).astype(int)
        cuff_mask[local_y, local_x] = True
        cuff_mask = cv2.morphologyEx(
            cuff_mask.astype("uint8"),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype="uint8"),
        ).astype(bool)
        polygons = _mask_to_polygons(
            cuff_mask,
            coordinate_offset=coordinate_offset,
        )
        if not polygons:
            continue
        cuff_y, cuff_x = np.nonzero(cuff_mask)
        x_offset, y_offset = coordinate_offset
        cuffs.append(
            LocalizedRegion(
                region_label="cuff",
                matched_text=f"{prompt.matched_term} derived from sleeve",
                confidence=max(0.0, min(1.0, region.confidence * 0.9)),
                box=LocalizationBoundingBox(
                    x_min=float(cuff_x.min()) + x_offset,
                    y_min=float(cuff_y.min()) + y_offset,
                    x_max=float(cuff_x.max() + 1) + x_offset,
                    y_max=float(cuff_y.max() + 1) + y_offset,
                ),
                mask=polygons,
            )
        )
    return cuffs


def _localized_region_to_local_mask(
    region: LocalizedRegion,
) -> tuple[np.ndarray | None, tuple[float, float]]:
    """Rasterize one polygon region inside its integer-aligned local bounds."""
    x_min = int(math.floor(region.box.x_min))
    y_min = int(math.floor(region.box.y_min))
    x_max = int(math.ceil(region.box.x_max))
    y_max = int(math.ceil(region.box.y_max))
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None, (float(x_min), float(y_min))

    mask = np.zeros((height, width), dtype="uint8")
    polygons = []
    for polygon in region.mask:
        coordinates = np.asarray(polygon, dtype=np.float64)
        if coordinates.size < 6 or coordinates.size % 2:
            continue
        points = coordinates.reshape(-1, 2)
        points[:, 0] -= x_min
        points[:, 1] -= y_min
        polygons.append(np.rint(points).astype(np.int32))
    if not polygons:
        return None, (float(x_min), float(y_min))
    cv2.fillPoly(mask, polygons, 1)
    if not mask.any():
        return None, (float(x_min), float(y_min))
    return mask.astype(bool), (float(x_min), float(y_min))


def _derive_hems_from_garments(
    garment_instances: list[SegmentationInstance],
    *,
    prompt: LocalizationPrompt,
    query: str,
    min_garment_confidence: float = 0.5,
    max_hems: int = 2,
    band_fraction: float = 0.06,
) -> list[LocalizedRegion]:
    """Derive lower-edge bands from supported garment instance masks."""
    if not 0.0 <= min_garment_confidence <= 1.0:
        raise ValueError("min_garment_confidence must be between 0 and 1")
    if max_hems < 1:
        raise ValueError("max_hems must be positive")
    if not 0.0 < band_fraction < 0.5:
        raise ValueError("band_fraction must be between 0 and 0.5")

    requested_labels = _hem_garment_labels_for_query(query)
    eligible_instances = sorted(
        (
            instance
            for instance in garment_instances
            if instance.category_label in requested_labels
            and instance.confidence >= min_garment_confidence
        ),
        key=lambda instance: instance.confidence,
        reverse=True,
    )
    candidates = []
    for instance in eligible_instances:
        if any(
            _segmentation_box_iou(instance, retained) >= 0.7 for retained in candidates
        ):
            continue
        candidates.append(instance)
        if len(candidates) == max_hems:
            break
    hems = []
    for instance in candidates:
        garment_region = _segmentation_instance_to_localized_region(instance)
        garment_mask, coordinate_offset = _localized_region_to_local_mask(
            garment_region
        )
        if garment_mask is None:
            continue
        height, width = garment_mask.shape
        band_height = max(3, int(math.ceil(height * band_fraction)))
        x_min = int(math.floor(width * 0.20))
        x_max = int(math.ceil(width * 0.80))
        column_bottoms = {}
        for x_value in range(x_min, x_max):
            column_y = np.flatnonzero(garment_mask[:, x_value])
            if len(column_y):
                column_bottoms[x_value] = int(column_y.max())
        if not column_bottoms:
            continue
        target_bottom = float(np.median(list(column_bottoms.values())))
        bottom_tolerance = max(band_height * 2, int(math.ceil(height * 0.08)))
        hem_mask = np.zeros_like(garment_mask, dtype=bool)
        for x_value, column_bottom in column_bottoms.items():
            if abs(column_bottom - target_bottom) > bottom_tolerance:
                continue
            column_top = max(0, column_bottom - band_height + 1)
            hem_mask[column_top : column_bottom + 1, x_value] = garment_mask[
                column_top : column_bottom + 1,
                x_value,
            ]
        polygons = _mask_to_polygons(
            hem_mask,
            coordinate_offset=coordinate_offset,
        )
        if not polygons:
            continue
        y_values, x_values = np.nonzero(hem_mask)
        x_offset, y_offset = coordinate_offset
        hems.append(
            LocalizedRegion(
                region_label="hem",
                matched_text=(
                    f"{prompt.matched_term} derived from "
                    f"{instance.category_label} mask"
                ),
                confidence=max(0.0, min(1.0, instance.confidence * 0.9)),
                box=LocalizationBoundingBox(
                    x_min=float(x_values.min()) + x_offset,
                    y_min=float(y_values.min()) + y_offset,
                    x_max=float(x_values.max() + 1) + x_offset,
                    y_max=float(y_values.max() + 1) + y_offset,
                ),
                mask=polygons,
            )
        )
    return hems


def _hem_garment_labels_for_query(query: str) -> frozenset[str]:
    """Narrow a hem request when it explicitly names a parent garment."""
    normalized_query = query.casefold()
    for label, terms in _HEM_GARMENT_QUERY_TERMS:
        if any(term in normalized_query for term in terms):
            return frozenset((label,))
    return _HEM_GARMENT_LABELS


def _derive_waists_from_garments(
    garment_instances: list[SegmentationInstance],
    *,
    prompt: LocalizationPrompt,
    query: str,
    min_garment_confidence: float = 0.5,
    band_fraction: float = 0.06,
) -> list[LocalizedRegion]:
    """Approximate one waist band from a named or highest-score garment."""
    if not 0.0 <= min_garment_confidence <= 1.0:
        raise ValueError("min_garment_confidence must be between 0 and 1")
    if not 0.0 < band_fraction < 0.5:
        raise ValueError("band_fraction must be between 0 and 0.5")

    requested_labels = _waist_garment_labels_for_query(query)
    eligible_instances = sorted(
        (
            instance
            for instance in garment_instances
            if instance.category_label in requested_labels
            and instance.confidence >= min_garment_confidence
        ),
        key=lambda instance: instance.confidence,
        reverse=True,
    )
    for instance in eligible_instances:
        garment_region = _segmentation_instance_to_localized_region(instance)
        garment_mask, coordinate_offset = _localized_region_to_local_mask(
            garment_region
        )
        if garment_mask is None:
            continue
        height, width = garment_mask.shape
        band_height = max(3, int(math.ceil(height * band_fraction)))
        center_y = _waist_center_y(instance.category_label, height)
        y_min = max(0, int(round(center_y - band_height / 2)))
        y_max = min(height, y_min + band_height)
        x_min = int(math.floor(width * 0.20))
        x_max = int(math.ceil(width * 0.80))
        waist_mask = np.zeros_like(garment_mask, dtype=bool)
        waist_mask[y_min:y_max, x_min:x_max] = garment_mask[y_min:y_max, x_min:x_max]
        polygons = _mask_to_polygons(
            waist_mask,
            coordinate_offset=coordinate_offset,
        )
        if not polygons:
            continue
        y_values, x_values = np.nonzero(waist_mask)
        x_offset, y_offset = coordinate_offset
        return [
            LocalizedRegion(
                region_label="waist",
                matched_text=(
                    f"{prompt.matched_term} derived from "
                    f"{instance.category_label} mask"
                ),
                confidence=max(0.0, min(1.0, instance.confidence * 0.85)),
                box=LocalizationBoundingBox(
                    x_min=float(x_values.min()) + x_offset,
                    y_min=float(y_values.min()) + y_offset,
                    x_max=float(x_values.max() + 1) + x_offset,
                    y_max=float(y_values.max() + 1) + y_offset,
                ),
                mask=polygons,
            )
        ]
    return []


def _waist_garment_labels_for_query(query: str) -> frozenset[str]:
    """Narrow a waist request when it explicitly names a parent garment."""
    normalized_query = query.casefold()
    for label, terms in _WAIST_GARMENT_QUERY_TERMS:
        if any(term in normalized_query for term in terms):
            return frozenset((label,))
    return _WAIST_GARMENT_LABELS


def _waist_center_y(category_label: str, garment_height: int) -> float:
    """Choose an anatomy-informed relative waistline for one garment class."""
    relative_center = {
        "pants": 0.08,
        "skirt": 0.08,
        "dress": 0.45,
        "top": 0.68,
        "outerwear": 0.55,
    }[category_label]
    return garment_height * relative_center


def _derive_patterns_from_garments(
    image_path: str,
    garment_instances: list[SegmentationInstance],
    *,
    prompt: LocalizationPrompt,
    query: str,
    min_garment_confidence: float = 0.5,
) -> list[LocalizedRegion]:
    """Find salient internal color regions within one selected garment mask."""
    if not 0.0 <= min_garment_confidence <= 1.0:
        raise ValueError("min_garment_confidence must be between 0 and 1")
    try:
        resolved_image_path = resolve_project_path(image_path)
    except ValueError as error:
        raise InvalidImageInputError(str(error)) from error
    image = cv2.imread(str(resolved_image_path))
    if image is None:
        raise InvalidImageInputError(f"Unable to read image: {image_path}")

    requested_labels = _pattern_garment_labels_for_query(query)
    eligible_instances = sorted(
        (
            instance
            for instance in garment_instances
            if instance.category_label in requested_labels
            and instance.confidence >= min_garment_confidence
        ),
        key=lambda instance: instance.confidence,
        reverse=True,
    )
    for instance in eligible_instances:
        garment_region = _segmentation_instance_to_localized_region(instance)
        garment_mask, coordinate_offset = _localized_region_to_local_mask(
            garment_region
        )
        if garment_mask is None:
            continue
        crop, clipped_mask, clipped_offset = _clip_local_mask_to_image(
            image,
            garment_mask,
            coordinate_offset,
        )
        if crop is None or clipped_mask is None:
            continue
        pattern_mask = _salient_pattern_mask(crop, clipped_mask)
        if pattern_mask is None:
            continue
        polygons = _mask_to_polygons(
            pattern_mask,
            coordinate_offset=clipped_offset,
        )
        if not polygons:
            continue
        y_values, x_values = np.nonzero(pattern_mask)
        x_offset, y_offset = clipped_offset
        return [
            LocalizedRegion(
                region_label="pattern",
                matched_text=(
                    f"{prompt.matched_term} derived from "
                    f"{instance.category_label} appearance"
                ),
                confidence=max(0.0, min(1.0, instance.confidence * 0.75)),
                box=LocalizationBoundingBox(
                    x_min=float(x_values.min()) + x_offset,
                    y_min=float(y_values.min()) + y_offset,
                    x_max=float(x_values.max() + 1) + x_offset,
                    y_max=float(y_values.max() + 1) + y_offset,
                ),
                mask=polygons,
            )
        ]
    return []


def _pattern_garment_labels_for_query(query: str) -> frozenset[str]:
    """Narrow a pattern request when it explicitly names a parent garment."""
    normalized_query = query.casefold()
    for label, terms in _PATTERN_GARMENT_QUERY_TERMS:
        if any(term in normalized_query for term in terms):
            return frozenset((label,))
    return _PATTERN_GARMENT_LABELS


def _clip_local_mask_to_image(
    image: np.ndarray,
    local_mask: np.ndarray,
    coordinate_offset: tuple[float, float],
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[float, float]]:
    """Clip a box-local mask to image bounds while preserving coordinates."""
    x_offset = int(round(coordinate_offset[0]))
    y_offset = int(round(coordinate_offset[1]))
    image_height, image_width = image.shape[:2]
    image_x_min = max(0, x_offset)
    image_y_min = max(0, y_offset)
    image_x_max = min(image_width, x_offset + local_mask.shape[1])
    image_y_max = min(image_height, y_offset + local_mask.shape[0])
    if image_x_max <= image_x_min or image_y_max <= image_y_min:
        return None, None, (float(image_x_min), float(image_y_min))

    mask_x_min = image_x_min - x_offset
    mask_y_min = image_y_min - y_offset
    mask_x_max = mask_x_min + image_x_max - image_x_min
    mask_y_max = mask_y_min + image_y_max - image_y_min
    return (
        image[image_y_min:image_y_max, image_x_min:image_x_max],
        local_mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max],
        (float(image_x_min), float(image_y_min)),
    )


def _salient_pattern_mask(
    image_crop: np.ndarray,
    garment_mask: np.ndarray,
    *,
    min_component_fraction: float = 0.0005,
    max_component_fraction: float = 0.20,
    max_components: int = 16,
) -> np.ndarray | None:
    """Extract compact color outliers while rejecting borders and broad shading."""
    garment_area = int(garment_mask.sum())
    if garment_area < 64:
        return None
    lab = cv2.cvtColor(image_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    garment_pixels = lab[garment_mask]
    dominant_color = np.median(garment_pixels, axis=0)
    color_distance = np.linalg.norm(lab - dominant_color, axis=2)
    garment_distances = color_distance[garment_mask]
    median_distance = float(np.median(garment_distances))
    mad = float(np.median(np.abs(garment_distances - median_distance)))
    threshold = median_distance + max(18.0, 3.0 * mad)

    erosion_size = max(3, int(round(min(garment_mask.shape) * 0.015)))
    if erosion_size % 2 == 0:
        erosion_size += 1
    interior_mask = cv2.erode(
        garment_mask.astype("uint8"),
        np.ones((erosion_size, erosion_size), dtype="uint8"),
    ).astype(bool)
    candidates = (color_distance >= threshold) & interior_mask
    candidates = cv2.morphologyEx(
        candidates.astype("uint8"),
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype="uint8"),
    )
    candidates = cv2.morphologyEx(
        candidates,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype="uint8"),
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates,
        connectivity=8,
    )
    min_area = max(12, int(math.ceil(garment_area * min_component_fraction)))
    max_area = max(min_area, int(math.floor(garment_area * max_component_fraction)))
    components = []
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        component_mask = labels == component_id
        mean_distance = float(color_distance[component_mask].mean())
        components.append((area * mean_distance, component_id))
    if not components:
        return None
    components.sort(reverse=True)
    selected_ids = [component_id for _, component_id in components[:max_components]]
    selected_mask = np.isin(labels, selected_ids) & garment_mask
    return selected_mask if selected_mask.any() else None


def _segmentation_box_iou(
    first: SegmentationInstance,
    second: SegmentationInstance,
) -> float:
    """Return bounding-box IoU for category-agnostic duplicate suppression."""
    x_min = max(first.box.x_min, second.box.x_min)
    y_min = max(first.box.y_min, second.box.y_min)
    x_max = min(first.box.x_max, second.box.x_max)
    y_max = min(first.box.y_max, second.box.y_max)
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    first_area = (first.box.x_max - first.box.x_min) * (
        first.box.y_max - first.box.y_min
    )
    second_area = (second.box.x_max - second.box.x_min) * (
        second.box.y_max - second.box.y_min
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _segmentation_instance_to_localized_region(
    instance: SegmentationInstance,
) -> LocalizedRegion:
    """Adapt a garment instance so shared polygon rasterization can be reused."""
    return LocalizedRegion(
        region_label=instance.category_label,
        matched_text=instance.category_label,
        confidence=instance.confidence,
        box=LocalizationBoundingBox(
            x_min=instance.box.x_min,
            y_min=instance.box.y_min,
            x_max=instance.box.x_max,
            y_max=instance.box.y_max,
        ),
        mask=instance.mask,
    )


def _crop_to_subject_roi(
    image: np.ndarray,
    subject_roi: SegmentationSubjectROI | None,
    *,
    margin: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Crop an expanded subject box and retain its image-coordinate offset."""
    if subject_roi is None:
        return image, (0.0, 0.0)

    image_height, image_width = image.shape[:2]
    roi_width = subject_roi.x_max - subject_roi.x_min
    roi_height = subject_roi.y_max - subject_roi.y_min
    x_margin = roi_width * margin
    y_margin = roi_height * margin
    x_min = max(0, int(math.floor(subject_roi.x_min - x_margin)))
    y_min = max(0, int(math.floor(subject_roi.y_min - y_margin)))
    x_max = min(image_width, int(math.ceil(subject_roi.x_max + x_margin)))
    y_max = min(image_height, int(math.ceil(subject_roi.y_max + y_margin)))
    if x_max <= x_min or y_max <= y_min:
        raise InvalidImageInputError("Subject ROI does not overlap the input image.")
    return image[y_min:y_max, x_min:x_max], (float(x_min), float(y_min))


def _candidate_to_localized_region(
    candidate: GroundedMaskCandidate,
    *,
    prompt: LocalizationPrompt,
    coordinate_offset: tuple[float, float],
    expected_shape: tuple[int, int],
    min_mask_area: int,
) -> LocalizedRegion | None:
    """Convert one dense local mask to API polygons and a mask-derived box."""
    mask = np.asarray(candidate.mask, dtype=bool)
    if mask.shape != expected_shape:
        raise ModelNotReadyError(
            "SAM-HQ returned a mask whose dimensions do not match its input image."
        )
    if int(mask.sum()) < min_mask_area:
        return None

    y_values, x_values = np.nonzero(mask)
    x_offset, y_offset = coordinate_offset
    box = LocalizationBoundingBox(
        x_min=float(x_values.min()) + x_offset,
        y_min=float(y_values.min()) + y_offset,
        x_max=float(x_values.max() + 1) + x_offset,
        y_max=float(y_values.max() + 1) + y_offset,
    )
    polygons = _mask_to_polygons(
        mask,
        coordinate_offset=coordinate_offset,
    )
    if not polygons:
        return None
    return LocalizedRegion(
        region_label=prompt.region_label,
        matched_text=candidate.phrase or prompt.matched_term,
        confidence=max(0.0, min(1.0, candidate.confidence)),
        box=box,
        mask=polygons,
    )


def _mask_to_polygons(
    mask: np.ndarray,
    *,
    coordinate_offset: tuple[float, float],
) -> list[list[float]]:
    """Convert one binary mask to external contour polygons."""
    contours, _ = cv2.findContours(
        mask.astype("uint8") * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    x_offset, y_offset = coordinate_offset
    polygons = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        polygon = [
            coordinate
            for x_value, y_value in points
            for coordinate in (
                float(x_value) + x_offset,
                float(y_value) + y_offset,
            )
        ]
        polygons.append(polygon)
    return polygons
