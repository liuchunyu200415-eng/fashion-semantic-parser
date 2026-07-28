"""Automatic primary-person ROI detection for garment segmentation."""

import math
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import cv2
from pydantic import BaseModel, Field

from fashion_semantic_parser.models.segmentation import SegmentationSubjectROI
from fashion_semantic_parser.service.segmentation_baseline import (
    _load_detectron2_modules,
    _load_torch_module,
    _run_predictor_with_precision,
    _tensor_to_list,
)


class PersonROIDetectorSettings(BaseModel):
    """Detectron2 COCO-person detector settings."""

    model_zoo_config: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    weights: str | None = None
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    min_area_ratio: float = Field(default=0.005, ge=0.0, le=1.0)
    center_bias: float = Field(default=0.5, ge=0.0)
    detections_per_image: int = Field(default=20, ge=1)
    precision: Literal["fp32", "fp16"] = "fp16"
    device: str = "cuda"


class Detectron2PersonROIDetector:
    """Detect and select the main person in a fashion image."""

    person_class_index = 0

    def __init__(self, settings: PersonROIDetectorSettings) -> None:
        self.settings = settings
        self._predictor: Any | None = None
        self._predictor_init_lock = Lock()
        self._inference_lock = Lock()

    def detect(self, image_path: Path) -> SegmentationSubjectROI | None:
        """Return the primary COCO-person box or None when no person is found."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        with self._inference_lock:
            outputs = _run_predictor_with_precision(
                self._get_predictor(),
                image,
                _load_torch_module(),
                device=self.settings.device,
                precision=self.settings.precision,
            )
        instances = outputs["instances"].to("cpu")
        return select_primary_person_roi(
            boxes=_tensor_to_list(instances.pred_boxes.tensor),
            scores=_tensor_to_list(instances.scores),
            classes=_tensor_to_list(instances.pred_classes),
            image_width=image.shape[1],
            image_height=image.shape[0],
            person_class_index=self.person_class_index,
            score_threshold=self.settings.score_threshold,
            min_area_ratio=self.settings.min_area_ratio,
            center_bias=self.settings.center_bias,
        )

    def _get_predictor(self) -> Any:
        """Build and cache the COCO person detector."""
        if self._predictor is not None:
            return self._predictor

        with self._predictor_init_lock:
            if self._predictor is None:
                detectron2 = _load_detectron2_modules()
                model_zoo = detectron2["model_zoo"]
                cfg = detectron2["get_cfg"]()
                cfg.merge_from_file(
                    model_zoo.get_config_file(self.settings.model_zoo_config)
                )
                cfg.MODEL.WEIGHTS = self.settings.weights or (
                    model_zoo.get_checkpoint_url(self.settings.model_zoo_config)
                )
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.settings.score_threshold
                cfg.TEST.DETECTIONS_PER_IMAGE = self.settings.detections_per_image
                cfg.MODEL.DEVICE = self.settings.device
                self._predictor = detectron2["DefaultPredictor"](cfg)
        return self._predictor


def select_primary_person_roi(
    *,
    boxes: list[list[float]],
    scores: list[float],
    classes: list[int],
    image_width: int,
    image_height: int,
    person_class_index: int = 0,
    score_threshold: float = 0.7,
    min_area_ratio: float = 0.005,
    center_bias: float = 0.5,
) -> SegmentationSubjectROI | None:
    """Select a large, confident, central person from COCO detections."""
    image_area = float(image_width * image_height)
    if image_area <= 0.0:
        return None

    candidates: list[tuple[float, float, SegmentationSubjectROI]] = []
    for box, score, class_index in zip(boxes, scores, classes, strict=False):
        if int(class_index) != person_class_index or float(score) < score_threshold:
            continue
        roi = _clamped_roi(box, image_width=image_width, image_height=image_height)
        if roi is None:
            continue
        area = (roi.x_max - roi.x_min) * (roi.y_max - roi.y_min)
        if area / image_area < min_area_ratio:
            continue

        center_distance = _normalized_center_distance(
            roi,
            image_width=image_width,
            image_height=image_height,
        )
        rank = area * float(score) / (1.0 + center_bias * center_distance)
        candidates.append((rank, float(score), roi))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[:2])[2]


def _clamped_roi(
    box: list[float],
    *,
    image_width: int,
    image_height: int,
) -> SegmentationSubjectROI | None:
    """Clamp one xyxy detection to valid image coordinates."""
    if len(box) < 4:
        return None
    x_min = max(0.0, min(float(image_width), float(box[0])))
    y_min = max(0.0, min(float(image_height), float(box[1])))
    x_max = max(0.0, min(float(image_width), float(box[2])))
    y_max = max(0.0, min(float(image_height), float(box[3])))
    if x_max <= x_min or y_max <= y_min:
        return None
    return SegmentationSubjectROI(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )


def _normalized_center_distance(
    roi: SegmentationSubjectROI,
    *,
    image_width: int,
    image_height: int,
) -> float:
    """Measure ROI-center distance from image center in half-frame units."""
    center_x = (roi.x_min + roi.x_max) / 2.0
    center_y = (roi.y_min + roi.y_max) / 2.0
    normalized_x = (center_x - image_width / 2.0) / max(image_width / 2.0, 1.0)
    normalized_y = (center_y - image_height / 2.0) / max(image_height / 2.0, 1.0)
    return math.hypot(normalized_x, normalized_y)
