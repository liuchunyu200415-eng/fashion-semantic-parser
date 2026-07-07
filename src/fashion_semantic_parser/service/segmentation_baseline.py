"""Detectron2 Mask R-CNN baseline for PRD 3.1.1 segmentation."""

from pathlib import Path
from typing import Any

import cv2
from pydantic import BaseModel, Field

from fashion_semantic_parser.common.exceptions import ModelNotReadyError
from fashion_semantic_parser.common.paths import to_project_relative_path
from fashion_semantic_parser.dao.segmentation.taxonomy import (
    PRD_SEGMENTATION_CATEGORIES,
)
from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
)


class SegmentationBaselineSettings(BaseModel):
    """Training and inference settings for a Detectron2 Mask R-CNN baseline."""

    train_json: str = "data/processed/autodl/segmentation/deepfashion2_train.json"
    val_json: str = "data/processed/autodl/segmentation/deepfashion2_validation.json"
    image_root: str = "."
    output_dir: str = "outputs/segmentation/mask_rcnn_r50_fpn"
    model_zoo_config: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    weights: str | None = None
    num_classes: int = Field(default=len(PRD_SEGMENTATION_CATEGORIES), ge=1)
    ims_per_batch: int = Field(default=2, ge=1)
    base_lr: float = Field(default=0.00025, gt=0.0)
    max_iter: int = Field(default=3000, ge=1)
    num_workers: int = Field(default=2, ge=0)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = "cuda"


class Detectron2SegmentationBaseline:
    """Thin adapter around Detectron2 for train and single-image inference."""

    train_dataset_name = "prd_3_1_1_deepfashion2_train"
    val_dataset_name = "prd_3_1_1_deepfashion2_validation"

    def __init__(self, settings: SegmentationBaselineSettings) -> None:
        """Create a baseline adapter with explicit settings."""
        self.settings = settings

    def build_config(self) -> Any:
        """Build a Detectron2 cfg object for Mask R-CNN.

        Raises:
            ModelNotReadyError: If Detectron2 is not installed.
        """
        detectron2 = _load_detectron2_modules()
        cfg = detectron2["get_cfg"]()
        cfg.merge_from_file(
            detectron2["model_zoo"].get_config_file(self.settings.model_zoo_config)
        )
        cfg.DATASETS.TRAIN = (self.train_dataset_name,)
        cfg.DATASETS.TEST = (self.val_dataset_name,)
        cfg.DATALOADER.NUM_WORKERS = self.settings.num_workers
        cfg.SOLVER.IMS_PER_BATCH = self.settings.ims_per_batch
        cfg.SOLVER.BASE_LR = self.settings.base_lr
        cfg.SOLVER.MAX_ITER = self.settings.max_iter
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.settings.num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.settings.score_threshold
        cfg.MODEL.DEVICE = self.settings.device
        cfg.OUTPUT_DIR = self.settings.output_dir
        cfg.MODEL.WEIGHTS = self._resolve_weights(detectron2["model_zoo"])
        return cfg

    def register_datasets(self) -> None:
        """Register converted COCO files as Detectron2 datasets."""
        detectron2 = _load_detectron2_modules()
        metadata = {
            "thing_classes": [
                category.english_name for category in PRD_SEGMENTATION_CATEGORIES
            ]
        }
        detectron2["register_coco_instances"](
            self.train_dataset_name,
            metadata,
            self.settings.train_json,
            self.settings.image_root,
        )
        detectron2["register_coco_instances"](
            self.val_dataset_name,
            metadata,
            self.settings.val_json,
            self.settings.image_root,
        )

    def train(self) -> None:
        """Train the Mask R-CNN baseline on the registered COCO datasets."""
        detectron2 = _load_detectron2_modules()
        self.register_datasets()
        cfg = self.build_config()
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        trainer = detectron2["DefaultTrainer"](cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def predict_image(self, image_path: Path) -> SegmentationPrediction:
        """Run instance segmentation on one RGB product image."""
        detectron2 = _load_detectron2_modules()
        cfg = self.build_config()
        predictor = detectron2["DefaultPredictor"](cfg)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        return convert_detectron2_instances(
            instances=instances,
            image_path=image_path,
        )

    def _resolve_weights(self, model_zoo: Any) -> str:
        """Resolve custom or model-zoo weights for the configured baseline."""
        if self.settings.weights:
            return self.settings.weights
        return model_zoo.get_checkpoint_url(self.settings.model_zoo_config)


def convert_detectron2_instances(
    instances: Any,
    image_path: Path,
) -> SegmentationPrediction:
    """Convert Detectron2 Instances to project prediction schema."""
    boxes = _tensor_to_list(instances.pred_boxes.tensor)
    scores = _tensor_to_list(instances.scores)
    classes = _tensor_to_list(instances.pred_classes)
    masks = _masks_to_polygons(instances.pred_masks)
    predictions: list[SegmentationInstance] = []

    for index, class_index in enumerate(classes):
        category = PRD_SEGMENTATION_CATEGORIES[int(class_index)]
        x_min, y_min, x_max, y_max = boxes[index]
        predictions.append(
            SegmentationInstance(
                category_id=category.id,
                category_label=category.english_name,
                confidence=float(scores[index]),
                box=SegmentationBoundingBox(
                    x_min=float(x_min),
                    y_min=float(y_min),
                    x_max=float(x_max),
                    y_max=float(y_max),
                ),
                mask=masks[index],
            )
        )

    return SegmentationPrediction(
        image_path=to_project_relative_path(image_path),
        instances=predictions,
    )


def _load_detectron2_modules() -> dict[str, Any]:
    """Import Detectron2 lazily so local non-GPU tooling keeps working."""
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultPredictor, DefaultTrainer
    except ImportError as error:
        raise ModelNotReadyError(
            "Detectron2 is required for PRD 3.1.1 baseline training and "
            "inference. Install it in the cloud GPU environment before "
            "running segmentation baseline scripts."
        ) from error

    return {
        "DefaultPredictor": DefaultPredictor,
        "DefaultTrainer": DefaultTrainer,
        "get_cfg": get_cfg,
        "model_zoo": model_zoo,
        "register_coco_instances": register_coco_instances,
    }


def _tensor_to_list(value: Any) -> list[Any]:
    """Convert tensor-like values to plain Python lists."""
    if hasattr(value, "numpy"):
        return value.numpy().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _masks_to_polygons(masks: Any) -> list[list[list[float]]]:
    """Convert binary masks to external contour polygons."""
    mask_list = _tensor_to_list(masks)
    polygons_by_mask: list[list[list[float]]] = []
    for mask in mask_list:
        contours, _ = cv2.findContours(
            _mask_to_uint8(mask),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons: list[list[float]] = []
        for contour in contours:
            points = contour.reshape(-1, 2)
            if len(points) < 3:
                continue
            polygon = points.astype(float).reshape(-1).tolist()
            polygons.append(polygon)
        polygons_by_mask.append(polygons)
    return polygons_by_mask


def _mask_to_uint8(mask: Any) -> Any:
    """Convert one mask-like object into an OpenCV-compatible uint8 array."""
    import numpy as np

    return np.asarray(mask, dtype="uint8") * 255
