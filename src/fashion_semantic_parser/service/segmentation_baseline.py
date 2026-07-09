"""Detectron2-based segmentation models for PRD 3.1.1."""

from pathlib import Path
from typing import Any, Literal

import cv2
from pydantic import BaseModel, Field

from fashion_semantic_parser.common.exceptions import ModelNotReadyError
from fashion_semantic_parser.common.paths import (
    resolve_project_path,
    to_project_relative_path,
)
from fashion_semantic_parser.dao.segmentation.taxonomy import (
    PRD_SEGMENTATION_CATEGORIES,
)
from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
)


class SegmentationBaselineSettings(BaseModel):
    """Training and inference settings for a Detectron2 segmentation model."""

    model_family: Literal["mask_rcnn", "mask2former"] = "mask_rcnn"
    train_json: str = "data/processed/autodl/segmentation/deepfashion2_train.json"
    val_json: str = "data/processed/autodl/segmentation/deepfashion2_validation.json"
    image_root: str = "."
    output_dir: str = "outputs/segmentation/mask_rcnn_r50_fpn"
    config_source: Literal["detectron2_model_zoo", "local"] = "detectron2_model_zoo"
    config_file: str | None = None
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
    """Adapter around Detectron2-family models for training and inference."""

    train_dataset_name = "prd_3_1_1_deepfashion2_train"
    val_dataset_name = "prd_3_1_1_deepfashion2_validation"

    def __init__(self, settings: SegmentationBaselineSettings) -> None:
        """Create a baseline adapter with explicit settings."""
        self.settings = settings

    def build_config(self) -> Any:
        """Build a Detectron2 cfg object for the configured model family.

        Raises:
            ModelNotReadyError: If the required optional model framework is missing.
        """
        detectron2 = _load_detectron2_modules()
        cfg = detectron2["get_cfg"]()
        if self.settings.model_family == "mask2former":
            mask2former = _load_mask2former_modules()
            mask2former["add_deeplab_config"](cfg)
            mask2former["add_maskformer2_config"](cfg)

        cfg.merge_from_file(self._resolve_config_file(detectron2["model_zoo"]))
        cfg.DATASETS.TRAIN = (self.train_dataset_name,)
        cfg.DATASETS.TEST = (self.val_dataset_name,)
        cfg.DATALOADER.NUM_WORKERS = self.settings.num_workers
        cfg.SOLVER.IMS_PER_BATCH = self.settings.ims_per_batch
        cfg.SOLVER.BASE_LR = self.settings.base_lr
        cfg.SOLVER.MAX_ITER = self.settings.max_iter
        self._apply_model_head_settings(cfg)
        self._apply_trainer_compatibility_settings(cfg)
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
        """Train the configured segmentation model on registered COCO data."""
        detectron2 = _load_detectron2_modules()
        self.register_datasets()
        cfg = self.build_config()
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        trainer_class = self._trainer_class(detectron2)
        trainer = trainer_class(cfg)
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
        """Resolve custom, local-config, or model-zoo weights."""
        if self.settings.weights:
            return self.settings.weights
        if self.settings.config_source == "detectron2_model_zoo":
            return model_zoo.get_checkpoint_url(self.settings.model_zoo_config)
        return ""

    def _resolve_config_file(self, model_zoo: Any) -> str:
        """Resolve a Detectron2 model-zoo or local config path."""
        config_file = self.settings.config_file or self.settings.model_zoo_config
        if self.settings.config_source == "detectron2_model_zoo":
            return model_zoo.get_config_file(config_file)

        path = Path(config_file)
        if path.is_absolute():
            if path.exists():
                return str(path)
            raise FileNotFoundError(f"Segmentation config file not found: {path}")

        project_path = resolve_project_path(path)
        if project_path.exists():
            return str(project_path)
        if path.exists():
            return str(path)
        raise FileNotFoundError(
            "Segmentation config file not found. For Mask2Former, clone the "
            "Mask2Former project or pass --config with a YAML file whose "
            f"config_file exists. Missing: {config_file}"
        )

    def _apply_model_head_settings(self, cfg: Any) -> None:
        """Apply class-count and score-threshold settings across model families."""
        if hasattr(cfg.MODEL, "ROI_HEADS"):
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.settings.num_classes
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.settings.score_threshold
        if hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = self.settings.num_classes
        if hasattr(cfg.MODEL, "PANOPTIC_FPN"):
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
                self.settings.score_threshold
            )
        if hasattr(cfg.MODEL, "MASK_FORMER"):
            cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
            if hasattr(cfg.MODEL.MASK_FORMER.TEST, "OBJECT_MASK_THRESHOLD"):
                cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = (
                    self.settings.score_threshold
                )

    def _apply_trainer_compatibility_settings(self, cfg: Any) -> None:
        """Adapt target-model configs to the generic Detectron2 trainer."""
        if self.settings.model_family != "mask2former":
            return
        cfg.MODEL.MASK_ON = True
        cfg.INPUT.MASK_FORMAT = "bitmask"
        clip_gradients = cfg.SOLVER.CLIP_GRADIENTS
        if (
            getattr(clip_gradients, "ENABLED", False)
            and getattr(clip_gradients, "CLIP_TYPE", "") == "full_model"
        ):
            clip_gradients.CLIP_TYPE = "norm"

    def _trainer_class(self, detectron2: dict[str, Any]) -> type:
        """Return the trainer class needed by the configured model family."""
        if self.settings.model_family != "mask2former":
            return detectron2["DefaultTrainer"]

        mask2former = _load_mask2former_modules()
        default_trainer = detectron2["DefaultTrainer"]
        build_detection_train_loader = detectron2["build_detection_train_loader"]
        mapper_class = mask2former["COCOInstanceNewBaselineDatasetMapper"]

        class Mask2FormerTrainer(default_trainer):  # type: ignore[misc, valid-type]
            """DefaultTrainer with Mask2Former's instance segmentation mapper."""

            @classmethod
            def build_train_loader(cls, cfg: Any) -> Any:
                """Build a loader that provides tensor masks expected by Mask2Former."""
                mapper = mapper_class(cfg, True)
                return build_detection_train_loader(cfg, mapper=mapper)

        return Mask2FormerTrainer


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
        from detectron2.data import build_detection_train_loader
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
        "build_detection_train_loader": build_detection_train_loader,
        "get_cfg": get_cfg,
        "model_zoo": model_zoo,
        "register_coco_instances": register_coco_instances,
    }


def _load_mask2former_modules() -> dict[str, Any]:
    """Import Mask2Former project config lazily for the PRD target model."""
    try:
        from detectron2.projects.deeplab import add_deeplab_config
        from mask2former import add_maskformer2_config
        from mask2former.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (  # noqa: E501
            COCOInstanceNewBaselineDatasetMapper,
        )
    except ImportError as error:
        raise ModelNotReadyError(
            "Mask2Former is the PRD-aligned target model for 3.1.1, but the "
            "Mask2Former or Detectron2 DeepLab project config is not importable. "
            "Install/clone Mask2Former, install Detectron2 with project modules, "
            "and add Mask2Former to PYTHONPATH before running the config."
        ) from error
    return {
        "COCOInstanceNewBaselineDatasetMapper": COCOInstanceNewBaselineDatasetMapper,
        "add_deeplab_config": add_deeplab_config,
        "add_maskformer2_config": add_maskformer2_config,
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
