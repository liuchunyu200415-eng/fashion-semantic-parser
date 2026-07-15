"""Detectron2-based segmentation models for PRD 3.1.1."""

import copy
import itertools
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
    SegmentationSubjectROI,
)
from fashion_semantic_parser.service.segmentation_metrics import (
    _coco_ap_at_iou,
    _coco_matched_mask_iou_metrics,
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
    checkpoint_period: int = Field(default=1000, ge=1)
    eval_period: int = Field(default=0, ge=0)
    num_workers: int = Field(default=2, ge=0)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = "cuda"
    resume: bool = False
    evaluate_after_training: bool = True


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
        cfg.SOLVER.CHECKPOINT_PERIOD = self.settings.checkpoint_period
        cfg.TEST.EVAL_PERIOD = self.settings.eval_period
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
        if not self.settings.evaluate_after_training:
            cfg.DATASETS.TEST = ()
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        trainer_class = self._trainer_class(detectron2)
        trainer = trainer_class(cfg)
        trainer.resume_or_load(resume=self.settings.resume)
        trainer.train()

    def evaluate(self) -> Any:
        """Evaluate the configured segmentation model without further training."""
        detectron2 = _load_detectron2_modules()
        self.register_datasets()
        cfg = self.build_config()
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        trainer_class = self._trainer_class(detectron2)
        trainer = trainer_class(cfg)
        trainer.resume_or_load(resume=False)
        return trainer.test(cfg, trainer.model)

    def predict_image(self, image_path: Path) -> SegmentationPrediction:
        """Run instance segmentation on one RGB product image."""
        detectron2 = _load_detectron2_modules()
        cfg = self.build_config()
        predictor = detectron2["DefaultPredictor"](cfg)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        outputs = predictor(image)
        instances = _filter_detectron2_instances_by_score(
            outputs["instances"],
            self.settings.score_threshold,
        ).to("cpu")
        return convert_detectron2_instances(
            instances=instances,
            image_path=image_path,
            score_threshold=self.settings.score_threshold,
        )

    def benchmark_latency(
        self,
        image_paths: list[Path],
        warmup_runs: int = 10,
        measured_runs: int = 100,
    ) -> dict[str, Any]:
        """Benchmark a loaded predictor without model-load or image-read time."""
        if not image_paths:
            raise ValueError("At least one image is required for latency benchmarking.")
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be greater than or equal to zero.")
        if measured_runs < 1:
            raise ValueError("measured_runs must be at least one.")

        detectron2 = _load_detectron2_modules()
        torch = _load_torch_module()
        cfg = self.build_config()
        predictor = detectron2["DefaultPredictor"](cfg)
        loaded_images = _load_benchmark_images(image_paths)

        for index in range(warmup_runs):
            image_path, image = loaded_images[index % len(loaded_images)]
            outputs = predictor(image)
            _synchronize_torch_device(torch, self.settings.device)
            instances = _filter_detectron2_instances_by_score(
                outputs["instances"],
                self.settings.score_threshold,
            ).to("cpu")
            convert_detectron2_instances(
                instances=instances,
                image_path=image_path,
                score_threshold=self.settings.score_threshold,
            )

        import time

        predictor_latencies_ms: list[float] = []
        pipeline_latencies_ms: list[float] = []
        for index in range(measured_runs):
            image_path, image = loaded_images[index % len(loaded_images)]
            _synchronize_torch_device(torch, self.settings.device)
            start_time = time.perf_counter()
            outputs = predictor(image)
            _synchronize_torch_device(torch, self.settings.device)
            predictor_end_time = time.perf_counter()
            instances = _filter_detectron2_instances_by_score(
                outputs["instances"],
                self.settings.score_threshold,
            ).to("cpu")
            convert_detectron2_instances(
                instances=instances,
                image_path=image_path,
                score_threshold=self.settings.score_threshold,
            )
            pipeline_end_time = time.perf_counter()
            predictor_latencies_ms.append((predictor_end_time - start_time) * 1000.0)
            pipeline_latencies_ms.append((pipeline_end_time - start_time) * 1000.0)

        return {
            "device": self.settings.device,
            "torch_version": str(torch.__version__),
            "gpu_name": _torch_device_name(torch, self.settings.device),
            "source_image_count": len(loaded_images),
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "score_threshold": self.settings.score_threshold,
            "excluded_from_timing": ["model_load", "weight_load", "image_decode"],
            "predictor_ms": _latency_summary(predictor_latencies_ms),
            "pipeline_ms": _latency_summary(pipeline_latencies_ms),
        }

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

    def _trainer_class(self, detectron2: dict[str, Any]) -> type:
        """Return the trainer class needed by the configured model family."""
        default_trainer = detectron2["DefaultTrainer"]
        coco_evaluator = _mask_box_coco_evaluator_class(
            detectron2,
            score_threshold=self.settings.score_threshold,
        )

        class SegmentationTrainer(default_trainer):  # type: ignore[misc, valid-type]
            """DefaultTrainer with COCO instance segmentation evaluation."""

            @classmethod
            def build_evaluator(
                cls,
                cfg: Any,
                dataset_name: str,
                output_folder: str | None = None,
            ) -> Any:
                """Build COCO metrics output for validation datasets."""
                if output_folder is None:
                    output_folder = str(Path(cfg.OUTPUT_DIR) / "inference")
                return coco_evaluator(dataset_name, output_dir=output_folder)

        if self.settings.model_family != "mask2former":
            return SegmentationTrainer

        mask2former = _load_mask2former_modules()
        build_detection_train_loader = detectron2["build_detection_train_loader"]
        mapper_class = mask2former["COCOInstanceNewBaselineDatasetMapper"]

        class Mask2FormerTrainer(SegmentationTrainer):
            """COCO-evaluated trainer with Mask2Former's instance mapper."""

            @classmethod
            def build_train_loader(cls, cfg: Any) -> Any:
                """Build a loader that provides tensor masks expected by Mask2Former."""
                mapper = mapper_class(cfg, True)
                return build_detection_train_loader(cfg, mapper=mapper)

            @classmethod
            def build_optimizer(cls, cfg: Any, model: Any) -> Any:
                """Build Mask2Former's parameter-grouped optimizer."""
                return _build_mask2former_optimizer(
                    cfg,
                    model,
                    mask2former["maybe_add_gradient_clipping"],
                )

            @classmethod
            def build_lr_scheduler(cls, cfg: Any, optimizer: Any) -> Any:
                """Use the DeepLab scheduler expected by Mask2Former configs."""
                return mask2former["build_lr_scheduler"](cfg, optimizer)

        return Mask2FormerTrainer


def convert_detectron2_instances(
    instances: Any,
    image_path: Path,
    score_threshold: float = 0.0,
) -> SegmentationPrediction:
    """Convert Detectron2 Instances to project prediction schema."""
    instances = _filter_detectron2_instances_by_score(instances, score_threshold)
    scores = _tensor_to_list(instances.scores)
    classes = _tensor_to_list(instances.pred_classes)
    mask_list = _masks_to_arrays(instances.pred_masks)
    boxes = _boxes_with_mask_fallback(instances, mask_list)
    masks = _masks_to_polygons(mask_list)
    predictions: list[SegmentationInstance] = []

    for index, class_index in enumerate(classes):
        if float(scores[index]) < score_threshold:
            continue
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


def filter_prediction_by_subject_roi(
    prediction: SegmentationPrediction,
    subject_roi: SegmentationSubjectROI,
    min_box_overlap: float = 0.05,
) -> SegmentationPrediction:
    """Keep predicted instances centered inside the subject/person ROI."""
    filtered_instances = [
        instance
        for instance in prediction.instances
        if _box_center_in_roi(instance.box, subject_roi)
        and _box_overlap_ratio(instance.box, subject_roi) >= min_box_overlap
    ]
    return SegmentationPrediction(
        image_path=prediction.image_path,
        instances=filtered_instances,
    )


def _load_detectron2_modules() -> dict[str, Any]:
    """Import Detectron2 lazily so local non-GPU tooling keeps working."""
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data import build_detection_train_loader
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultPredictor, DefaultTrainer
        from detectron2.evaluation import COCOEvaluator
        from detectron2.structures import BitMasks
    except ImportError as error:
        raise ModelNotReadyError(
            "Detectron2 is required for PRD 3.1.1 baseline training and "
            "inference. Install it in the cloud GPU environment before "
            "running segmentation baseline scripts."
        ) from error

    return {
        "COCOEvaluator": COCOEvaluator,
        "DefaultPredictor": DefaultPredictor,
        "DefaultTrainer": DefaultTrainer,
        "build_detection_train_loader": build_detection_train_loader,
        "BitMasks": BitMasks,
        "get_cfg": get_cfg,
        "model_zoo": model_zoo,
        "register_coco_instances": register_coco_instances,
    }


def _load_mask2former_modules() -> dict[str, Any]:
    """Import Mask2Former project config lazily for the PRD target model."""
    try:
        from detectron2.projects.deeplab import (
            add_deeplab_config,
            build_lr_scheduler,
        )
        from detectron2.solver.build import maybe_add_gradient_clipping
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
        "build_lr_scheduler": build_lr_scheduler,
        "maybe_add_gradient_clipping": maybe_add_gradient_clipping,
    }


def _load_torch_module() -> Any:
    """Import PyTorch lazily for GPU synchronization and device metadata."""
    try:
        import torch
    except ImportError as error:
        raise ModelNotReadyError(
            "PyTorch is required for segmentation latency benchmarking."
        ) from error
    return torch


def _load_benchmark_images(image_paths: list[Path]) -> list[tuple[Path, Any]]:
    """Read benchmark images once so disk decode is outside measured latency."""
    loaded_images = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read benchmark image: {image_path}")
        loaded_images.append((image_path, image))
    return loaded_images


def _synchronize_torch_device(torch: Any, device: str) -> None:
    """Synchronize CUDA timing while remaining a no-op for CPU benchmarks."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _torch_device_name(torch: Any, device: str) -> str | None:
    """Return the active CUDA device name when available."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(torch.cuda.current_device()))


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    """Summarize measured milliseconds with central and tail latency."""
    import numpy as np

    if not latencies_ms:
        raise ValueError("At least one latency sample is required.")
    values = np.asarray(latencies_ms, dtype=float)
    mean_ms = float(np.mean(values))
    return {
        "mean": mean_ms,
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "fps_from_mean": float(1000.0 / mean_ms),
    }


def _build_mask2former_optimizer(
    cfg: Any,
    model: Any,
    maybe_add_gradient_clipping: Any,
) -> Any:
    """Build the optimizer used by Mask2Former's official trainer."""
    try:
        import torch
    except ImportError as error:
        raise ModelNotReadyError(
            "PyTorch is required to build the Mask2Former optimizer."
        ) from error

    defaults = {
        "lr": cfg.SOLVER.BASE_LR,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
    }
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params = []
    memo = set()

    for module_name, module in model.named_modules():
        for parameter_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad or value in memo:
                continue
            memo.add(value)
            hyperparameters = copy.copy(defaults)
            if "backbone" in module_name:
                hyperparameters["lr"] *= cfg.SOLVER.BACKBONE_MULTIPLIER
            if parameter_name in {
                "relative_position_bias_table",
                "absolute_pos_embed",
            }:
                hyperparameters["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparameters["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_NORM
            if isinstance(module, torch.nn.Embedding):
                hyperparameters["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_EMBED
            params.append({"params": [value], **hyperparameters})

    optimizer_type = cfg.SOLVER.OPTIMIZER
    optimizer_class = _mask2former_optimizer_class(cfg, torch)
    if optimizer_type == "SGD":
        optimizer = optimizer_class(
            params,
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
        )
    elif optimizer_type == "ADAMW":
        optimizer = optimizer_class(params, cfg.SOLVER.BASE_LR)
    else:
        raise NotImplementedError(
            f"Unsupported Mask2Former optimizer: {optimizer_type}"
        )

    if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def _mask2former_optimizer_class(cfg: Any, torch: Any) -> type:
    """Return SGD or AdamW with Mask2Former full-model gradient clipping."""
    optimizer_type = cfg.SOLVER.OPTIMIZER
    optimizer_class = torch.optim.SGD if optimizer_type == "SGD" else torch.optim.AdamW
    clip_config = cfg.SOLVER.CLIP_GRADIENTS
    if not (
        clip_config.ENABLED
        and clip_config.CLIP_TYPE == "full_model"
        and clip_config.CLIP_VALUE > 0.0
    ):
        return optimizer_class

    clip_value = clip_config.CLIP_VALUE

    class FullModelGradientClippingOptimizer(optimizer_class):  # type: ignore[misc]
        """Apply one global norm clip before each optimizer step."""

        def step(self, closure: Any = None) -> Any:
            all_params = itertools.chain(
                *(group["params"] for group in self.param_groups)
            )
            torch.nn.utils.clip_grad_norm_(all_params, clip_value)
            return super().step(closure=closure)

    return FullModelGradientClippingOptimizer


def _tensor_to_list(value: Any) -> list[Any]:
    """Convert tensor-like values to plain Python lists."""
    if hasattr(value, "numpy"):
        return value.numpy().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _box_overlap_ratio(
    box: SegmentationBoundingBox,
    roi: SegmentationSubjectROI,
) -> float:
    """Return the fraction of a prediction box covered by a subject ROI."""
    intersection_x_min = max(box.x_min, roi.x_min)
    intersection_y_min = max(box.y_min, roi.y_min)
    intersection_x_max = min(box.x_max, roi.x_max)
    intersection_y_max = min(box.y_max, roi.y_max)
    intersection_width = max(0.0, intersection_x_max - intersection_x_min)
    intersection_height = max(0.0, intersection_y_max - intersection_y_min)
    box_area = max(0.0, box.x_max - box.x_min) * max(0.0, box.y_max - box.y_min)
    if box_area <= 0.0:
        return 0.0
    return (intersection_width * intersection_height) / box_area


def _box_center_in_roi(
    box: SegmentationBoundingBox,
    roi: SegmentationSubjectROI,
) -> bool:
    """Return whether a prediction box center falls inside the subject ROI."""
    center_x = (box.x_min + box.x_max) / 2.0
    center_y = (box.y_min + box.y_max) / 2.0
    return roi.x_min <= center_x <= roi.x_max and roi.y_min <= center_y <= roi.y_max


def _boxes_with_mask_fallback(
    instances: Any, mask_list: list[Any]
) -> list[list[float]]:
    """Use model boxes when valid, otherwise derive boxes from predicted masks."""
    boxes = _tensor_to_list(instances.pred_boxes.tensor)
    fallback_boxes = [_mask_to_box(mask) for mask in mask_list]

    fixed_boxes: list[list[float]] = []
    for index, fallback_box in enumerate(fallback_boxes):
        if index < len(boxes) and _is_valid_box(boxes[index]):
            fixed_boxes.append([float(value) for value in boxes[index]])
            continue
        fixed_boxes.append(fallback_box)
    return fixed_boxes


def _is_valid_box(box: Any) -> bool:
    """Return whether a box has positive width and height."""
    if len(box) < 4:
        return False
    x_min, y_min, x_max, y_max = box[:4]
    return float(x_max) > float(x_min) and float(y_max) > float(y_min)


def _mask_to_box(mask: Any) -> list[float]:
    """Compute an xyxy pixel-boundary box from a binary mask."""
    import numpy as np

    y_indices, x_indices = np.where(np.asarray(mask, dtype=bool))
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(x_indices.min()),
        float(y_indices.min()),
        float(x_indices.max() + 1),
        float(y_indices.max() + 1),
    ]


def _mask_box_coco_evaluator_class(
    detectron2: dict[str, Any],
    score_threshold: float,
) -> type:
    """Return a COCOEvaluator that filters scores and derives boxes from masks."""
    coco_evaluator = detectron2["COCOEvaluator"]
    bit_masks_class = detectron2.get("BitMasks")

    class MaskBoxCOCOEvaluator(coco_evaluator):  # type: ignore[misc, valid-type]
        """COCO evaluator with mask-derived boxes for mask-first models."""

        def process(self, inputs: Any, outputs: Any) -> None:
            """Replace invalid predicted boxes before COCO metric serialization."""
            for output in outputs:
                instances = output.get("instances")
                if instances is None or not instances.has("pred_masks"):
                    continue
                instances = _filter_detectron2_instances_by_score(
                    instances,
                    score_threshold,
                )
                if not instances.has("pred_boxes") or _instances_have_invalid_boxes(
                    instances
                ):
                    instances.pred_boxes = _detectron2_boxes_from_masks(
                        instances.pred_masks,
                        bit_masks_class,
                    )
                output["instances"] = instances
            super().process(inputs, outputs)

        def _derive_coco_results(
            self,
            coco_eval: Any,
            iou_type: str,
            class_names: list[str] | None = None,
        ) -> dict[str, float]:
            """Add strict AP and direct mask-IoU metrics for PRD evaluation."""
            results: dict[str, float] = dict(
                super()._derive_coco_results(
                    coco_eval,
                    iou_type,
                    class_names,
                )
            )
            if coco_eval is None:
                return results

            for threshold in (0.85, 0.90):
                metric_name = f"AP{int(threshold * 100)}"
                results[metric_name] = _coco_ap_at_iou(coco_eval, threshold)
                if class_names is not None:
                    for category_index, category_name in enumerate(class_names):
                        results[f"{metric_name}-{category_name}"] = _coco_ap_at_iou(
                            coco_eval,
                            threshold,
                            category_index=category_index,
                        )
            if iou_type == "segm":
                results.update(
                    _coco_matched_mask_iou_metrics(
                        coco_eval,
                        class_names=class_names,
                    )
                )
            return results

    return MaskBoxCOCOEvaluator


def _filter_detectron2_instances_by_score(
    instances: Any, score_threshold: float
) -> Any:
    """Filter Detectron2 Instances by score for Mask2Former outputs."""
    has_field = getattr(instances, "has", None)
    if score_threshold <= 0.0 or not callable(has_field) or not has_field("scores"):
        return instances
    selection = instances.scores >= score_threshold
    get_fields = getattr(instances, "get_fields", None)
    set_field = getattr(instances, "set", None)
    if not callable(get_fields) or not callable(set_field):
        return instances[selection]

    filtered = instances.__class__(instances.image_size)
    for field_name, field_value in get_fields().items():
        field_selection = _selection_for_instance_field(selection, field_value)
        filtered.set(field_name, field_value[field_selection])
    return filtered


def _selection_for_instance_field(selection: Any, field_value: Any) -> Any:
    """Move a Detectron2 instance selection to each field's own device."""
    device = getattr(field_value, "device", None)
    if device is None:
        tensor = getattr(field_value, "tensor", None)
        device = getattr(tensor, "device", None)
    if device is not None and hasattr(selection, "to"):
        return selection.to(device=device)
    return selection


def _instances_have_invalid_boxes(instances: Any) -> bool:
    """Return whether any Detectron2 instance box has non-positive area."""
    boxes = instances.pred_boxes.tensor
    if hasattr(boxes, "numel") and boxes.numel() == 0:
        return False
    invalid = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
    if hasattr(invalid, "any"):
        invalid = invalid.any()
    if hasattr(invalid, "item"):
        return bool(invalid.item())
    return bool(invalid)


def _detectron2_boxes_from_masks(masks: Any, bit_masks_class: Any) -> Any:
    """Build Detectron2 Boxes from tensor masks or BitMasks."""
    if hasattr(masks, "get_bounding_boxes"):
        return masks.get_bounding_boxes()
    if bit_masks_class is None:
        raise ModelNotReadyError("Detectron2 BitMasks is required for mask box eval.")
    return bit_masks_class(masks).get_bounding_boxes()


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


def _masks_to_arrays(masks: Any) -> list[Any]:
    """Keep mask pixels in dense arrays instead of nested Python lists."""
    import numpy as np

    values = masks.numpy() if hasattr(masks, "numpy") else np.asarray(masks)
    if values.ndim == 2:
        values = values[np.newaxis, ...]
    if values.ndim != 3:
        return []
    return [np.asarray(mask, dtype=bool) for mask in values]


def _mask_to_uint8(mask: Any) -> Any:
    """Convert one mask-like object into an OpenCV-compatible uint8 array."""
    import numpy as np

    return np.asarray(mask, dtype="uint8") * 255
