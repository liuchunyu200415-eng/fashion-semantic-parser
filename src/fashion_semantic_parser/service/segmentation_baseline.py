"""Detectron2-based segmentation models for PRD 3.1.1."""

import copy
import itertools
import logging
import math
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal, Sequence

import cv2
from pydantic import BaseModel, Field, model_validator

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

logger = logging.getLogger(__name__)


class SegmentationBaselineSettings(BaseModel):
    """Training and inference settings for a Detectron2 segmentation model."""

    model_family: Literal["mask_rcnn", "mask2former"] = "mask_rcnn"
    train_json: str = "data/processed/autodl/segmentation/deepfashion2_train.json"
    additional_train_jsons: list[str] = Field(default_factory=list)
    train_source_repeat_factors: list[float] | None = None
    repeat_factor_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    val_json: str = "data/processed/autodl/segmentation/deepfashion2_validation.json"
    image_root: str = "."
    output_dir: str = "outputs/segmentation/mask_rcnn_r50_fpn"
    config_source: Literal["detectron2_model_zoo", "local"] = "detectron2_model_zoo"
    config_file: str | None = None
    model_zoo_config: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    weights: str | None = None
    num_classes: int = Field(default=len(PRD_SEGMENTATION_CATEGORIES), ge=1)
    category_names: list[str] | None = None
    ims_per_batch: int = Field(default=2, ge=1)
    base_lr: float = Field(default=0.00025, gt=0.0)
    max_iter: int = Field(default=3000, ge=1)
    checkpoint_period: int = Field(default=1000, ge=1)
    eval_period: int = Field(default=0, ge=0)
    num_workers: int = Field(default=2, ge=0)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    category_score_thresholds: dict[str, float] = Field(default_factory=dict)
    min_size_test: int | None = Field(default=None, ge=1)
    max_size_test: int | None = Field(default=None, ge=1)
    detections_per_image: int | None = Field(default=None, ge=1)
    mask2former_eager_losses: bool = False
    subject_roi_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    precision: Literal["fp32", "fp16"] = "fp32"
    device: str = "cuda"
    resume: bool = False
    evaluate_after_training: bool = True

    @model_validator(mode="after")
    def validate_training_sources(self) -> "SegmentationBaselineSettings":
        """Keep mixed-source paths and repeat factors unambiguous."""
        train_jsons = [self.train_json, *self.additional_train_jsons]
        if len(set(train_jsons)) != len(train_jsons):
            raise ValueError("Training COCO paths must be unique.")
        if self.train_source_repeat_factors is not None:
            if self.repeat_factor_threshold is not None:
                raise ValueError(
                    "Source repeat factors and category repeat sampling cannot "
                    "be enabled together."
                )
            if len(self.train_source_repeat_factors) != len(train_jsons):
                raise ValueError(
                    "train_source_repeat_factors must contain one value per "
                    "training COCO file."
                )
            if any(
                not math.isfinite(factor) or factor <= 0.0
                for factor in self.train_source_repeat_factors
            ):
                raise ValueError(
                    "Training source repeat factors must be finite and positive."
                )
        category_names = self.resolved_category_names()
        if len(category_names) != self.num_classes:
            raise ValueError("category_names must contain exactly num_classes values.")
        if any(not name.strip() for name in category_names):
            raise ValueError("Category names cannot be empty.")
        if len(set(category_names)) != len(category_names):
            raise ValueError("Category names must be unique.")
        unknown_threshold_names = set(self.category_score_thresholds) - set(
            category_names
        )
        if unknown_threshold_names:
            raise ValueError(
                "category_score_thresholds contains unknown categories: "
                + ", ".join(sorted(unknown_threshold_names))
            )
        if any(
            not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0
            for threshold in self.category_score_thresholds.values()
        ):
            raise ValueError(
                "Category score thresholds must be finite values between 0 and 1."
            )
        return self

    def resolved_category_names(self) -> tuple[str, ...]:
        """Return configured labels or the default PRD 3.1.1 taxonomy."""
        if self.category_names is not None:
            return tuple(self.category_names)
        return tuple(category.english_name for category in PRD_SEGMENTATION_CATEGORIES)

    def model_score_threshold(self) -> float:
        """Return the lowest threshold needed before category-aware filtering."""
        return min([self.score_threshold, *self.category_score_thresholds.values()])


class Detectron2SegmentationBaseline:
    """Adapter around Detectron2-family models for training and inference."""

    train_dataset_name = "prd_3_1_1_deepfashion2_train"
    additional_train_dataset_name_prefix = "prd_3_1_1_additional_train"
    val_dataset_name = "prd_3_1_1_deepfashion2_validation"

    def __init__(self, settings: SegmentationBaselineSettings) -> None:
        """Create a baseline adapter with explicit settings."""
        self.settings = settings
        self._predictor: Any | None = None
        self._predictor_init_lock = Lock()
        self._inference_lock = Lock()

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
        self._apply_training_dataset_settings(cfg)
        cfg.DATASETS.TEST = (self.val_dataset_name,)
        cfg.DATALOADER.NUM_WORKERS = self.settings.num_workers
        cfg.SOLVER.IMS_PER_BATCH = self.settings.ims_per_batch
        cfg.SOLVER.BASE_LR = self.settings.base_lr
        cfg.SOLVER.MAX_ITER = self.settings.max_iter
        cfg.SOLVER.CHECKPOINT_PERIOD = self.settings.checkpoint_period
        cfg.TEST.EVAL_PERIOD = self.settings.eval_period
        self._apply_inference_size_settings(cfg)
        self._apply_model_head_settings(cfg)
        self._apply_trainer_compatibility_settings(cfg)
        cfg.MODEL.DEVICE = self.settings.device
        cfg.OUTPUT_DIR = self.settings.output_dir
        cfg.MODEL.WEIGHTS = self._resolve_weights(detectron2["model_zoo"])
        return cfg

    def register_datasets(self) -> None:
        """Register converted COCO files as Detectron2 datasets."""
        detectron2 = _load_detectron2_modules()
        metadata = {"thing_classes": list(self.settings.resolved_category_names())}
        for dataset_name, json_path in self._training_dataset_specs():
            detectron2["register_coco_instances"](
                dataset_name,
                metadata,
                json_path,
                self.settings.image_root,
            )
        detectron2["register_coco_instances"](
            self.val_dataset_name,
            metadata,
            self.settings.val_json,
            self.settings.image_root,
        )

    def _training_dataset_specs(self) -> tuple[tuple[str, str], ...]:
        """Return stable Detectron2 names for every configured training source."""
        additional_specs = tuple(
            (
                f"{self.additional_train_dataset_name_prefix}_{index}",
                json_path,
            )
            for index, json_path in enumerate(
                self.settings.additional_train_jsons,
                start=1,
            )
        )
        return (
            (self.train_dataset_name, self.settings.train_json),
            *additional_specs,
        )

    def _apply_training_dataset_settings(self, cfg: Any) -> None:
        """Configure one or more train sets and optional source balancing."""
        dataset_names = tuple(name for name, _ in self._training_dataset_specs())
        cfg.DATASETS.TRAIN = dataset_names
        repeat_factors = self.settings.train_source_repeat_factors
        if repeat_factors is not None:
            cfg.DATALOADER.SAMPLER_TRAIN = "WeightedTrainingSampler"
            cfg.DATASETS.TRAIN_REPEAT_FACTOR = tuple(
                zip(dataset_names, repeat_factors, strict=True)
            )
            return
        if self.settings.repeat_factor_threshold is not None:
            cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
            cfg.DATALOADER.REPEAT_THRESHOLD = self.settings.repeat_factor_threshold

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
        torch = _load_torch_module()
        return _run_with_precision(
            lambda: trainer.test(cfg, trainer.model),
            torch,
            device=self.settings.device,
            precision=self.settings.precision,
        )

    def predict_image(
        self,
        image_path: Path,
        subject_roi: SegmentationSubjectROI | None = None,
    ) -> SegmentationPrediction:
        """Run segmentation on one image or an expanded subject crop."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        coordinate_offset = (0.0, 0.0)
        if subject_roi is not None:
            image, coordinate_offset = _crop_image_to_subject_roi(
                image,
                subject_roi,
                margin=self.settings.subject_roi_margin,
            )
        predictor = self._get_predictor()

        with self._inference_lock:
            outputs = _run_predictor_with_precision(
                predictor,
                image,
                _load_torch_module(),
                device=self.settings.device,
                precision=self.settings.precision,
            )
        instances = _filter_detectron2_instances_by_score(
            outputs["instances"],
            self.settings.model_score_threshold(),
        ).to("cpu")
        return convert_detectron2_instances(
            instances=instances,
            image_path=image_path,
            score_threshold=self.settings.score_threshold,
            category_score_thresholds=self.settings.category_score_thresholds,
            coordinate_offset=coordinate_offset,
            category_names=self.settings.resolved_category_names(),
        )

    def _get_predictor(self, detectron2: dict[str, Any] | None = None) -> Any:
        """Build the expensive predictor once and reuse it across requests."""
        if self._predictor is not None:
            return self._predictor

        with self._predictor_init_lock:
            if self._predictor is None:
                modules = detectron2 or _load_detectron2_modules()
                self._predictor = modules["DefaultPredictor"](self.build_config())
        return self._predictor

    def benchmark_latency(
        self,
        image_paths: list[Path],
        warmup_runs: int = 10,
        measured_runs: int = 100,
        precision: Literal["fp32", "fp16"] = "fp32",
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
            outputs = _run_predictor_with_precision(
                predictor,
                image,
                torch,
                device=self.settings.device,
                precision=precision,
            )
            _synchronize_torch_device(torch, self.settings.device)
            instances = _filter_detectron2_instances_by_score(
                outputs["instances"],
                self.settings.model_score_threshold(),
            ).to("cpu")
            convert_detectron2_instances(
                instances=instances,
                image_path=image_path,
                score_threshold=self.settings.score_threshold,
                category_score_thresholds=self.settings.category_score_thresholds,
                category_names=self.settings.resolved_category_names(),
            )

        import time

        predictor_latencies_ms: list[float] = []
        pipeline_latencies_ms: list[float] = []
        for index in range(measured_runs):
            image_path, image = loaded_images[index % len(loaded_images)]
            _synchronize_torch_device(torch, self.settings.device)
            start_time = time.perf_counter()
            outputs = _run_predictor_with_precision(
                predictor,
                image,
                torch,
                device=self.settings.device,
                precision=precision,
            )
            _synchronize_torch_device(torch, self.settings.device)
            predictor_end_time = time.perf_counter()
            instances = _filter_detectron2_instances_by_score(
                outputs["instances"],
                self.settings.model_score_threshold(),
            ).to("cpu")
            convert_detectron2_instances(
                instances=instances,
                image_path=image_path,
                score_threshold=self.settings.score_threshold,
                category_score_thresholds=self.settings.category_score_thresholds,
                category_names=self.settings.resolved_category_names(),
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
            "category_score_thresholds": self.settings.category_score_thresholds,
            "precision": precision,
            "input_size": {
                "min_size_test": _json_safe_config_value(cfg.INPUT.MIN_SIZE_TEST),
                "max_size_test": _json_safe_config_value(cfg.INPUT.MAX_SIZE_TEST),
            },
            "detections_per_image": _json_safe_config_value(
                cfg.TEST.DETECTIONS_PER_IMAGE
            ),
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
        model_score_threshold = self.settings.model_score_threshold()
        if self.settings.detections_per_image is not None:
            cfg.TEST.DETECTIONS_PER_IMAGE = self.settings.detections_per_image
        if hasattr(cfg.MODEL, "ROI_HEADS"):
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.settings.num_classes
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_score_threshold
        if hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = self.settings.num_classes
        if hasattr(cfg.MODEL, "PANOPTIC_FPN"):
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
                model_score_threshold
            )
        if hasattr(cfg.MODEL, "MASK_FORMER"):
            cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
            if hasattr(cfg.MODEL.MASK_FORMER.TEST, "OBJECT_MASK_THRESHOLD"):
                cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = model_score_threshold

    def _apply_inference_size_settings(self, cfg: Any) -> None:
        """Override test-time resize limits without changing training transforms."""
        if self.settings.min_size_test is not None:
            cfg.INPUT.MIN_SIZE_TEST = self.settings.min_size_test
        if self.settings.max_size_test is not None:
            cfg.INPUT.MAX_SIZE_TEST = self.settings.max_size_test

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
        if self.settings.mask2former_eager_losses:
            _configure_mask2former_eager_losses(
                mask2former["matcher_module"],
                mask2former["criterion_module"],
            )
        build_detection_train_loader = detectron2["build_detection_train_loader"]
        mapper_class = mask2former["COCOInstanceNewBaselineDatasetMapper"]

        class Mask2FormerTrainer(SegmentationTrainer):
            """COCO-evaluated trainer with Mask2Former's instance mapper."""

            @classmethod
            def build_train_loader(cls, cfg: Any) -> Any:
                """Build a loader that accepts both polygon and RLE masks."""
                mapper = _Mask2FormerMixedMaskDatasetMapper(
                    mapper_class(cfg, True),
                    detection_utils=detectron2["detection_utils"],
                    transforms=detectron2["transforms"],
                    torch=_load_torch_module(),
                )
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


class _Mask2FormerMixedMaskDatasetMapper:
    """Adapt Mask2Former's COCO mapper to mixed polygon and RLE masks.

    The upstream mapper always asks Detectron2 for polygon masks. Detectron2
    transforms an RLE annotation into a dense array first, so that path fails
    before Mask2Former receives the sample. This adapter preserves the upstream
    augmentations while constructing BitMasks for every supported COCO format.
    """

    def __init__(
        self,
        upstream_mapper: Any,
        *,
        detection_utils: Any,
        transforms: Any,
        torch: Any,
    ) -> None:
        """Store the configured upstream mapper and optional framework modules."""
        self.upstream_mapper = upstream_mapper
        self.detection_utils = detection_utils
        self.transforms = transforms
        self.torch = torch

    def __call__(self, dataset_dict: dict[str, Any]) -> dict[str, Any]:
        """Map one Detectron2 record without assuming polygon-only masks."""
        import numpy as np

        dataset_dict = copy.deepcopy(dataset_dict)
        mapper = self.upstream_mapper
        image = self.detection_utils.read_image(
            dataset_dict["file_name"],
            format=mapper.img_format,
        )
        self.detection_utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2])
        image, applied_transforms = self.transforms.apply_transform_gens(
            mapper.tfm_gens,
            image,
        )
        padding_mask = applied_transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)
        image_shape = image.shape[:2]
        dataset_dict["image"] = self.torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict["padding_mask"] = self.torch.as_tensor(
            np.ascontiguousarray(padding_mask)
        )

        if not mapper.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" not in dataset_dict:
            return dataset_dict

        for annotation in dataset_dict["annotations"]:
            annotation.pop("keypoints", None)
        annotations = [
            self.detection_utils.transform_instance_annotations(
                annotation,
                applied_transforms,
                image_shape,
            )
            for annotation in dataset_dict.pop("annotations")
            if annotation.get("iscrowd", 0) == 0
        ]
        instances = self.detection_utils.annotations_to_instances(
            annotations,
            image_shape,
            mask_format="bitmask",
        )
        if instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        instances = self.detection_utils.filter_empty_instances(instances)
        if instances.has("gt_masks"):
            instances.gt_masks = instances.gt_masks.tensor
        dataset_dict["instances"] = instances
        return dataset_dict


def convert_detectron2_instances(
    instances: Any,
    image_path: Path,
    score_threshold: float = 0.0,
    category_score_thresholds: dict[str, float] | None = None,
    coordinate_offset: tuple[float, float] = (0.0, 0.0),
    category_names: Sequence[str] | None = None,
) -> SegmentationPrediction:
    """Convert Detectron2 Instances to project prediction schema."""
    category_score_thresholds = category_score_thresholds or {}
    model_score_threshold = min([score_threshold, *category_score_thresholds.values()])
    instances = _filter_detectron2_instances_by_score(
        instances,
        model_score_threshold,
    )
    scores = _tensor_to_list(instances.scores)
    classes = _tensor_to_list(instances.pred_classes)
    mask_list = _masks_to_arrays(instances.pred_masks)
    boxes = _boxes_with_mask_fallback(instances, mask_list)
    masks = _masks_to_polygons(mask_list)
    predictions: list[SegmentationInstance] = []
    x_offset, y_offset = coordinate_offset
    resolved_category_names = tuple(category_names or ())
    if not resolved_category_names:
        resolved_category_names = tuple(
            category.english_name for category in PRD_SEGMENTATION_CATEGORIES
        )

    for index, class_index in enumerate(classes):
        category_index = int(class_index)
        if not 0 <= category_index < len(resolved_category_names):
            raise ValueError(
                f"Predicted class index {category_index} is outside the "
                f"configured {len(resolved_category_names)} categories."
            )
        category_name = resolved_category_names[category_index]
        category_threshold = category_score_thresholds.get(
            category_name,
            score_threshold,
        )
        if float(scores[index]) < category_threshold:
            continue
        x_min, y_min, x_max, y_max = boxes[index]
        predictions.append(
            SegmentationInstance(
                category_id=category_index + 1,
                category_label=category_name,
                confidence=float(scores[index]),
                box=SegmentationBoundingBox(
                    x_min=float(x_min) + x_offset,
                    y_min=float(y_min) + y_offset,
                    x_max=float(x_max) + x_offset,
                    y_max=float(y_max) + y_offset,
                ),
                mask=_offset_mask_polygons(
                    masks[index],
                    x_offset=x_offset,
                    y_offset=y_offset,
                ),
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
        subject_roi=prediction.subject_roi,
        subject_roi_source=prediction.subject_roi_source,
    )


def _crop_image_to_subject_roi(
    image: Any,
    subject_roi: SegmentationSubjectROI,
    *,
    margin: float,
) -> tuple[Any, tuple[float, float]]:
    """Crop an expanded ROI and return its original-image coordinate offset."""
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
        raise ValueError("Subject ROI does not overlap the input image.")

    return image[y_min:y_max, x_min:x_max], (float(x_min), float(y_min))


def _offset_mask_polygons(
    polygons: list[list[float]],
    *,
    x_offset: float,
    y_offset: float,
) -> list[list[float]]:
    """Map crop-relative polygon coordinates back to the original image."""
    return [
        [
            coordinate + (x_offset if index % 2 == 0 else y_offset)
            for index, coordinate in enumerate(polygon)
        ]
        for polygon in polygons
    ]


def _load_detectron2_modules() -> dict[str, Any]:
    """Import Detectron2 lazily so local non-GPU tooling keeps working."""
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data import (
            build_detection_train_loader,
            detection_utils,
            transforms,
        )
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
        "detection_utils": detection_utils,
        "transforms": transforms,
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
        from mask2former.modeling import criterion as criterion_module
        from mask2former.modeling import matcher as matcher_module
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
        "criterion_module": criterion_module,
        "matcher_module": matcher_module,
        "maybe_add_gradient_clipping": maybe_add_gradient_clipping,
    }


def _configure_mask2former_eager_losses(
    matcher_module: Any,
    criterion_module: Any,
) -> None:
    """Replace scripted Mask2Former losses with equivalent eager functions."""
    replacements = (
        (matcher_module, "batch_dice_loss_jit", "batch_dice_loss"),
        (matcher_module, "batch_sigmoid_ce_loss_jit", "batch_sigmoid_ce_loss"),
        (criterion_module, "dice_loss_jit", "dice_loss"),
        (criterion_module, "sigmoid_ce_loss_jit", "sigmoid_ce_loss"),
    )
    try:
        for module, scripted_name, eager_name in replacements:
            setattr(module, scripted_name, getattr(module, eager_name))
    except AttributeError as error:
        raise ModelNotReadyError(
            "The installed Mask2Former loss modules are incompatible with "
            "mask2former_eager_losses."
        ) from error
    logger.info(
        "Using eager Mask2Former Dice/BCE losses to avoid TorchScript fusion "
        "allocation failures."
    )


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


def _run_predictor_with_precision(
    predictor: Any,
    image: Any,
    torch: Any,
    device: str,
    precision: Literal["fp32", "fp16"],
) -> Any:
    """Run one predictor call in FP32 or CUDA autocast FP16."""
    return _run_with_precision(
        lambda: predictor(image),
        torch,
        device=device,
        precision=precision,
    )


def _run_with_precision(
    operation: Callable[[], Any],
    torch: Any,
    device: str,
    precision: Literal["fp32", "fp16"],
) -> Any:
    """Run a synchronous inference operation in FP32 or CUDA autocast FP16."""
    if precision == "fp32":
        return operation()
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("FP16 segmentation inference requires a CUDA device.")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        return operation()


def _json_safe_config_value(value: Any) -> Any:
    """Convert tuple-like config values into stable JSON report values."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
