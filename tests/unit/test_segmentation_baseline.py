"""Tests for PRD 3.1.1 segmentation baseline helpers."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

import fashion_semantic_parser.service.segmentation_baseline as segmentation_module
from fashion_semantic_parser.common.exceptions import ModelNotReadyError
from fashion_semantic_parser.models.segmentation import SegmentationSubjectROI
from fashion_semantic_parser.service.segmentation_baseline import (
    Detectron2SegmentationBaseline,
    SegmentationBaselineSettings,
    _apply_mask2former_class_loss_weights,
    _configure_mask2former_eager_losses,
    _crop_image_to_subject_roi,
    _json_safe_config_value,
    _latency_summary,
    _Mask2FormerMixedMaskDatasetMapper,
    _masks_to_arrays,
    _run_predictor_with_precision,
    _run_with_precision,
    _selection_for_instance_field,
    convert_detectron2_instances,
    filter_prediction_by_subject_roi,
)
from fashion_semantic_parser.service.segmentation_metrics import (
    _coco_ap_at_iou,
    _coco_matched_mask_iou_metrics,
    _greedy_match_ious,
    _summarize_mask_iou_matches,
)


class _FakeBoxes:
    """Minimal Detectron2 Boxes stand-in."""

    def __init__(self, tensor: list[list[float]]) -> None:
        self.tensor = tensor


class _FakeInstances:
    """Minimal Detectron2 Instances stand-in."""

    def __init__(self) -> None:
        self.pred_boxes = _FakeBoxes([[10.0, 20.0, 110.0, 220.0]])
        self.scores = [0.91]
        self.pred_classes = [0]
        self.pred_masks = [
            np.array(
                [
                    [False, False, False, False],
                    [False, True, True, False],
                    [False, True, True, False],
                    [False, False, False, False],
                ]
            )
        ]


class _FakeZeroBoxInstances(_FakeInstances):
    """Detectron2-like instances with an invalid model box."""

    def __init__(self) -> None:
        super().__init__()
        self.pred_boxes = _FakeBoxes([[0.0, 0.0, 0.0, 0.0]])


class _FakeTwoScoreInstances(_FakeInstances):
    """Detectron2-like instances with one low-score and one high-score item."""

    def __init__(self) -> None:
        super().__init__()
        self.selection_count = 0
        self.pred_boxes = _FakeBoxes(
            [
                [0.0, 0.0, 4.0, 4.0],
                [10.0, 20.0, 110.0, 220.0],
            ]
        )
        self.scores = np.array([0.05, 0.91])
        self.pred_classes = np.array([0, 4])
        self.pred_masks = np.array(
            [
                np.array(
                    [
                        [True, True, False, False],
                        [True, True, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ),
                np.array(
                    [
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ]
                ),
            ]
        )

    def has(self, field_name: str) -> bool:
        """Return whether the fake exposes a Detectron2 field."""
        return hasattr(self, field_name)

    def __getitem__(self, selection: Any) -> Any:
        """Return a score-filtered fake instance batch."""
        self.selection_count += 1
        selected = np.asarray(selection, dtype=bool)
        filtered = object.__new__(_FakeTwoScoreInstances)
        filtered.selection_count = 0
        filtered.pred_boxes = _FakeBoxes(
            np.asarray(self.pred_boxes.tensor)[selected].tolist()
        )
        filtered.scores = self.scores[selected]
        filtered.pred_classes = self.pred_classes[selected]
        filtered.pred_masks = self.pred_masks[selected]
        return filtered


class _NumpyOnlyMaskBatch:
    """Mask batch that forbids expensive nested-list conversion."""

    def numpy(self) -> np.ndarray:
        """Return one dense binary mask."""
        return np.ones((1, 4, 4), dtype=bool)

    def tolist(self) -> Any:
        """Fail if production code converts every pixel to Python objects."""
        raise AssertionError("mask batch should remain a dense array")


class _FakeDeviceSelection:
    """Tensor-like selection that records device moves."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def to(self, *, device: str) -> Any:
        """Return a selection placed on the requested fake device."""
        return _FakeDeviceSelection(device=device)


class _FakeAutocast:
    """Context manager recording an FP16 autocast scope."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self) -> None:
        self.events.append("enter_fp16")

    def __exit__(self, *args: Any) -> None:
        self.events.append("exit_fp16")


class _FakeCuda:
    """Available CUDA stand-in for precision tests."""

    @staticmethod
    def is_available() -> bool:
        """Report CUDA availability."""
        return True


class _FakeTorch:
    """Minimal torch stand-in exposing CUDA autocast."""

    float16 = "float16"
    cuda = _FakeCuda()

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def autocast(self, *, device_type: str, dtype: Any) -> _FakeAutocast:
        """Record autocast parameters and return its context manager."""
        assert device_type == "cuda"
        assert dtype == self.float16
        return _FakeAutocast(self.events)


class _FakeDefaultTrainer:
    """Minimal trainer base for testing dynamic trainer subclasses."""


class _FakeAppliedTransforms:
    """Identity transforms for mixed-mask mapper tests."""

    @staticmethod
    def apply_segmentation(mask: Any) -> Any:
        """Return an unchanged dense mask."""
        return mask


class _FakeTransformsModule:
    """Detectron2 transform-module stand-in."""

    @staticmethod
    def apply_transform_gens(generators: Any, image: Any) -> tuple[Any, Any]:
        """Return an unchanged image and identity annotation transforms."""
        assert generators == ["resize-and-crop"]
        return image, _FakeAppliedTransforms()


class _FakeGroundTruthMasks:
    """BitMasks-like object returned by Detectron2."""

    tensor = "dense-mask-tensor"

    @staticmethod
    def get_bounding_boxes() -> str:
        """Return mask-derived boxes."""
        return "tight-mask-boxes"


class _FakeMappedInstances:
    """Instances-like object used by the mixed-mask mapper test."""

    def __init__(self) -> None:
        self.gt_masks: Any = _FakeGroundTruthMasks()
        self.gt_boxes: Any = None

    def has(self, field_name: str) -> bool:
        """Return whether a field is present."""
        return hasattr(self, field_name)


class _FakeMixedMaskDetectionUtils:
    """Record the mask format requested by the project mapper."""

    requested_mask_format: str | None = None

    @staticmethod
    def read_image(file_name: str, format: str) -> np.ndarray:
        """Return a deterministic fixture image."""
        assert file_name == "fashionpedia.jpg"
        assert format == "BGR"
        return np.zeros((4, 5, 3), dtype=np.uint8)

    @staticmethod
    def check_image_size(dataset_dict: Any, image: Any) -> None:
        """Accept the fixture dimensions."""
        assert image.shape[:2] == (4, 5)

    @staticmethod
    def transform_instance_annotations(
        annotation: dict[str, Any],
        transforms: Any,
        image_shape: tuple[int, int],
    ) -> dict[str, Any]:
        """Model Detectron2 converting transformed RLE into a dense array."""
        assert image_shape == (4, 5)
        if isinstance(annotation["segmentation"], dict):
            annotation["segmentation"] = np.ones(image_shape, dtype=np.uint8)
        return annotation

    @classmethod
    def annotations_to_instances(
        cls,
        annotations: list[dict[str, Any]],
        image_shape: tuple[int, int],
        *,
        mask_format: str,
    ) -> _FakeMappedInstances:
        """Require the bitmask route for transformed RLE and polygons."""
        cls.requested_mask_format = mask_format
        assert annotations[0]["segmentation"].shape == image_shape
        assert isinstance(annotations[1]["segmentation"], list)
        return _FakeMappedInstances()

    @staticmethod
    def filter_empty_instances(instances: Any) -> Any:
        """Return all fixture instances."""
        return instances


class _FakeTensorModule:
    """Torch stand-in that preserves arrays for assertions."""

    @staticmethod
    def as_tensor(value: Any) -> Any:
        """Return the contiguous NumPy input unchanged."""
        return value


class _FakeCOCOEvaluator:
    """Minimal COCOEvaluator stand-in that records constructor values."""

    def __init__(self, dataset_name: str, output_dir: str) -> None:
        self.dataset_name = dataset_name
        self.output_dir = output_dir


class _FakeCOCOEvalResult:
    """Small pycocotools result stand-in for exact-IoU AP tests."""

    def __init__(self) -> None:
        precision = np.full((2, 3, 2, 1, 1), -1.0)
        precision[1, :, 0, 0, 0] = [0.8, 0.6, 0.4]
        precision[1, :, 1, 0, 0] = [0.3, 0.2, 0.1]
        self.params = SimpleNamespace(iouThrs=np.array([0.5, 0.85]))
        self.eval = {"precision": precision}


class _FakeCOCOMaskIoUResult:
    """Small pycocotools result stand-in for direct mask IoU metrics."""

    def __init__(self) -> None:
        self.params = SimpleNamespace(
            catIds=[1, 2],
            imgIds=[10],
            maxDets=[1, 10, 100],
        )
        self._gts = {
            (10, 1): [
                {"id": 1, "area": 400},
                {"id": 2, "area": 2500},
                {"id": 3, "area": 10000},
            ],
            (10, 2): [],
        }
        self._dts = {
            (10, 1): [{"id": 11}, {"id": 12}, {"id": 13}],
            (10, 2): [{"id": 21}],
        }
        self.ious = {
            (10, 1): np.array(
                [
                    [0.90, 0.10, 0.05],
                    [0.80, 0.70, 0.20],
                    [0.10, 0.20, 0.30],
                ]
            ),
            (10, 2): [],
        }


def test_segmentation_baseline_settings_defaults() -> None:
    """Default baseline config should target all PRD 3.1.1 categories."""
    settings = SegmentationBaselineSettings()

    assert settings.num_classes == 8
    assert settings.train_json.endswith("deepfashion2_train.json")
    assert settings.additional_train_jsons == []
    assert settings.train_source_repeat_factors is None
    assert settings.repeat_factor_threshold is None
    assert settings.val_json.endswith("deepfashion2_validation.json")
    assert settings.model_family == "mask_rcnn"
    assert settings.device == "cuda"
    assert settings.checkpoint_period == 1000
    assert settings.eval_period == 0
    assert settings.min_size_test is None
    assert settings.max_size_test is None
    assert settings.detections_per_image is None
    assert settings.category_score_thresholds == {}
    assert settings.class_loss_weights == {}
    assert settings.precision == "fp32"
    assert settings.resume is False
    assert settings.evaluate_after_training is True
    assert settings.resolved_category_names() == (
        "top",
        "pants",
        "skirt",
        "outerwear",
        "dress",
        "shoes",
        "bag",
        "accessory",
    )


def test_mixed_training_settings_require_one_positive_factor_per_source() -> None:
    """Invalid source balancing must fail before a long GPU run starts."""
    with pytest.raises(ValueError, match="one value per training COCO"):
        SegmentationBaselineSettings(
            additional_train_jsons=["fashionpedia.json"],
            train_source_repeat_factors=[1.0],
        )
    with pytest.raises(ValueError, match="finite and positive"):
        SegmentationBaselineSettings(train_source_repeat_factors=[0.0])
    with pytest.raises(ValueError, match="must be unique"):
        SegmentationBaselineSettings(
            train_json="same.json",
            additional_train_jsons=["same.json"],
        )
    with pytest.raises(ValueError, match="cannot be enabled together"):
        SegmentationBaselineSettings(
            train_source_repeat_factors=[1.0],
            repeat_factor_threshold=0.01,
        )


def test_custom_segmentation_taxonomy_must_match_model_head() -> None:
    """A non-default model head needs ordered, unique category labels."""
    settings = SegmentationBaselineSettings(
        num_classes=2,
        category_names=["collar", "pocket"],
    )

    assert settings.resolved_category_names() == ("collar", "pocket")
    with pytest.raises(ValueError, match="exactly num_classes"):
        SegmentationBaselineSettings(
            num_classes=2,
            category_names=["collar"],
        )
    with pytest.raises(ValueError, match="unique"):
        SegmentationBaselineSettings(
            num_classes=2,
            category_names=["collar", "collar"],
        )
    with pytest.raises(ValueError, match="unknown categories"):
        SegmentationBaselineSettings(category_score_thresholds={"shoe": 0.3})


def test_category_thresholds_lower_model_output_floor() -> None:
    """The model must retain small-class candidates for category filtering."""
    settings = SegmentationBaselineSettings(
        score_threshold=0.6,
        category_score_thresholds={"shoes": 0.4, "bag": 0.3},
    )

    assert settings.model_score_threshold() == 0.3


def test_class_loss_weights_require_known_positive_mask2former_categories() -> None:
    """Invalid class-weighted losses must fail before a GPU run starts."""
    with pytest.raises(ValueError, match="unknown categories"):
        SegmentationBaselineSettings(
            model_family="mask2former",
            class_loss_weights={"unknown": 2.0},
        )
    with pytest.raises(ValueError, match="finite and positive"):
        SegmentationBaselineSettings(
            model_family="mask2former",
            class_loss_weights={"top": 0.0},
        )
    with pytest.raises(ValueError, match="only for Mask2Former"):
        SegmentationBaselineSettings(class_loss_weights={"top": 2.0})


def test_mixed_training_configures_weighted_source_sampler() -> None:
    """Repeat factors should balance sources without materializing merged JSON."""
    baseline = Detectron2SegmentationBaseline(
        SegmentationBaselineSettings(
            additional_train_jsons=["fashionpedia.json"],
            train_source_repeat_factors=[1.0, 4.3],
        )
    )
    cfg = SimpleNamespace(
        DATASETS=SimpleNamespace(TRAIN=()),
        DATALOADER=SimpleNamespace(SAMPLER_TRAIN="TrainingSampler"),
    )

    baseline._apply_training_dataset_settings(cfg)

    assert cfg.DATASETS.TRAIN == (
        "prd_3_1_1_deepfashion2_train",
        "prd_3_1_1_additional_train_1",
    )
    assert cfg.DATALOADER.SAMPLER_TRAIN == "WeightedTrainingSampler"
    assert cfg.DATASETS.TRAIN_REPEAT_FACTOR == (
        ("prd_3_1_1_deepfashion2_train", 1.0),
        ("prd_3_1_1_additional_train_1", 4.3),
    )


def test_single_source_training_can_repeat_rare_category_images() -> None:
    """Part training should use Detectron2's category-frequency sampler."""
    baseline = Detectron2SegmentationBaseline(
        SegmentationBaselineSettings(repeat_factor_threshold=0.01)
    )
    cfg = SimpleNamespace(
        DATASETS=SimpleNamespace(TRAIN=()),
        DATALOADER=SimpleNamespace(
            SAMPLER_TRAIN="TrainingSampler",
            REPEAT_THRESHOLD=0.0,
        ),
    )

    baseline._apply_training_dataset_settings(cfg)

    assert cfg.DATALOADER.SAMPLER_TRAIN == "RepeatFactorTrainingSampler"
    assert cfg.DATALOADER.REPEAT_THRESHOLD == 0.01


def test_register_datasets_includes_every_mixed_training_source(
    monkeypatch: Any,
) -> None:
    """Each mixed COCO source must receive its own Detectron2 registration."""
    registrations: list[tuple[str, str]] = []

    def register(
        name: str,
        metadata: dict[str, Any],
        json_path: str,
        image_root: str,
    ) -> None:
        assert len(metadata["thing_classes"]) == 8
        assert image_root == "."
        registrations.append((name, json_path))

    monkeypatch.setattr(
        segmentation_module,
        "_load_detectron2_modules",
        lambda: {"register_coco_instances": register},
    )
    baseline = Detectron2SegmentationBaseline(
        SegmentationBaselineSettings(
            train_json="deepfashion2.json",
            additional_train_jsons=["fashionpedia.json"],
            val_json="validation.json",
        )
    )

    baseline.register_datasets()

    assert registrations == [
        ("prd_3_1_1_deepfashion2_train", "deepfashion2.json"),
        ("prd_3_1_1_additional_train_1", "fashionpedia.json"),
        ("prd_3_1_1_deepfashion2_validation", "validation.json"),
    ]


def test_register_datasets_uses_configured_category_names(
    monkeypatch: Any,
) -> None:
    """The reusable trainer must not force the eight garment labels."""
    registered_classes: list[str] = []

    def register(
        name: str,
        metadata: dict[str, Any],
        json_path: str,
        image_root: str,
    ) -> None:
        registered_classes.extend(metadata["thing_classes"])

    monkeypatch.setattr(
        segmentation_module,
        "_load_detectron2_modules",
        lambda: {"register_coco_instances": register},
    )
    baseline = Detectron2SegmentationBaseline(
        SegmentationBaselineSettings(
            num_classes=2,
            category_names=["collar", "pocket"],
        )
    )

    baseline.register_datasets()

    assert registered_classes == ["collar", "pocket"] * 2


def test_mask2former_settings_use_local_project_config() -> None:
    """Mask2Former settings should represent the PRD target model path."""
    settings = SegmentationBaselineSettings(
        model_family="mask2former",
        config_source="local",
        config_file=(
            "external/Mask2Former/configs/coco/instance-segmentation/"
            "maskformer2_R50_bs16_50ep.yaml"
        ),
        output_dir="outputs/segmentation/mask2former_r50",
        base_lr=0.0001,
    )

    assert settings.model_family == "mask2former"
    assert settings.config_source == "local"
    assert settings.config_file is not None
    assert settings.config_file.endswith("maskformer2_R50_bs16_50ep.yaml")
    assert settings.output_dir.endswith("mask2former_r50")


def test_inference_size_overrides_only_selected_config_values() -> None:
    """Explicit test sizes should replace defaults used by the predictor."""
    baseline = Detectron2SegmentationBaseline(
        SegmentationBaselineSettings(min_size_test=640, max_size_test=1067)
    )
    cfg = SimpleNamespace(INPUT=SimpleNamespace(MIN_SIZE_TEST=800, MAX_SIZE_TEST=1333))

    baseline._apply_inference_size_settings(cfg)

    assert cfg.INPUT.MIN_SIZE_TEST == 640
    assert cfg.INPUT.MAX_SIZE_TEST == 1067


def test_inference_size_defaults_preserve_model_config() -> None:
    """Absent overrides should keep the model family's configured test size."""
    baseline = Detectron2SegmentationBaseline(SegmentationBaselineSettings())
    cfg = SimpleNamespace(
        INPUT=SimpleNamespace(MIN_SIZE_TEST=(640, 800), MAX_SIZE_TEST=1333)
    )

    baseline._apply_inference_size_settings(cfg)

    assert cfg.INPUT.MIN_SIZE_TEST == (640, 800)
    assert cfg.INPUT.MAX_SIZE_TEST == 1333
    assert _json_safe_config_value(cfg.INPUT.MIN_SIZE_TEST) == [640, 800]


def test_detection_limit_overrides_detectron2_default() -> None:
    """Deployment may reduce returned candidates without changing model weights."""
    baseline = Detectron2SegmentationBaseline(
        SegmentationBaselineSettings(detections_per_image=20)
    )
    cfg = SimpleNamespace(
        TEST=SimpleNamespace(DETECTIONS_PER_IMAGE=100),
        MODEL=SimpleNamespace(),
    )

    baseline._apply_model_head_settings(cfg)

    assert cfg.TEST.DETECTIONS_PER_IMAGE == 20


def test_inference_predictor_is_initialized_once(monkeypatch: Any) -> None:
    """API requests should reuse one loaded Detectron2 predictor."""
    created_predictors = []
    expected_predictor = object()
    baseline = Detectron2SegmentationBaseline(SegmentationBaselineSettings())
    monkeypatch.setattr(baseline, "build_config", lambda: "test-config")

    def build_predictor(config: Any) -> Any:
        created_predictors.append(config)
        return expected_predictor

    modules = {"DefaultPredictor": build_predictor}

    assert baseline._get_predictor(modules) is expected_predictor
    assert baseline._get_predictor(modules) is expected_predictor
    assert created_predictors == ["test-config"]


def test_mask2former_project_config_uses_pretrained_weights() -> None:
    """Short Mask2Former fine-tuning should start from COCO pretrained weights."""
    config_path = Path("configs/segmentation_mask2former.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model_family"] == "mask2former"
    assert config["weights"].endswith("model_final_3c8ec9.pkl")
    assert "coco/instance/maskformer2_R50_bs16_50ep" in config["weights"]
    assert config["max_iter"] == 20000
    assert config["ims_per_batch"] == 4
    assert config["base_lr"] == 0.000025
    assert config["checkpoint_period"] == 1000
    assert config["resume"] is False
    assert config["evaluate_after_training"] is False


def test_mask2former_deployment_config_records_validated_profile() -> None:
    """Deployment config should preserve the selected eight-class profile."""
    config_path = Path("configs/segmentation_mask2former_deployment.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model_family"] == "mask2former"
    assert config["weights"].endswith("model_0001999.pth")
    assert config["val_json"].endswith("fashionpedia_validation.json")
    assert config["num_classes"] == 8
    assert config["score_threshold"] == 0.6
    assert config["min_size_test"] == 512
    assert config["max_size_test"] == 853
    assert "detections_per_image" not in config
    assert config["precision"] == "fp16"
    assert config["device"] == "cuda"


def test_mask2former_small_object_config_preserves_baseline_checkpoint() -> None:
    """Small-object experiments should be isolated from the accepted profile."""
    config_path = Path("configs/segmentation_mask2former_small_objects.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["weights"].endswith("model_0001999.pth")
    assert config["score_threshold"] == 0.6
    assert config["category_score_thresholds"] == {
        "shoes": 0.4,
        "bag": 0.3,
        "accessory": 0.3,
    }
    assert config["min_size_test"] == 640
    assert config["max_size_test"] == 1067
    SegmentationBaselineSettings.model_validate(config)


def test_small_object_finetune_config_adds_targeted_training_source() -> None:
    """Fine-tuning should repeat targeted records without dropping either corpus."""
    config_path = Path("configs/segmentation_mask2former_small_object_finetune.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["weights"].endswith("model_0001999.pth")
    assert config["additional_train_jsons"] == [
        "data/processed/autodl/segmentation/fashionpedia_train.json",
        "data/processed/autodl/segmentation/fashionpedia_train_small_objects.json",
    ]
    assert config["train_source_repeat_factors"] == [1.0, 4.3, 4.3]
    assert config["base_lr"] == 0.0000025
    assert config["max_iter"] == 1000
    assert config["evaluate_after_training"] is False
    SegmentationBaselineSettings.model_validate(config)


def test_mask2former_fashionpedia_config_is_isolated_transfer_stage() -> None:
    """Transfer training should stay isolated until mixed consolidation."""
    config_path = Path("configs/segmentation_mask2former_fashionpedia.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model_family"] == "mask2former"
    assert config["train_json"].endswith("fashionpedia_train.json")
    assert config["val_json"].endswith("fashionpedia_validation.json")
    assert config["weights"].endswith("model_official_0004999.pth")
    assert config["num_classes"] == 8
    assert config["base_lr"] == 0.00001
    assert config["max_iter"] == 10000
    assert config["output_dir"].endswith("fashionpedia/mask2former_r50_stage1")
    assert config["score_threshold"] == 0.0
    assert config["evaluate_after_training"] is False


def test_mask2former_mixed_config_balances_both_training_sources() -> None:
    """Consolidation should preserve old classes while retaining new classes."""
    config_path = Path("configs/segmentation_mask2former_mixed.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["train_json"].endswith("deepfashion2_train.json")
    assert config["additional_train_jsons"] == [
        "data/processed/autodl/segmentation/fashionpedia_train.json"
    ]
    assert config["train_source_repeat_factors"] == [1.0, 4.3]
    assert config["weights"].endswith("model_0000999.pth")
    assert config["base_lr"] == 0.000005
    assert config["resume"] is False


def test_localization_parts_config_uses_supervised_nineteen_class_masks() -> None:
    """PRD 3.1.2 should have an isolated, class-balanced training profile."""
    config_path = Path("configs/localization_mask2former_parts.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model_family"] == "mask2former"
    assert config["train_json"].endswith("fashionpedia_parts_train.json")
    assert config["val_json"].endswith("fashionpedia_parts_validation.json")
    assert config["weights"].endswith("model_0001999.pth")
    assert config["num_classes"] == 19
    assert len(config["category_names"]) == 19
    assert config["category_names"][0] == "hood"
    assert config["category_names"][-1] == "tassel"
    assert config["repeat_factor_threshold"] == 0.01
    assert config["mask2former_eager_losses"] is True
    assert config["base_lr"] == 0.00001
    assert config["evaluate_after_training"] is False
    SegmentationBaselineSettings.model_validate(config)


def test_localization_long_tail_config_preserves_stage_one_checkpoint() -> None:
    """Rare-class tuning should be isolated and use stronger repeat sampling."""
    config_path = Path("configs/localization_mask2former_parts_long_tail.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["weights"].endswith("mask2former_parts_r50_10000.pth")
    assert config["repeat_factor_threshold"] == 0.05
    assert config["base_lr"] == 0.000005
    assert config["max_iter"] == 5000
    assert config["checkpoint_period"] == 1000
    assert config["resume"] is False
    assert config["evaluate_after_training"] is False
    assert len(config["category_names"]) == 19
    SegmentationBaselineSettings.model_validate(config)


def test_localization_targeted_config_replays_critical_classes() -> None:
    """Targeted tuning should replay weak classes from the long-tail checkpoint."""
    config_path = Path("configs/localization_mask2former_parts_targeted.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["additional_train_jsons"] == [
        "data/processed/autodl/localization/"
        "fashionpedia_parts_train_critical_long_tail.json"
    ]
    assert config["train_source_repeat_factors"] == [1.0, 2.0]
    assert config["weights"].endswith("mask2former_parts_r50_long_tail_5000.pth")
    assert config["base_lr"] == 0.0000025
    assert config["max_iter"] == 3000
    assert config["evaluate_after_training"] is False
    SegmentationBaselineSettings.model_validate(config)


def test_localization_class_weighted_config_continues_targeted_checkpoint() -> None:
    """Class weighting should isolate a conservative post-targeted stage."""
    config_path = Path("configs/localization_mask2former_parts_class_weighted.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["weights"].endswith("mask2former_parts_r50_targeted_3000.pth")
    assert config["train_source_repeat_factors"] == [1.0, 2.0]
    assert config["class_loss_weights"] == {
        "buckle": 1.5,
        "bow": 2.0,
        "ribbon": 2.5,
        "rivet": 1.5,
        "tassel": 3.0,
    }
    assert config["base_lr"] == 0.000001
    assert config["max_iter"] == 3000
    SegmentationBaselineSettings.model_validate(config)


def test_localization_parts_deployment_uses_selected_checkpoint() -> None:
    """The API profile should freeze the selected 10,000-iteration model."""
    config_path = Path("configs/localization_mask2former_parts_deployment.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["weights"].endswith("mask2former_parts_r50_10000.pth")
    assert config["num_classes"] == 19
    assert len(config["category_names"]) == 19
    assert config["score_threshold"] == 0.25
    assert config["subject_roi_margin"] == 0.35
    assert config["precision"] == "fp16"
    SegmentationBaselineSettings.model_validate(config)


def test_mask2former_eager_losses_replace_scripted_wrappers() -> None:
    """The compatibility path must retain upstream eager loss definitions."""

    def matcher_dice() -> str:
        return "matcher-dice"

    def matcher_ce() -> str:
        return "matcher-ce"

    def criterion_dice() -> str:
        return "criterion-dice"

    def criterion_ce() -> str:
        return "criterion-ce"

    matcher_module = SimpleNamespace(
        batch_dice_loss=matcher_dice,
        batch_dice_loss_jit=object(),
        batch_sigmoid_ce_loss=matcher_ce,
        batch_sigmoid_ce_loss_jit=object(),
    )
    criterion_module = SimpleNamespace(
        dice_loss=criterion_dice,
        dice_loss_jit=object(),
        sigmoid_ce_loss=criterion_ce,
        sigmoid_ce_loss_jit=object(),
    )

    _configure_mask2former_eager_losses(matcher_module, criterion_module)

    assert matcher_module.batch_dice_loss_jit is matcher_dice
    assert matcher_module.batch_sigmoid_ce_loss_jit is matcher_ce
    assert criterion_module.dice_loss_jit is criterion_dice
    assert criterion_module.sigmoid_ce_loss_jit is criterion_ce


def test_mask2former_mixed_mask_mapper_accepts_rle_and_polygons() -> None:
    """Fashionpedia mask formats must share Detectron2's bitmask path."""
    upstream_mapper = SimpleNamespace(
        img_format="BGR",
        tfm_gens=["resize-and-crop"],
        is_train=True,
    )
    mapper = _Mask2FormerMixedMaskDatasetMapper(
        upstream_mapper,
        detection_utils=_FakeMixedMaskDetectionUtils,
        transforms=_FakeTransformsModule,
        torch=_FakeTensorModule,
    )
    source = {
        "file_name": "fashionpedia.jpg",
        "height": 4,
        "width": 5,
        "annotations": [
            {
                "bbox": [0, 0, 3, 3],
                "category_id": 5,
                "iscrowd": 0,
                "keypoints": [1, 1, 2],
                "segmentation": {"size": [4, 5], "counts": "encoded"},
            },
            {
                "bbox": [1, 1, 3, 3],
                "category_id": 7,
                "iscrowd": 0,
                "segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]],
            },
        ],
    }

    mapped = mapper(source)

    assert _FakeMixedMaskDetectionUtils.requested_mask_format == "bitmask"
    assert mapped["instances"].gt_boxes == "tight-mask-boxes"
    assert mapped["instances"].gt_masks == "dense-mask-tensor"
    assert mapped["image"].shape == (3, 4, 5)
    assert mapped["padding_mask"].shape == (4, 5)
    assert isinstance(source["annotations"][0]["segmentation"], dict)


def test_mask2former_trainer_uses_target_optimizer_and_scheduler(
    monkeypatch: Any,
) -> None:
    """Mask2Former must not fall back to Detectron2's default SGD trainer."""
    expected_optimizer = object()
    expected_scheduler = object()

    def eager_matcher_dice() -> None:
        return None

    def eager_matcher_ce() -> None:
        return None

    def eager_criterion_dice() -> None:
        return None

    def eager_criterion_ce() -> None:
        return None

    matcher_module = SimpleNamespace(
        batch_dice_loss=eager_matcher_dice,
        batch_dice_loss_jit=object(),
        batch_sigmoid_ce_loss=eager_matcher_ce,
        batch_sigmoid_ce_loss_jit=object(),
    )
    criterion_module = SimpleNamespace(
        dice_loss=eager_criterion_dice,
        dice_loss_jit=object(),
        sigmoid_ce_loss=eager_criterion_ce,
        sigmoid_ce_loss_jit=object(),
    )
    monkeypatch.setattr(
        segmentation_module,
        "_build_mask2former_optimizer",
        lambda cfg, model, clipper: expected_optimizer,
    )
    monkeypatch.setattr(
        segmentation_module,
        "_load_mask2former_modules",
        lambda: {
            "COCOInstanceNewBaselineDatasetMapper": object,
            "build_lr_scheduler": lambda cfg, optimizer: expected_scheduler,
            "criterion_module": criterion_module,
            "matcher_module": matcher_module,
            "maybe_add_gradient_clipping": object(),
        },
    )
    baseline = Detectron2SegmentationBaseline(
        SegmentationBaselineSettings(
            model_family="mask2former",
            mask2former_eager_losses=True,
        )
    )
    trainer_class = baseline._trainer_class(
        {
            "COCOEvaluator": _FakeCOCOEvaluator,
            "DefaultTrainer": _FakeDefaultTrainer,
            "BitMasks": None,
            "build_detection_train_loader": object(),
        }
    )

    assert trainer_class.build_optimizer(object(), object()) is expected_optimizer
    assert trainer_class.build_lr_scheduler(object(), object()) is expected_scheduler
    assert matcher_module.batch_dice_loss_jit is eager_matcher_dice
    assert matcher_module.batch_sigmoid_ce_loss_jit is eager_matcher_ce
    assert criterion_module.dice_loss_jit is eager_criterion_dice
    assert criterion_module.sigmoid_ce_loss_jit is eager_criterion_ce


def test_mask2former_class_loss_weights_preserve_other_and_no_object_weights() -> None:
    """Named weights must align with classes and leave no-object unchanged."""

    class FakeTensor:
        def __init__(self, values: list[float]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __setitem__(self, index: int, value: float) -> None:
            self.values[index] = value

        def detach(self) -> "FakeTensor":
            return self

        def clone(self) -> "FakeTensor":
            return FakeTensor(self.values.copy())

        def copy_(self, other: "FakeTensor") -> None:
            self.values = other.values.copy()

    weights = FakeTensor([1.0, 1.0, 1.0, 0.1])
    model = SimpleNamespace(criterion=SimpleNamespace(empty_weight=weights))

    _apply_mask2former_class_loss_weights(
        model,
        category_names=("collar", "ribbon", "tassel"),
        class_loss_weights={"ribbon": 2.5, "tassel": 3.0},
    )

    assert weights.values == [1.0, 2.5, 3.0, 0.1]


def test_mask2former_class_loss_weights_validate_model_class_count() -> None:
    """A stale model head must not receive misaligned category weights."""

    class FakeInvalidTensor:
        def __len__(self) -> int:
            return 2

    weights = FakeInvalidTensor()
    model = SimpleNamespace(criterion=SimpleNamespace(empty_weight=weights))

    with pytest.raises(ModelNotReadyError, match="class count"):
        _apply_mask2former_class_loss_weights(
            model,
            category_names=("collar", "ribbon"),
            class_loss_weights={"ribbon": 2.0},
        )


def test_mask2former_trainer_applies_configured_class_loss_weights(
    monkeypatch: Any,
) -> None:
    """The dynamic trainer must apply weights immediately after model build."""
    captured: dict[str, Any] = {}
    model = object()

    class FakeModelTrainer:
        @classmethod
        def build_model(cls, cfg: Any) -> Any:
            captured["cfg"] = cfg
            return model

    monkeypatch.setattr(
        segmentation_module,
        "_load_mask2former_modules",
        lambda: {
            "COCOInstanceNewBaselineDatasetMapper": object,
            "build_lr_scheduler": object(),
            "criterion_module": object(),
            "matcher_module": object(),
            "maybe_add_gradient_clipping": object(),
        },
    )
    monkeypatch.setattr(
        segmentation_module,
        "_apply_mask2former_class_loss_weights",
        lambda built_model, **kwargs: captured.update(
            model=built_model,
            **kwargs,
        ),
    )
    baseline = Detectron2SegmentationBaseline(
        SegmentationBaselineSettings(
            model_family="mask2former",
            class_loss_weights={"bag": 2.0},
        )
    )
    trainer_class = baseline._trainer_class(
        {
            "COCOEvaluator": _FakeCOCOEvaluator,
            "DefaultTrainer": FakeModelTrainer,
            "BitMasks": None,
            "build_detection_train_loader": object(),
        }
    )
    cfg = object()

    assert trainer_class.build_model(cfg) is model
    assert captured == {
        "cfg": cfg,
        "model": model,
        "category_names": baseline.settings.resolved_category_names(),
        "class_loss_weights": {"bag": 2.0},
    }


def test_trainer_class_builds_coco_evaluator() -> None:
    """Training should report COCO instance metrics after validation."""
    baseline = Detectron2SegmentationBaseline(SegmentationBaselineSettings())
    trainer_class = baseline._trainer_class(
        {
            "COCOEvaluator": _FakeCOCOEvaluator,
            "DefaultTrainer": _FakeDefaultTrainer,
            "BitMasks": None,
        }
    )

    evaluator = trainer_class.build_evaluator(
        SimpleNamespace(OUTPUT_DIR="outputs/segmentation/test"),
        "validation_dataset",
    )

    assert evaluator.dataset_name == "validation_dataset"
    assert evaluator.output_dir == "outputs/segmentation/test/inference"


def test_coco_ap_at_exact_iou_threshold() -> None:
    """PRD evaluation should expose aggregate and per-category AP at IoU 0.85."""
    coco_eval = _FakeCOCOEvalResult()

    assert np.isclose(_coco_ap_at_iou(coco_eval, 0.85), 40.0)
    assert np.isclose(_coco_ap_at_iou(coco_eval, 0.85, category_index=0), 60.0)
    assert np.isclose(_coco_ap_at_iou(coco_eval, 0.85, category_index=1), 20.0)
    assert np.isnan(_coco_ap_at_iou(coco_eval, 0.90))


def test_greedy_mask_iou_matching_is_one_to_one() -> None:
    """Direct mask IoU evaluation should not reuse predictions or ground truth."""
    matched_ious = _greedy_match_ious(
        np.array(
            [
                [0.90, 0.10],
                [0.80, 0.70],
            ]
        )
    )

    assert matched_ious == [0.90, 0.70]


def test_mask_iou_summary_counts_unmatched_ground_truth_as_zero() -> None:
    """All-GT IoU should expose misses that matched-only mean IoU hides."""
    summary = _summarize_mask_iou_matches(
        matched_ious=[0.90, 0.70],
        ground_truth_count=3,
        prediction_count=4,
    )

    assert np.isclose(summary["MatchedMeanIoU"], 80.0)
    assert np.isclose(summary["AllGTMeanIoU"], 160.0 / 3.0)
    assert np.isclose(summary["Precision50"], 50.0)
    assert np.isclose(summary["Recall50"], 200.0 / 3.0)
    assert np.isclose(summary["MatchedIoU85Rate"], 50.0)
    assert np.isclose(summary["AllGTIoU85Rate"], 100.0 / 3.0)


def test_coco_matched_mask_iou_metrics_include_per_category_results() -> None:
    """COCO mask evaluation should report direct aggregate and class IoU."""
    results = _coco_matched_mask_iou_metrics(
        _FakeCOCOMaskIoUResult(),
        class_names=["top", "pants"],
    )

    assert results["MatchedCount"] == 2.0
    assert results["GroundTruthCount"] == 3.0
    assert results["PredictionCount"] == 3.0
    assert np.isclose(results["MatchedMeanIoU"], 80.0)
    assert np.isclose(results["AllGTMeanIoU"], 160.0 / 3.0)
    assert np.isclose(results["Recall50-top"], 200.0 / 3.0)
    assert np.isnan(results["MatchedMeanIoU-pants"])
    assert np.isclose(results["MatchedMeanIoU-small"], 90.0)
    assert np.isclose(results["AllGTMeanIoU-medium"], 70.0)
    assert results["GroundTruthCount-large"] == 1.0
    assert results["MatchedCount-large"] == 0.0
    assert np.isclose(results["AllGTMeanIoU-large"], 0.0)


def test_latency_summary_reports_median_tail_and_throughput() -> None:
    """Latency reports should expose both typical and p95 runtime."""
    summary = _latency_summary([10.0, 20.0, 30.0, 40.0])

    assert summary["mean"] == 25.0
    assert summary["median"] == 25.0
    assert np.isclose(summary["p95"], 38.5)
    assert summary["min"] == 10.0
    assert summary["max"] == 40.0
    assert summary["fps_from_mean"] == 40.0


def test_fp16_predictor_runs_inside_cuda_autocast() -> None:
    """FP16 latency tests should autocast only the predictor call."""
    events: list[str] = []

    def predictor(image: Any) -> Any:
        events.append("predict")
        return image

    output = _run_predictor_with_precision(
        predictor=predictor,
        image="image",
        torch=_FakeTorch(events),
        device="cuda",
        precision="fp16",
    )

    assert output == "image"
    assert events == ["enter_fp16", "predict", "exit_fp16"]


def test_fp16_evaluation_operation_runs_inside_cuda_autocast() -> None:
    """Full evaluation should use the same FP16 autocast path as prediction."""
    events: list[str] = []

    def evaluate() -> str:
        events.append("evaluate")
        return "metrics"

    output = _run_with_precision(
        operation=evaluate,
        torch=_FakeTorch(events),
        device="cuda",
        precision="fp16",
    )

    assert output == "metrics"
    assert events == ["enter_fp16", "evaluate", "exit_fp16"]


def test_convert_detectron2_instances_to_prediction_schema() -> None:
    """Detectron2-like instances should become project prediction objects."""
    prediction = convert_detectron2_instances(
        instances=_FakeInstances(),
        image_path="data/raw/example.jpg",
    )

    assert prediction.image_path == "data/raw/example.jpg"
    assert len(prediction.instances) == 1
    instance = prediction.instances[0]
    assert instance.category_id == 1
    assert instance.category_label == "top"
    assert instance.confidence == 0.91
    assert instance.box.x_min == 10.0
    assert instance.box.y_min == 20.0
    assert instance.box.x_max == 110.0
    assert instance.box.y_max == 220.0
    assert len(instance.mask) == 1
    assert len(instance.mask[0]) >= 6


def test_convert_detectron2_instances_uses_custom_taxonomy() -> None:
    """Part-model inference should return its configured category labels."""
    prediction = convert_detectron2_instances(
        instances=_FakeInstances(),
        image_path="data/raw/example.jpg",
        category_names=("collar", "pocket"),
    )

    assert prediction.instances[0].category_id == 1
    assert prediction.instances[0].category_label == "collar"


def test_convert_detectron2_instances_derives_invalid_box_from_mask() -> None:
    """Mask-first models should still return usable PRD bounding boxes."""
    prediction = convert_detectron2_instances(
        instances=_FakeZeroBoxInstances(),
        image_path="data/raw/example.jpg",
    )

    instance = prediction.instances[0]
    assert instance.box.x_min == 1.0
    assert instance.box.y_min == 1.0
    assert instance.box.x_max == 3.0
    assert instance.box.y_max == 3.0


def test_convert_detectron2_instances_filters_low_scores() -> None:
    """Prediction conversion should honor explicit score thresholds."""
    instances = _FakeTwoScoreInstances()
    prediction = convert_detectron2_instances(
        instances=instances,
        image_path="data/raw/example.jpg",
        score_threshold=0.1,
    )

    assert instances.selection_count == 1
    assert len(prediction.instances) == 1
    assert prediction.instances[0].category_label == "dress"
    assert prediction.instances[0].confidence == 0.91


def test_convert_detectron2_instances_uses_category_score_thresholds() -> None:
    """Small categories may retain candidates below the global threshold."""
    prediction = convert_detectron2_instances(
        instances=_FakeTwoScoreInstances(),
        image_path="data/raw/example.jpg",
        score_threshold=0.6,
        category_score_thresholds={"top": 0.05},
    )

    assert [instance.category_label for instance in prediction.instances] == [
        "top",
        "dress",
    ]


def test_convert_detectron2_instances_maps_crop_coordinates_to_full_image() -> None:
    """ROI predictions should retain original-image box and mask coordinates."""
    prediction = convert_detectron2_instances(
        instances=_FakeInstances(),
        image_path="data/raw/example.jpg",
        coordinate_offset=(50.0, 70.0),
    )

    instance = prediction.instances[0]
    assert instance.box.model_dump() == {
        "x_min": 60.0,
        "y_min": 90.0,
        "x_max": 160.0,
        "y_max": 290.0,
    }
    polygon = instance.mask[0]
    assert min(polygon[0::2]) >= 50.0
    assert min(polygon[1::2]) >= 70.0


def test_crop_image_to_subject_roi_expands_and_clamps_bounds() -> None:
    """ROI crop should add context without leaving the source image."""
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    crop, offset = _crop_image_to_subject_roi(
        image,
        SegmentationSubjectROI(
            x_min=50.0,
            y_min=20.0,
            x_max=100.0,
            y_max=80.0,
        ),
        margin=0.1,
    )

    assert crop.shape == (72, 60, 3)
    assert offset == (45.0, 14.0)


def test_crop_image_to_subject_roi_rejects_non_overlapping_region() -> None:
    """An ROI outside the image cannot produce a usable model input."""
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="does not overlap"):
        _crop_image_to_subject_roi(
            image,
            SegmentationSubjectROI(
                x_min=210.0,
                y_min=20.0,
                x_max=260.0,
                y_max=80.0,
            ),
            margin=0.0,
        )


def test_mask_conversion_keeps_pixels_in_dense_arrays() -> None:
    """Output conversion should not materialize masks as nested Python lists."""
    masks = _masks_to_arrays(_NumpyOnlyMaskBatch())

    assert len(masks) == 1
    assert isinstance(masks[0], np.ndarray)
    assert masks[0].dtype == np.bool_


def test_instance_selection_uses_each_field_device() -> None:
    """Mixed CPU/GPU Detectron2 fields need device-local boolean indices."""
    selection = _FakeDeviceSelection(device="cuda")

    tensor_selection = _selection_for_instance_field(
        selection,
        SimpleNamespace(device="cuda"),
    )
    boxes_selection = _selection_for_instance_field(
        selection,
        SimpleNamespace(tensor=SimpleNamespace(device="cpu")),
    )

    assert tensor_selection.device == "cuda"
    assert boxes_selection.device == "cpu"


def test_filter_prediction_by_subject_roi_removes_background_instances() -> None:
    """Subject ROI filtering should remove predictions outside the model area."""
    prediction = convert_detectron2_instances(
        instances=_FakeTwoScoreInstances(),
        image_path="data/raw/example.jpg",
        score_threshold=0.0,
    )

    filtered = filter_prediction_by_subject_roi(
        prediction,
        SegmentationSubjectROI(
            x_min=8.0,
            y_min=18.0,
            x_max=120.0,
            y_max=230.0,
        ),
    )

    assert len(filtered.instances) == 1
    assert filtered.instances[0].category_label == "dress"


def test_filter_prediction_by_subject_roi_removes_off_center_large_instances() -> None:
    """Large background predictions should not survive by tiny ROI overlap alone."""
    prediction = convert_detectron2_instances(
        instances=_FakeTwoScoreInstances(),
        image_path="data/raw/example.jpg",
        score_threshold=0.0,
    )

    filtered = filter_prediction_by_subject_roi(
        prediction,
        SegmentationSubjectROI(
            x_min=3.0,
            y_min=3.0,
            x_max=20.0,
            y_max=20.0,
        ),
    )

    assert filtered.instances == []
