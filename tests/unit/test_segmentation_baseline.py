"""Tests for PRD 3.1.1 segmentation baseline helpers."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from fashion_semantic_parser.service.segmentation_baseline import (
    Detectron2SegmentationBaseline,
    SegmentationBaselineSettings,
    convert_detectron2_instances,
    filter_prediction_by_subject_roi,
)
from fashion_semantic_parser.models.segmentation import SegmentationSubjectROI


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
        self.pred_boxes = _FakeBoxes(
            [
                [0.0, 0.0, 4.0, 4.0],
                [10.0, 20.0, 110.0, 220.0],
            ]
        )
        self.scores = [0.05, 0.91]
        self.pred_classes = [0, 4]
        self.pred_masks = [
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


class _FakeDefaultTrainer:
    """Minimal trainer base for testing dynamic trainer subclasses."""


class _FakeCOCOEvaluator:
    """Minimal COCOEvaluator stand-in that records constructor values."""

    def __init__(self, dataset_name: str, output_dir: str) -> None:
        self.dataset_name = dataset_name
        self.output_dir = output_dir


def test_segmentation_baseline_settings_defaults() -> None:
    """Default baseline config should target all PRD 3.1.1 categories."""
    settings = SegmentationBaselineSettings()

    assert settings.num_classes == 8
    assert settings.train_json.endswith("deepfashion2_train.json")
    assert settings.val_json.endswith("deepfashion2_validation.json")
    assert settings.model_family == "mask_rcnn"
    assert settings.device == "cuda"


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


def test_mask2former_project_config_uses_pretrained_weights() -> None:
    """Short Mask2Former fine-tuning should start from COCO pretrained weights."""
    config_path = Path("configs/segmentation_mask2former.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model_family"] == "mask2former"
    assert config["weights"].endswith("model_final_3c8ec9.pkl")
    assert "coco/instance/maskformer2_R50_bs16_50ep" in config["weights"]


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
    prediction = convert_detectron2_instances(
        instances=_FakeTwoScoreInstances(),
        image_path="data/raw/example.jpg",
        score_threshold=0.1,
    )

    assert len(prediction.instances) == 1
    assert prediction.instances[0].category_label == "dress"
    assert prediction.instances[0].confidence == 0.91


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
