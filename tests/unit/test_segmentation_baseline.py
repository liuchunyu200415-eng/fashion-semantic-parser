"""Tests for PRD 3.1.1 segmentation baseline helpers."""

import numpy as np

from fashion_semantic_parser.service.segmentation_baseline import (
    SegmentationBaselineSettings,
    convert_detectron2_instances,
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
