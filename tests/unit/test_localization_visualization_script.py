"""Tests for language-guided localization comparison helpers."""

import numpy as np
import pytest

from scripts.visualize_localization_comparison import (
    _direct_mask_iou_metrics,
    _join_panels,
    _load_ground_truth,
    _parse_prediction_specs,
    _polygons_to_mask,
    _validate_prediction_payloads,
)


class _FakeCOCO:
    """Minimal COCO stand-in for evaluation category selection."""

    def __init__(self) -> None:
        self.cats = {
            1: {"name": "collar", "supercategory": "collar"},
            2: {"name": "lapel", "supercategory": "collar"},
            3: {"name": "zipper", "supercategory": "decoration"},
        }
        self.selected_category_ids: list[int] = []

    def getAnnIds(
        self,
        *,
        imgIds: list[int],
        catIds: list[int],
        iscrowd: bool,
    ) -> list[int]:
        assert imgIds == [7]
        assert iscrowd is False
        self.selected_category_ids = catIds
        return catIds

    def loadAnns(self, annotation_ids: list[int]) -> list[dict[str, int]]:
        return [{"id": annotation_id} for annotation_id in annotation_ids]


def test_ground_truth_prefers_exact_category_over_region_group() -> None:
    """A collar query must not be penalized for separate lapel annotations."""
    coco = _FakeCOCO()

    annotations, scope = _load_ground_truth(coco, image_id=7, target_label="collar")

    assert scope == "exact_category"
    assert coco.selected_category_ids == [1]
    assert annotations == [{"id": 1}]


def test_ground_truth_supports_broad_region_groups() -> None:
    """A group query such as decoration should include all member categories."""
    coco = _FakeCOCO()

    annotations, scope = _load_ground_truth(
        coco,
        image_id=7,
        target_label="decoration",
    )

    assert scope == "region_group"
    assert coco.selected_category_ids == [3]
    assert annotations == [{"id": 3}]


def test_unmapped_query_does_not_load_unrelated_annotations() -> None:
    """An unsupported free-form query must not treat every image mask as truth."""
    coco = _FakeCOCO()

    annotations, scope = _load_ground_truth(
        coco,
        image_id=7,
        target_label="custom",
    )

    assert scope == "unmapped"
    assert annotations == []
    assert coco.selected_category_ids == []


def test_direct_mask_iou_counts_extra_prediction_as_false_positive() -> None:
    """An exact mask plus one false mask should have full recall and half precision."""
    ground_truth = np.zeros((8, 8), dtype=bool)
    ground_truth[1:4, 1:4] = True
    exact_prediction = ground_truth.copy()
    false_prediction = np.zeros((8, 8), dtype=bool)
    false_prediction[5:7, 5:7] = True

    metrics = _direct_mask_iou_metrics(
        [exact_prediction, false_prediction],
        [ground_truth],
    )

    assert metrics["MatchedCount"] == 1.0
    assert metrics["PredictionCount"] == 2.0
    assert metrics["GroundTruthCount"] == 1.0
    assert metrics["Precision50"] == pytest.approx(50.0)
    assert metrics["Recall50"] == pytest.approx(100.0)
    assert metrics["MatchedMeanIoU"] == pytest.approx(100.0)
    assert metrics["AllGTMeanIoU"] == pytest.approx(100.0)
    assert metrics["matches"] == [
        {
            "prediction_index": 0,
            "ground_truth_index": 0,
            "mask_iou": 100.0,
        }
    ]


def test_empty_prediction_metrics_are_json_safe() -> None:
    """Missing predictions should use null-compatible values instead of NaN."""
    ground_truth = np.ones((4, 4), dtype=bool)

    metrics = _direct_mask_iou_metrics([], [ground_truth])

    assert metrics["MatchedCount"] == 0.0
    assert metrics["Precision50"] is None
    assert metrics["Recall50"] == 0.0
    assert metrics["MatchedMeanIoU"] is None
    assert metrics["AllGTMeanIoU"] == 0.0


def test_polygons_are_clipped_and_rasterized() -> None:
    """Saved polygons outside image bounds should still produce a valid mask."""
    mask = _polygons_to_mask(
        [[-2.0, -2.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0]],
        height=5,
        width=5,
    )

    assert mask.dtype == np.bool_
    assert mask.shape == (5, 5)
    assert mask[1, 1]
    assert not mask[4, 4]


def test_prediction_specs_require_unique_labels() -> None:
    """Comparison labels should be explicit and unique."""
    assert _parse_prediction_specs(
        ["full=outputs/full.json", "auto=outputs/auto.json"]
    ) == [
        ("full", "outputs/full.json"),
        ("auto", "outputs/auto.json"),
    ]

    with pytest.raises(ValueError, match="Duplicate prediction label"):
        _parse_prediction_specs(["full=a.json", "full=b.json"])


def test_comparison_requires_same_image_and_query() -> None:
    """Panels must compare inference modes for exactly the same request."""
    valid_payload = {
        "image_path": "images/a.jpg",
        "query": "collar",
        "regions": [],
    }

    assert _validate_prediction_payloads([("full", valid_payload)]) == (
        "images/a.jpg",
        "collar",
    )
    with pytest.raises(ValueError, match="different image_path"):
        _validate_prediction_payloads(
            [
                ("full", valid_payload),
                (
                    "auto",
                    {
                        "image_path": "images/b.jpg",
                        "query": "collar",
                        "regions": [],
                    },
                ),
            ]
        )


def test_join_panels_pads_shorter_images() -> None:
    """Different panel heights should concatenate without stretching content."""
    tall = np.full((8, 4, 3), 50, dtype=np.uint8)
    short = np.full((5, 4, 3), 100, dtype=np.uint8)

    result = _join_panels([tall, short])

    assert result.shape == (8, 8, 3)
    assert np.all(result[:5, 4:] == 100)
    assert np.all(result[5:, 4:] == 0)
