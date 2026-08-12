"""Tests for DINOv2 dense similarity Mask construction."""

import numpy as np
import pytest

from fashion_semantic_parser.service.dense_region_localization import (
    binary_mask_iou,
    box_iou,
    dense_similarity_scores,
    quantile_mask_candidates,
)


def test_dense_similarity_scores_preserve_query_and_grid_geometry() -> None:
    """Cosine scoring should retain one grid per query."""
    patches = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ],
        dtype=np.float32,
    )
    queries = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    scores = dense_similarity_scores(patches, queries)

    assert scores.shape == (2, 2, 2)
    assert scores[0, 0, 0] == pytest.approx(1.0)
    assert scores[1, 0, 1] == pytest.approx(1.0)


def test_quantile_candidates_become_tighter_and_keep_boxes() -> None:
    """Higher score quantiles should retain fewer high-similarity pixels."""
    scores = np.arange(100, dtype=np.float32).reshape(10, 10)

    candidates = quantile_mask_candidates(scores, (0.5, 0.9))

    assert candidates[0].mask.sum() == 50
    assert candidates[1].mask.sum() == 10
    assert candidates[1].box == (0.0, 9.0, 10.0, 10.0)


def test_mask_and_box_iou_retain_misses() -> None:
    """Exact geometry should score one while disjoint or missing output scores zero."""
    target = np.zeros((4, 4), dtype=bool)
    target[:2, :2] = True
    miss = np.zeros((4, 4), dtype=bool)
    miss[2:, 2:] = True

    assert binary_mask_iou(target, target) == 1.0
    assert binary_mask_iou(target, miss) == 0.0
    assert box_iou((0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0)) == 1.0
    assert box_iou((0.0, 0.0, 2.0, 2.0), None) == 0.0


def test_dense_localization_rejects_invalid_features_and_quantiles() -> None:
    """Dimension drift, zero features, and unordered thresholds must fail."""
    with pytest.raises(ValueError, match="dimensions must match"):
        dense_similarity_scores(np.ones((2, 2, 3)), np.ones((1, 2)))
    with pytest.raises(ValueError, match="zero vectors"):
        dense_similarity_scores(np.zeros((2, 2, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="unique, ascending"):
        quantile_mask_candidates(np.ones((2, 2)), (0.9, 0.5))
