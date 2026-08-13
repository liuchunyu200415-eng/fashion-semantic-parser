"""Tests for full-query dense patch target-area control."""

import numpy as np
import pytest

from fashion_semantic_parser.service.dense_patch_alignment import (
    DensePatchAlignmentSettings,
)
from fashion_semantic_parser.service.dense_patch_area import (
    build_query_area_predictor,
    oracle_area_topk_masks,
    query_area_logits,
    query_area_loss,
    topk_patch_masks,
)


def test_query_area_predictor_preserves_batch_geometry() -> None:
    """Each complete image/query pair should produce one finite area logit."""
    torch = pytest.importorskip("torch")
    settings = DensePatchAlignmentSettings(
        area_hidden_dimension=8,
        area_dropout=0.0,
    )
    predictor = build_query_area_predictor(4, settings)

    logits = query_area_logits(
        predictor,
        torch.randn(3, 9, 4),
        torch.randn(3, 4),
    )

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_query_area_loss_rewards_correct_foreground_fraction() -> None:
    """A matching area logit should have lower loss than a broad prediction."""
    torch = pytest.importorskip("torch")
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    matching = torch.logit(torch.tensor([0.25]))
    too_broad = torch.logit(torch.tensor([0.90]))

    assert query_area_loss(matching, targets) < query_area_loss(too_broad, targets)


def test_topk_patch_masks_use_query_specific_predicted_area() -> None:
    """Area control should select exact query-specific counts without labels."""
    probabilities = np.asarray(
        [
            [0.8, 0.9, 0.2, 0.1],
            [0.8, 0.9, 0.2, 0.1],
        ],
        dtype=np.float32,
    )

    masks = topk_patch_masks(
        probabilities,
        np.asarray([0.25, 0.75], dtype=np.float32),
    )

    assert masks.sum(axis=1).tolist() == [1, 3]
    assert masks[0].tolist() == [False, True, False, False]
    assert masks[1].tolist() == [True, True, True, False]


def test_topk_patch_masks_reject_nonfinite_area() -> None:
    """Invalid predicted area must not silently become an arbitrary Mask."""
    with pytest.raises(ValueError, match="invalid"):
        topk_patch_masks(
            np.asarray([[0.5, 0.4]], dtype=np.float32),
            np.asarray([np.nan], dtype=np.float32),
        )


def test_oracle_area_audit_separates_pixel_mass_and_patch_support() -> None:
    """Thin targets should expose different pixel-area and support oracles."""
    selections, fractions = oracle_area_topk_masks(
        np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
        np.asarray([[0.1, 0.1], [0.0, 0.0]], dtype=np.float32),
    )

    assert fractions.tolist() == pytest.approx([0.05, 0.5])
    assert selections.reshape(2, -1).sum(axis=1).tolist() == [1, 2]
