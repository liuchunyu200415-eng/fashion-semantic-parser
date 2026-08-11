"""Tests for multi-positive DINOv2/BGE-M3 alignment."""

import pytest
from pydantic import ValidationError

from fashion_semantic_parser.service.region_text_alignment import (
    RegionTextAlignmentSettings,
    build_positive_region_mask,
    load_region_text_alignment_settings,
    multi_positive_contrastive_loss,
    positive_top1_accuracy,
)


def test_positive_mask_preserves_multi_target_and_shared_regions() -> None:
    """Broad and modified queries can share positives without false negatives."""
    mask = build_positive_region_mask(
        [(11, 12), (12,), (13,)],
        [11, 12, 13],
    )

    assert mask.tolist() == [
        [True, True, False],
        [False, True, False],
        [False, False, True],
    ]


def test_positive_mask_rejects_missing_candidate() -> None:
    """Every source target must be present in the candidate region bank."""
    with pytest.raises(ValueError, match="missing candidate"):
        build_positive_region_mask([(11, 12)], [11])


def test_alignment_config_matches_encoder_dimensions() -> None:
    """The committed head must bridge the two pinned encoder geometries."""
    settings = load_region_text_alignment_settings()

    assert settings.text_dimension == 1024
    assert settings.region_dimension == 384
    assert settings.hidden_dimension == 512
    assert settings.temperature == 0.07


def test_alignment_hidden_dimension_cannot_undercut_region_dimension() -> None:
    """The smoke head cannot silently introduce an unintended bottleneck."""
    with pytest.raises(ValidationError, match="hidden_dimension"):
        RegionTextAlignmentSettings(hidden_dimension=128)


def test_multi_positive_loss_rewards_correct_alignment() -> None:
    """Aligned pairs must score better than a deliberately swapped pairing."""
    torch = pytest.importorskip("torch")
    text = torch.eye(3)
    regions = torch.eye(3)
    positives = torch.eye(3, dtype=torch.bool)

    aligned_loss, aligned_logits = multi_positive_contrastive_loss(
        text,
        regions,
        positives,
        temperature=0.1,
    )
    swapped_loss, _ = multi_positive_contrastive_loss(
        text,
        regions[[1, 2, 0]],
        positives,
        temperature=0.1,
    )

    assert aligned_loss.item() < swapped_loss.item()
    assert positive_top1_accuracy(aligned_logits, positives) == 1.0
