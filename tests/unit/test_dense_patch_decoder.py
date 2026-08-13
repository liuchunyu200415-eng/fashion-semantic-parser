"""Tests for the query-conditioned multiscale DINOv2 patch decoder."""

import pytest

from fashion_semantic_parser.service.dense_patch_alignment import (
    DensePatchAlignmentSettings,
)
from fashion_semantic_parser.service.dense_patch_decoder import (
    build_multiscale_patch_decoder,
    multiscale_patch_decoder_logits,
)


def test_multiscale_decoder_preserves_query_patch_geometry() -> None:
    """The decoder should emit one spatial logit for every query patch."""
    torch = pytest.importorskip("torch")
    settings = DensePatchAlignmentSettings(
        decoder_hidden_dimension=8,
        decoder_branch_dimension=8,
        decoder_dilations=(1, 2),
        decoder_dropout=0.0,
    )
    decoder = build_multiscale_patch_decoder(4, settings)
    patches = torch.randn(2, 9, 4)
    queries = torch.randn(2, 4)

    logits = multiscale_patch_decoder_logits(
        decoder,
        patches,
        queries,
        (torch.tensor(0.0), torch.tensor(0.0), 100.0),
    )

    assert logits.shape == (2, 9)
    assert torch.isfinite(logits).all()


def test_multiscale_decoder_rejects_non_square_patch_count() -> None:
    """Spatial convolution cannot silently reshape a non-square token set."""
    torch = pytest.importorskip("torch")
    settings = DensePatchAlignmentSettings(
        decoder_hidden_dimension=8,
        decoder_branch_dimension=8,
        decoder_dilations=(1,),
    )
    decoder = build_multiscale_patch_decoder(4, settings)

    with pytest.raises(ValueError, match="geometry"):
        multiscale_patch_decoder_logits(
            decoder,
            torch.randn(1, 8, 4),
            torch.randn(1, 4),
            (torch.tensor(0.0), torch.tensor(0.0), 100.0),
        )
