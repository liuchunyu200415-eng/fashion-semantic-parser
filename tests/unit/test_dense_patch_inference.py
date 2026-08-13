"""Tests for shared frozen dense patch inference dispatch."""

import numpy as np
import pytest

from fashion_semantic_parser.service.dense_patch_alignment import (
    DensePatchAlignmentCheckpoint,
    DensePatchAlignmentSettings,
)
from fashion_semantic_parser.service.dense_patch_inference import (
    predict_patch_outputs,
)
from fashion_semantic_parser.service.region_text_alignment import (
    RegionTextAlignmentSettings,
)


def _checkpoint(model_type: str) -> DensePatchAlignmentCheckpoint:
    """Build a minimal frozen checkpoint for inference dispatch tests."""
    return DensePatchAlignmentCheckpoint(
        projection=object(),
        alignment_settings=RegionTextAlignmentSettings(
            text_dimension=2,
            region_dimension=2,
            hidden_dimension=2,
        ),
        dense_settings=DensePatchAlignmentSettings(),
        logit_scale=1.0,
        logit_bias=0.0,
        model_type=model_type,
        decoder=None,
        area_predictor=None,
        training_input_size=518,
    )


def test_shared_inference_retains_cosine_checkpoint_behavior() -> None:
    """Schema-one inference should remain available without PyTorch dispatch."""
    probabilities, areas = predict_patch_outputs(
        _checkpoint("cosine_calibration"),
        np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        "cpu",
    )

    assert probabilities.shape == (1, 1, 2)
    assert np.all(np.isfinite(probabilities))
    assert probabilities[0, 0, 0] > probabilities[0, 0, 1]
    assert areas is None


def test_shared_inference_rejects_missing_decoder() -> None:
    """Malformed multiscale checkpoint must fail before runtime execution."""
    with pytest.raises(ValueError, match="decoder"):
        predict_patch_outputs(
            _checkpoint("multiscale_decoder"),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            "cpu",
        )
