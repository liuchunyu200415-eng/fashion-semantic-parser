"""Tests for DINOv2 ONNX export parity helpers."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.export_dinov2_onnx import (
    _parity_metrics,
    _tensorrt_profile_executed,
)


def test_parity_metrics_report_identical_tokens() -> None:
    """Identical provider output must have exact parity."""
    tokens = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)

    metrics = _parity_metrics(tokens, tokens.copy())

    assert metrics["max_abs_error"] == 0.0
    assert metrics["mean_abs_error"] == 0.0
    assert metrics["mean_cosine_similarity"] == 1.0


def test_parity_metrics_reject_shape_drift() -> None:
    """A changed patch-token contract must fail before scoring."""
    with pytest.raises(ValueError, match="shape differs"):
        _parity_metrics(
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 3, 3), dtype=np.float32),
        )


def test_profile_requires_recorded_tensorrt_execution(tmp_path: Path) -> None:
    """Provider availability alone must not count as TensorRT execution.

    Args:
        tmp_path: Pytest temporary directory for synthetic ORT profiles.
    """
    trt_profile = tmp_path / "trt.json"
    trt_profile.write_text(
        json.dumps(
            [
                {
                    "name": "TRTKernel_graph_0_kernel_time",
                    "args": {"provider": "TensorrtExecutionProvider"},
                }
            ]
        ),
        encoding="utf-8",
    )
    cuda_profile = tmp_path / "cuda.json"
    cuda_profile.write_text(
        json.dumps(
            [
                {
                    "name": "MatMul_kernel_time",
                    "args": {"provider": "CUDAExecutionProvider"},
                }
            ]
        ),
        encoding="utf-8",
    )

    assert _tensorrt_profile_executed(trt_profile)
    assert not _tensorrt_profile_executed(cuda_profile)
