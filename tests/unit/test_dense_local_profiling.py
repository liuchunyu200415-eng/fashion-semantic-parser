"""Tests for opt-in dense localization stage timing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fashion_semantic_parser.service import dense_local_profiling


def test_disabled_profiling_does_not_import_or_synchronize(monkeypatch) -> None:
    """The normal inference path must not introduce profiling work.

    Args:
        monkeypatch: Pytest fixture for replacing the import boundary.
    """
    import_module = Mock(side_effect=AssertionError("unexpected import"))
    monkeypatch.setattr(dense_local_profiling, "import_module", import_module)

    assert dense_local_profiling.start_profile_stage(None) == 0.0
    dense_local_profiling.record_profile_stage(None, "ignored", 0.0)
    import_module.assert_not_called()


def test_profiled_stage_synchronizes_cuda_boundaries(monkeypatch) -> None:
    """GPU timing must synchronize both start and completed boundaries.

    Args:
        monkeypatch: Pytest fixture for replacing CUDA and timer boundaries.
    """
    cuda = SimpleNamespace(is_available=Mock(return_value=True), synchronize=Mock())
    monkeypatch.setattr(
        dense_local_profiling,
        "import_module",
        Mock(return_value=SimpleNamespace(cuda=cuda)),
    )
    monkeypatch.setattr(
        dense_local_profiling,
        "perf_counter",
        Mock(side_effect=[10.0, 10.025]),
    )
    timings: dict[str, float] = {}

    started = dense_local_profiling.start_profile_stage(timings)
    dense_local_profiling.record_profile_stage(timings, "coarse_dinov2", started)

    assert timings == {"coarse_dinov2": pytest.approx(25.0)}
    assert cuda.synchronize.call_count == 2
