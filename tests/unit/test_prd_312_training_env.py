"""Tests for the exact PRD 3.1.2 training-environment contract."""

from scripts.check_prd_312_training_env import _base_version


def test_base_version_strips_cuda_build_suffix() -> None:
    """CUDA wheel suffixes must not change the pinned PyTorch comparison."""
    assert _base_version("2.1.2+cu121") == "2.1.2"
    assert _base_version("2.1.2") == "2.1.2"
    assert _base_version(None) is None
