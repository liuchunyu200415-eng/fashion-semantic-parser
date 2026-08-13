"""Opt-in synchronized timing helpers for dense localization diagnostics."""

from importlib import import_module
from time import perf_counter


def start_profile_stage(stage_timings: dict[str, float] | None) -> float:
    """Return a synchronized stage start without affecting normal inference.

    Args:
        stage_timings: Mutable stage output, or ``None`` outside profiling.

    Returns:
        Synchronized monotonic start time, or zero outside profiling.
    """
    if stage_timings is None:
        return 0.0
    _synchronize_cuda()
    return perf_counter()


def record_profile_stage(
    stage_timings: dict[str, float] | None,
    name: str,
    started: float,
) -> None:
    """Record one completed non-overlapping stage when profiling is enabled.

    Args:
        stage_timings: Mutable stage output, or ``None`` outside profiling.
        name: Stable diagnostic stage name.
        started: Monotonic value returned by :func:`start_profile_stage`.
    """
    if stage_timings is not None:
        _synchronize_cuda()
        stage_timings[name] = (perf_counter() - started) * 1000.0


def _synchronize_cuda() -> None:
    """Synchronize queued CUDA work only for explicit diagnostic profiling."""
    try:
        torch_module = import_module("torch")
    except ImportError:
        return
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is not None and cuda_module.is_available():
        cuda_module.synchronize()
