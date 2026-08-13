"""Tests for the exact PRD 3.1.2 deployment environment audit."""

from scripts import check_prd_312_deployment_env


def test_report_requires_exact_stack_and_active_gpu(monkeypatch) -> None:
    """A complete pinned GPU stack should pass every environment check.

    Args:
        monkeypatch: Pytest fixture for replacing runtime status probes.
    """
    monkeypatch.setattr(
        check_prd_312_deployment_env.platform,
        "python_version",
        lambda: "3.10.12",
    )
    monkeypatch.setattr(
        check_prd_312_deployment_env,
        "_torch_status",
        lambda: {
            "installed": True,
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 3090",
        },
    )
    monkeypatch.setattr(
        check_prd_312_deployment_env,
        "_onnxruntime_status",
        lambda: {
            "installed": True,
            "version": "1.17.3",
            "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        },
    )
    monkeypatch.setattr(
        check_prd_312_deployment_env,
        "_module_status",
        lambda _name: {"installed": True, "version": "8.6.1.6"},
    )

    report = check_prd_312_deployment_env.build_report()

    assert report["deployment_environment_ready"] is True
    assert all(report["checks"].values())


def test_report_rejects_cpu_only_onnxruntime(monkeypatch) -> None:
    """The CPU package must not be mistaken for PRD deployment readiness.

    Args:
        monkeypatch: Pytest fixture for replacing runtime status probes.
    """
    monkeypatch.setattr(
        check_prd_312_deployment_env.platform,
        "python_version",
        lambda: "3.10.12",
    )
    monkeypatch.setattr(
        check_prd_312_deployment_env,
        "_torch_status",
        lambda: {
            "installed": True,
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 3090",
        },
    )
    monkeypatch.setattr(
        check_prd_312_deployment_env,
        "_onnxruntime_status",
        lambda: {
            "installed": True,
            "version": "1.17.3",
            "available_providers": ["CPUExecutionProvider"],
        },
    )
    monkeypatch.setattr(
        check_prd_312_deployment_env,
        "_module_status",
        lambda _name: {"installed": True, "version": "8.6.1.6"},
    )

    report = check_prd_312_deployment_env.build_report()

    assert report["checks"]["onnxruntime_cuda_provider"] is False
    assert report["deployment_environment_ready"] is False


def test_version_series_rejects_neighbouring_minor() -> None:
    """PRD series checks must not accept a different runtime minor."""
    assert check_prd_312_deployment_env._version_in_series("1.17.3", "1.17")
    assert not check_prd_312_deployment_env._version_in_series("1.18.0", "1.17")
