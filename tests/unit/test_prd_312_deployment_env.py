"""Tests for the exact PRD 3.1.2 deployment environment audit."""

from pathlib import Path

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
            "available_providers": [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        },
    )
    monkeypatch.setattr(
        check_prd_312_deployment_env,
        "_tensorrt_status",
        lambda: {
            "installed": True,
            "version": "8.6.1.6",
            "builder_ready": True,
        },
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
        "_tensorrt_status",
        lambda: {
            "installed": True,
            "version": "8.6.1.6",
            "builder_ready": True,
        },
    )

    report = check_prd_312_deployment_env.build_report()

    assert report["checks"]["onnxruntime_cuda_provider"] is False
    assert report["checks"]["onnxruntime_tensorrt_provider"] is False
    assert report["deployment_environment_ready"] is False


def test_version_series_rejects_neighbouring_minor() -> None:
    """PRD series checks must not accept a different runtime minor."""
    assert check_prd_312_deployment_env._version_in_series("1.17.3", "1.17")
    assert not check_prd_312_deployment_env._version_in_series("1.18.0", "1.17")


def test_setup_script_pins_cuda12_runtimes_without_cache() -> None:
    """Deployment installation must not float versions or fill the system disk."""
    script = Path(check_prd_312_deployment_env.__file__).with_name(
        "setup_prd_312_deployment_env.sh"
    )
    content = script.read_text(encoding="utf-8")

    assert "onnxruntime-cuda-12" in content
    assert "onnxruntime-gpu==1.17.1" in content
    assert "tensorrt==8.6.1.post1" in content
    assert content.count("--no-cache-dir") == 3
