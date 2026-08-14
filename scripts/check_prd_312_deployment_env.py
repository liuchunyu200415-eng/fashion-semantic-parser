"""Audit the exact PRD 3.1.2 ONNX Runtime and TensorRT environment."""

import importlib
import json
import platform
from importlib import metadata

REQUIRED_PYTHON = "3.10.12"
REQUIRED_ONNXRUNTIME_SERIES = "1.17"
REQUIRED_TENSORRT_SERIES = "8.6.1"
REQUIRED_GPU_NAME = "RTX 3090"
REQUIRED_CUDA_PACKAGES = {
    "nvidia-cublas-cu12": "12.1.3.1",
    "nvidia-cuda-nvrtc-cu12": "12.1.105",
    "nvidia-cuda-runtime-cu12": "12.1.105",
    "nvidia-cudnn-cu12": "8.9.2.26",
}


def main() -> None:
    """Print deployment-runtime readiness and fail on an unmet requirement.

    Raises:
        SystemExit: If any required deployment component is unavailable.
    """
    report = build_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["deployment_environment_ready"]:
        raise SystemExit(1)


def build_report() -> dict[str, object]:
    """Inspect exact runtime versions and active GPU providers.

    Returns:
        JSON-safe environment report with individual acceptance checks.
    """
    python_version = platform.python_version()
    torch_status = _torch_status()
    onnxruntime_status = _onnxruntime_status()
    tensorrt_status = _tensorrt_status()
    cuda_packages = _cuda_package_status()
    gpu_name = torch_status.get("device_name")
    providers = onnxruntime_status.get("available_providers")
    checks = {
        "python_exact": python_version == REQUIRED_PYTHON,
        "cuda_available": torch_status.get("cuda_available") is True,
        "cuda_operation_ready": torch_status.get("cuda_operation_ready") is True,
        "rtx_3090_device": isinstance(gpu_name, str) and REQUIRED_GPU_NAME in gpu_name,
        "onnxruntime_1_17": _version_in_series(
            onnxruntime_status.get("version"),
            REQUIRED_ONNXRUNTIME_SERIES,
        ),
        "onnxruntime_cuda_provider": isinstance(providers, list)
        and "CUDAExecutionProvider" in providers,
        "onnxruntime_tensorrt_provider": isinstance(providers, list)
        and "TensorrtExecutionProvider" in providers,
        "tensorrt_8_6_1": _version_in_series(
            tensorrt_status.get("version"),
            REQUIRED_TENSORRT_SERIES,
        ),
        "tensorrt_builder_ready": tensorrt_status.get("builder_ready") is True,
        "pytorch_cuda_packages_exact": cuda_packages.get("all_exact") is True,
    }
    return {
        "python": {"required": REQUIRED_PYTHON, "actual": python_version},
        "torch_cuda": torch_status,
        "onnxruntime": {
            "required_series": REQUIRED_ONNXRUNTIME_SERIES,
            **onnxruntime_status,
        },
        "tensorrt": {
            "required_series": REQUIRED_TENSORRT_SERIES,
            **tensorrt_status,
        },
        "pytorch_cuda_packages": cuda_packages,
        "checks": checks,
        "deployment_environment_ready": all(checks.values()),
        "scope": (
            "Environment readiness only. It does not prove successful model export, "
            + "TensorRT engine parity, 92% accuracy, 30 ms latency, or 60 QPS."
        ),
    }


def _torch_status() -> dict[str, object]:
    """Report CUDA availability and the active device name."""
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as error:
        return {"installed": False, "error": str(error)}
    cuda_available = bool(torch.cuda.is_available())
    result: dict[str, object] = {
        "installed": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": cuda_available,
    }
    if cuda_available:
        result["device_name"] = torch.cuda.get_device_name(0)
        try:
            value = torch.ones(1, device="cuda", dtype=torch.float16) + 1
            torch.cuda.synchronize()
            result["cuda_operation_ready"] = float(value.item()) == 2.0
            result["cudnn_version"] = torch.backends.cudnn.version()
        except RuntimeError as error:
            result["cuda_operation_ready"] = False
            result["cuda_operation_error"] = str(error)
    return result


def _onnxruntime_status() -> dict[str, object]:
    """Report ONNX Runtime version and compiled execution providers."""
    try:
        module = importlib.import_module("onnxruntime")
    except (ImportError, OSError) as error:
        return {"installed": False, "error": str(error)}
    try:
        providers = list(module.get_available_providers())
    except (AttributeError, RuntimeError) as error:
        return {
            "installed": True,
            "version": getattr(module, "__version__", None),
            "available_providers": [],
            "provider_error": str(error),
        }
    return {
        "installed": True,
        "version": getattr(module, "__version__", None),
        "available_providers": providers,
    }


def _tensorrt_status() -> dict[str, object]:
    """Report the native TensorRT Python runtime and Builder state."""
    try:
        module = importlib.import_module("tensorrt")
    except (ImportError, OSError) as error:
        return {"installed": False, "error": str(error)}
    result: dict[str, object] = {
        "installed": True,
        "version": getattr(module, "__version__", None),
    }
    try:
        logger = module.Logger(module.Logger.ERROR)
        result["builder_ready"] = module.Builder(logger) is not None
    except (AttributeError, RuntimeError) as error:
        result["builder_ready"] = False
        result["builder_error"] = str(error)
    return result


def _cuda_package_status() -> dict[str, object]:
    """Report exact CUDA Python packages required by PyTorch 2.1.2 cu121."""
    packages: dict[str, object] = {}
    for package_name, required_version in REQUIRED_CUDA_PACKAGES.items():
        try:
            actual_version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            actual_version = None
        packages[package_name] = {
            "required": required_version,
            "actual": actual_version,
            "exact": actual_version == required_version,
        }
    return {
        "packages": packages,
        "all_exact": all(
            isinstance(status, dict) and status.get("exact") is True
            for status in packages.values()
        ),
    }


def _version_in_series(value: object, required_series: str) -> bool:
    """Return whether a version belongs to one required dotted series."""
    if not isinstance(value, str):
        return False
    base_version = value.split("+", maxsplit=1)[0]
    return base_version == required_series or base_version.startswith(
        required_series + "."
    )


if __name__ == "__main__":
    main()
