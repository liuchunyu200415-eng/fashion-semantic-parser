"""Check the exact PRD 3.1.2 foundation-training environment."""

import importlib
import json
import platform
from typing import Any

REQUIRED_PYTHON = "3.10.12"
IMPLEMENTATION_TORCH = "2.1.2"
IMPLEMENTATION_CUDA = "12.1"


def main() -> None:
    """Print a compact readiness report and fail on a hard blocker."""
    report = build_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["foundation_training_ready"]:
        raise SystemExit(1)


def build_report() -> dict[str, Any]:
    """Inspect versions without treating unavailable modules as exceptions."""
    python_version = platform.python_version()
    torch_status = _torch_status()
    opencv_status = _module_status("cv2")
    coco_status = _module_status("pycocotools")
    checks = {
        "python_exact": python_version == REQUIRED_PYTHON,
        "torch_version": _base_version(torch_status.get("version"))
        == IMPLEMENTATION_TORCH,
        "cuda_version": torch_status.get("cuda_version") == IMPLEMENTATION_CUDA,
        "cuda_available": torch_status.get("cuda_available") is True,
        "opencv_available": opencv_status.get("installed") is True,
        "pycocotools_available": coco_status.get("installed") is True,
    }
    return {
        "python": {
            "required_by_prd": REQUIRED_PYTHON,
            "actual": python_version,
        },
        "torch": {
            "project_implementation_version": IMPLEMENTATION_TORCH,
            "project_implementation_cuda": IMPLEMENTATION_CUDA,
            **torch_status,
        },
        "opencv": opencv_status,
        "pycocotools": coco_status,
        "checks": checks,
        "foundation_training_ready": all(checks.values()),
        "scope": (
            "This check covers data loading and PyTorch foundation training only. "
            "DINOv2, the selected PRD-stack text encoder, SAM-HQ 1.0+, ONNX "
            "Runtime 1.17, and TensorRT 8.6.1 require separate model/deployment "
            "checks before acceptance."
        ),
    }


def _torch_status() -> dict[str, Any]:
    """Report the pinned project implementation and CUDA device state."""
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as error:
        return {"installed": False, "error": str(error)}
    status: dict[str, Any] = {
        "installed": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if status["cuda_available"]:
        status["device_name"] = torch.cuda.get_device_name(0)
    return status


def _module_status(module_name: str) -> dict[str, Any]:
    """Report one required Python module without crashing the full audit."""
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as error:
        return {"installed": False, "error": str(error)}
    return {
        "installed": True,
        "version": getattr(module, "__version__", None),
    }


def _base_version(value: object) -> str | None:
    """Strip local CUDA/build suffixes from one package version."""
    if not isinstance(value, str):
        return None
    return value.split("+", maxsplit=1)[0]


if __name__ == "__main__":
    main()
