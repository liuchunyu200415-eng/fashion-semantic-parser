"""Verify Detectron2, Mask2Former, and BGE-M3 coexist in the PRD environment."""

# Mask2Former must load only after installed ``datasets`` has resolved.
# pylint: disable=import-outside-toplevel

import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

EXPECTED_PYTHON = "3.10.12"
EXPECTED_TORCH = "2.1.2+cu121"
EXPECTED_DETECTRON2 = "0.6"
EXPECTED_CUDA_ARCH = "8.6"


def add_src_to_python_path() -> None:
    """Add project source without placing external repositories first."""
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    """Print the joint runtime report and fail when any contract is unmet."""
    add_src_to_python_path()
    report = _runtime_report()
    report["ready"] = detectron2_runtime_ready(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


def _runtime_report() -> dict[str, Any]:
    """Import dependencies in collision-safe order and report exact versions."""
    errors: list[str] = []
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": None,
        "cuda_available": False,
        "detectron2": None,
        "detectron2_cuda_arch": [],
        "sentence_transformers": None,
        "mask2former_importable": False,
        "errors": errors,
    }
    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
        report["sentence_transformers"] = sentence_transformers.__version__
        torch = importlib.import_module("torch")
        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        detectron2 = importlib.import_module("detectron2")
        extension = importlib.import_module("detectron2._C")
        report["detectron2"] = detectron2.__version__
        report["detectron2_cuda_arch"] = extension.get_cuda_arch_flags()
        from fashion_semantic_parser.service.segmentation_baseline import (
            _append_local_mask2former_path,
        )

        _append_local_mask2former_path()
        importlib.import_module("mask2former")
        report["mask2former_importable"] = True
    except (AttributeError, ImportError, OSError, RuntimeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    return report


def detectron2_runtime_ready(report: dict[str, Any]) -> bool:
    """Return whether the exact PRD runtime and CUDA extension are available."""
    architecture_text = str(report.get("detectron2_cuda_arch", "")).lower()
    architecture_ready = any(
        marker in architecture_text for marker in (EXPECTED_CUDA_ARCH, "sm_86")
    )
    return bool(
        report["python"] == EXPECTED_PYTHON
        and report["torch"] == EXPECTED_TORCH
        and report["cuda_available"]
        and report["detectron2"] == EXPECTED_DETECTRON2
        and architecture_ready
        and report["sentence_transformers"] is not None
        and report["mask2former_importable"]
        and not report["errors"]
    )


if __name__ == "__main__":
    main()
