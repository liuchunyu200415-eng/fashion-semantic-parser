"""Check the AutoDL environment for PRD 3.1.1 segmentation training."""

import importlib
import json
import sys
from pathlib import Path
from typing import Any


def add_src_to_python_path() -> None:
    """Add the local src directory when the package is not installed yet."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    """Print a JSON report for segmentation training dependencies."""
    add_src_to_python_path()

    report = {
        "python": sys.version.split()[0],
        "project": _project_status(),
        "torch": _torch_status(),
        "opencv": _module_status("cv2"),
        "detectron2": _module_status("detectron2"),
        "mask2former": _module_status("mask2former"),
        "recommendations": _recommendations(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _project_status() -> dict[str, Any]:
    """Check project-level files needed for 3.1.1 training."""
    from fashion_semantic_parser.common.paths import resolve_project_path

    paths = {
        "train_coco": "data/processed/autodl/segmentation/deepfashion2_train.json",
        "val_coco": ("data/processed/autodl/segmentation/deepfashion2_validation.json"),
        "mask_rcnn_config": "configs/segmentation_mask_rcnn.yaml",
        "mask2former_config": "configs/segmentation_mask2former.yaml",
        "mask2former_repo": "external/Mask2Former",
    }
    return {name: resolve_project_path(path).exists() for name, path in paths.items()}


def _torch_status() -> dict[str, Any]:
    """Return PyTorch and CUDA availability."""
    try:
        import torch
    except ImportError:
        return {"installed": False}

    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }


def _module_status(module_name: str) -> dict[str, Any]:
    """Check whether a Python module is importable."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return {"installed": False}

    return {
        "installed": True,
        "version": getattr(module, "__version__", None),
        "path": getattr(module, "__file__", None),
    }


def _recommendations() -> list[str]:
    """Return concise next-step guidance."""
    recommendations = []
    torch_status = _torch_status()
    if not torch_status.get("installed"):
        recommendations.append("Install PyTorch with CUDA support first.")
    elif not torch_status.get("cuda_available"):
        recommendations.append("PyTorch is installed but CUDA is not available.")

    if not _module_status("detectron2").get("installed"):
        recommendations.append("Install Detectron2 before running any trainer.")
    if not _module_status("mask2former").get("installed"):
        recommendations.append(
            "Clone Mask2Former under external/Mask2Former and add it to PYTHONPATH."
        )
    return recommendations


if __name__ == "__main__":
    main()
