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

    dataset_status = _dataset_status()
    report = {
        "python": sys.version.split()[0],
        "project": _project_status(),
        "datasets": dataset_status,
        "torch": _torch_status(),
        "opencv": _module_status("cv2"),
        "detectron2": _module_status("detectron2"),
        "mask2former": _module_status("mask2former"),
        "recommendations": _recommendations(dataset_status),
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


def _dataset_status() -> dict[str, Any]:
    """Return counts from the exact COCO files consumed by the trainer."""
    from fashion_semantic_parser.common.paths import resolve_project_path

    paths = {
        "train": "data/processed/autodl/segmentation/deepfashion2_train.json",
        "validation": (
            "data/processed/autodl/segmentation/deepfashion2_validation.json"
        ),
    }
    return {
        split: _coco_file_status(resolve_project_path(path))
        for split, path in paths.items()
    }


def _coco_file_status(path: Path) -> dict[str, Any]:
    """Read image, annotation, and category counts from one COCO JSON file."""
    if not path.exists():
        return {"exists": False}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"exists": True, "valid": False, "error": str(error)}

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    return {
        "exists": True,
        "valid": True,
        "image_count": len(images) if isinstance(images, list) else None,
        "annotation_count": (
            len(annotations) if isinstance(annotations, list) else None
        ),
        "category_count": len(categories) if isinstance(categories, list) else None,
    }


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


def _recommendations(dataset_status: dict[str, Any]) -> list[str]:
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

    train_image_count = dataset_status.get("train", {}).get("image_count")
    if isinstance(train_image_count, int) and train_image_count <= 10:
        recommendations.append(
            "Training COCO has 10 or fewer images. Regenerate it without --limit."
        )
    validation_image_count = dataset_status.get("validation", {}).get("image_count")
    if isinstance(validation_image_count, int) and validation_image_count <= 10:
        recommendations.append(
            "Validation COCO has 10 or fewer images; use 500 for staged evaluation."
        )
    return recommendations


if __name__ == "__main__":
    main()
