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
    torch_status = _torch_status()
    detectron2_status = _detectron2_status()
    mask2former_status = _module_status("mask2former")
    report = {
        "python": sys.version.split()[0],
        "project": _project_status(),
        "datasets": dataset_status,
        "torch": torch_status,
        "opencv": _module_status("cv2"),
        "detectron2": detectron2_status,
        "mask2former": mask2former_status,
        "recommendations": _recommendations(
            dataset_status,
            torch_status,
            detectron2_status,
            mask2former_status,
        ),
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
        "fashionpedia_config": ("configs/segmentation_mask2former_fashionpedia.yaml"),
        "mixed_training_config": "configs/segmentation_mask2former_mixed.yaml",
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
        "fashionpedia_train": (
            "data/processed/autodl/segmentation/fashionpedia_train.json"
        ),
        "fashionpedia_validation": (
            "data/processed/autodl/segmentation/fashionpedia_validation.json"
        ),
    }
    status = {
        split: _coco_file_status(resolve_project_path(path))
        for split, path in paths.items()
    }
    status["fashionpedia_train_images"] = _directory_file_status(
        resolve_project_path("data/raw/fashionpedia/train")
    )
    status["fashionpedia_validation_images"] = _directory_file_status(
        resolve_project_path("data/raw/fashionpedia/test")
    )
    return status


def _coco_file_status(path: Path) -> dict[str, Any]:
    """Read image, annotation, and category counts from one COCO JSON file."""
    if not path.exists():
        return {"exists": False, "is_symlink": path.is_symlink()}

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


def _directory_file_status(path: Path) -> dict[str, Any]:
    """Count top-level files and expose broken data-volume symlinks."""
    if not path.is_dir():
        return {"exists": False, "is_symlink": path.is_symlink()}
    try:
        file_count = sum(1 for child in path.iterdir() if child.is_file())
    except OSError as error:
        return {
            "exists": True,
            "valid": False,
            "is_symlink": path.is_symlink(),
            "error": str(error),
        }
    return {
        "exists": True,
        "valid": True,
        "is_symlink": path.is_symlink(),
        "file_count": file_count,
    }


def _torch_status() -> dict[str, Any]:
    """Return PyTorch and CUDA availability."""
    try:
        import torch
    except ImportError:
        return {"installed": False}

    status: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        capability = torch.cuda.get_device_capability(0)
        properties = torch.cuda.get_device_properties(0)
        status.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "memory_gib": round(properties.total_memory / 1024**3, 2),
            }
        )
    return status


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


def _detectron2_status() -> dict[str, Any]:
    """Report Detectron2 and the architectures compiled into its CUDA extension."""
    status = _module_status("detectron2")
    if not status.get("installed"):
        return status
    try:
        extension = importlib.import_module("detectron2._C")
        get_arch_flags = getattr(extension, "get_cuda_arch_flags", None)
        if callable(get_arch_flags):
            status["cuda_arch_flags"] = get_arch_flags()
    except (ImportError, OSError, RuntimeError) as error:
        status["extension_error"] = str(error)
    return status


def _recommendations(
    dataset_status: dict[str, Any],
    torch_status: dict[str, Any],
    detectron2_status: dict[str, Any],
    mask2former_status: dict[str, Any],
) -> list[str]:
    """Return concise next-step guidance."""
    recommendations = []
    if not torch_status.get("installed"):
        recommendations.append("Install PyTorch with CUDA support first.")
    elif not torch_status.get("cuda_available"):
        recommendations.append("PyTorch is installed but CUDA is not available.")

    if not detectron2_status.get("installed"):
        recommendations.append("Install Detectron2 before running any trainer.")
    elif detectron2_status.get("extension_error"):
        recommendations.append(
            "Detectron2 CUDA extension cannot be loaded; rebuild it for this host."
        )
    elif not _detectron2_arch_is_compatible(torch_status, detectron2_status):
        capability = torch_status.get("compute_capability")
        arch_flags = detectron2_status.get("cuda_arch_flags")
        recommendations.append(
            "Detectron2 CUDA architectures "
            f"({arch_flags}) do not include the current GPU ({capability}); "
            "rebuild Detectron2 and Mask2Former CUDA extensions."
        )
    if not mask2former_status.get("installed"):
        recommendations.append(
            "Clone Mask2Former under external/Mask2Former or install it."
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
    for dataset_name in ("fashionpedia_train", "fashionpedia_validation"):
        status = dataset_status.get(dataset_name, {})
        if not status.get("exists"):
            recommendations.append(
                f"{dataset_name} COCO is missing or its data-volume symlink is broken."
            )
        elif not status.get("valid"):
            recommendations.append(f"{dataset_name} COCO is not valid JSON.")
    for directory_name in (
        "fashionpedia_train_images",
        "fashionpedia_validation_images",
    ):
        status = dataset_status.get(directory_name, {})
        if not status.get("exists"):
            recommendations.append(
                f"{directory_name} directory is missing or its symlink is broken."
            )
        elif status.get("file_count") == 0:
            recommendations.append(f"{directory_name} directory contains no files.")
    return recommendations


def _detectron2_arch_is_compatible(
    torch_status: dict[str, Any],
    detectron2_status: dict[str, Any],
) -> bool:
    """Return whether Detectron2 contains code for the active CUDA device."""
    capability = torch_status.get("compute_capability")
    arch_flags = detectron2_status.get("cuda_arch_flags")
    if not isinstance(capability, str) or arch_flags is None:
        return True
    normalized_flags = str(arch_flags).lower()
    compact_capability = capability.replace(".", "")
    return (
        capability in normalized_flags or f"sm_{compact_capability}" in normalized_flags
    )


if __name__ == "__main__":
    main()
