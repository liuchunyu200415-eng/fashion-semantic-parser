"""Check the AutoDL environment for PRD 3.1.2 localization inference."""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def add_project_paths() -> None:
    """Expose local source and optional official model checkouts."""
    project_root = Path(__file__).resolve().parents[1]
    for path in (
        project_root / "src",
        project_root / "external" / "GroundingDINO",
        project_root / "external" / "sam-hq",
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    """Parse the localization configuration to inspect."""
    parser = argparse.ArgumentParser(
        description="Check Grounding DINO + SAM-HQ localization dependencies."
    )
    parser.add_argument(
        "--config",
        default="configs/localization_grounded_sam_hq.yaml",
        help="Project-relative localization YAML.",
    )
    return parser.parse_args()


def main() -> None:
    """Print one JSON readiness report with actionable recommendations."""
    add_project_paths()
    args = parse_args()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.service.grounded_sam_hq import (
        load_grounded_sam_hq_settings,
    )

    settings = load_grounded_sam_hq_settings(args.config)
    paths = {
        "config": _path_status(resolve_project_path(args.config)),
        "train_coco": _coco_status(
            resolve_project_path(
                "data/processed/autodl/localization/" "fashionpedia_parts_train.json"
            )
        ),
        "validation_coco": _coco_status(
            resolve_project_path(
                "data/processed/autodl/localization/"
                "fashionpedia_parts_validation.json"
            )
        ),
        "grounding_dino_repo": _path_status(
            resolve_project_path(settings.grounding_dino_repo)
        ),
        "grounding_dino_config": _path_status(
            resolve_project_path(settings.grounding_dino_config)
        ),
        "grounding_dino_weights": _path_status(
            resolve_project_path(settings.grounding_dino_weights)
        ),
        "sam_hq_repo": (
            _path_status(resolve_project_path(settings.sam_hq_repo))
            if settings.sam_hq_repo
            else {"configured": False}
        ),
        "sam_hq_weights": _path_status(resolve_project_path(settings.sam_hq_weights)),
    }
    report = {
        "python": sys.version.split()[0],
        "settings": {
            "device": settings.device,
            "precision": settings.precision,
            "sam_hq_model_type": settings.sam_hq_model_type,
            "box_threshold": settings.box_threshold,
            "text_threshold": settings.text_threshold,
            "subject_roi_margin": settings.subject_roi_margin,
        },
        "paths": paths,
        "torch": _torch_status(),
        "transformers": _transformers_status(),
        "detectron2": _module_status("detectron2"),
        "grounding_dino": _module_status("groundingdino.util.inference"),
        "sam_hq": _sam_hq_status(settings.sam_hq_module),
    }
    report["recommendations"] = _recommendations(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _path_status(path: Path) -> dict[str, Any]:
    """Report whether one required file or directory exists."""
    status: dict[str, Any] = {
        "exists": path.exists(),
        "path": str(path),
    }
    if path.is_file():
        status["size_mb"] = round(path.stat().st_size / 1024**2, 2)
    elif path.is_dir():
        status["kind"] = "directory"
    return status


def _coco_status(path: Path) -> dict[str, Any]:
    """Read counts from one converted Fashionpedia local-part COCO file."""
    status = _path_status(path)
    if not path.is_file():
        return status
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        status.update({"valid": False, "error": str(error)})
        return status
    status.update(
        {
            "valid": True,
            "image_count": len(data.get("images", [])),
            "annotation_count": len(data.get("annotations", [])),
            "category_count": len(data.get("categories", [])),
        }
    )
    return status


def _torch_status() -> dict[str, Any]:
    """Return PyTorch and CUDA readiness."""
    try:
        torch = importlib.import_module("torch")
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
        status.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": f"{capability[0]}.{capability[1]}",
            }
        )
    return status


def _module_status(module_name: str) -> dict[str, Any]:
    """Import one optional model dependency without crashing the report."""
    try:
        module = importlib.import_module(module_name)
    except (ImportError, NameError, OSError, RuntimeError) as error:
        return {"installed": False, "error": str(error)}
    return {
        "installed": True,
        "module": module_name,
        "version": getattr(module, "__version__", None),
        "path": getattr(module, "__file__", None),
    }


def _transformers_status() -> dict[str, Any]:
    """Check that Transformers can expose its PyTorch BERT implementation."""
    status = _module_status("transformers")
    if not status.get("installed"):
        return status

    try:
        transformers = importlib.import_module("transformers")
        transformers_utils = importlib.import_module("transformers.utils")
        status["torch_available"] = bool(transformers_utils.is_torch_available())
        getattr(transformers, "BertModel")
        status["bert_model_available"] = True
    except (AttributeError, ImportError, NameError, OSError, RuntimeError) as error:
        status["bert_model_available"] = False
        status["error"] = str(error)
    return status


def _sam_hq_status(module_name: str) -> dict[str, Any]:
    """Prefer the explicit pip package, then the official source module."""
    candidates = (
        ("segment_anything_hq", "segment_anything")
        if module_name == "auto"
        else (module_name,)
    )
    attempts: list[dict[str, Any]] = []
    for candidate in candidates:
        status = _module_status(candidate)
        if status.get("installed"):
            status["attempts"] = attempts
            return status
        attempts.append({"module": candidate, "error": status.get("error")})
    return {"installed": False, "attempts": attempts}


def _recommendations(report: dict[str, Any]) -> list[str]:
    """Return only blockers that still need work."""
    recommendations = []
    torch_status = report.get("torch", {})
    if not torch_status.get("installed"):
        recommendations.append("Install CUDA-enabled PyTorch first.")
    elif not torch_status.get("cuda_available"):
        recommendations.append("Start a GPU instance; CUDA is not available.")

    paths = report.get("paths", {})
    for name in ("train_coco", "validation_coco"):
        status = paths.get(name, {})
        if not status.get("exists"):
            recommendations.append(f"{name} is missing; run the part converter.")
        elif not status.get("valid"):
            recommendations.append(f"{name} is not valid COCO JSON.")
    for name in (
        "grounding_dino_repo",
        "grounding_dino_config",
        "grounding_dino_weights",
        "sam_hq_weights",
    ):
        if not paths.get(name, {}).get("exists"):
            recommendations.append(f"{name} is missing.")

    if not report.get("grounding_dino", {}).get("installed"):
        recommendations.append(
            "Install the official GroundingDINO checkout with its CUDA extension."
        )
    transformers_status = report.get("transformers", {})
    if not transformers_status.get("installed"):
        recommendations.append(
            "Install transformers==4.35.2 for Grounding DINO's BERT encoder."
        )
    elif not transformers_status.get("torch_available") or not transformers_status.get(
        "bert_model_available"
    ):
        version = transformers_status.get("version", "unknown")
        recommendations.append(
            "Install transformers==4.35.2; the current Transformers "
            f"{version} cannot expose its PyTorch BERT model."
        )
    if not report.get("detectron2", {}).get("installed"):
        recommendations.append(
            "Install Detectron2 for automatic primary-person ROI detection."
        )
    if not report.get("sam_hq", {}).get("installed"):
        recommendations.append(
            "Install segment-anything-hq or expose the official sam-hq checkout."
        )
    return recommendations


if __name__ == "__main__":
    main()
