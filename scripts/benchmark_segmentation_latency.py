"""Benchmark PRD 3.1.1 single-image segmentation latency."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import yaml


def add_src_to_python_path() -> None:
    """Add the local src directory when the package is not installed yet."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark PRD 3.1.1 single-image segmentation latency."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Optional image paths; otherwise sample images from validation COCO.",
    )
    parser.add_argument(
        "--config",
        default="configs/segmentation_mask2former.yaml",
        help="Project-relative segmentation YAML config path.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Project-relative or absolute trained Detectron2 weights path.",
    )
    parser.add_argument(
        "--val-json",
        default=None,
        help="Optional validation COCO file used to select benchmark images.",
    )
    parser.add_argument("--image-limit", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--min-size-test", type=int, default=None)
    parser.add_argument("--max-size-test", type=int, default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional project-relative output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    """Load one predictor, warm it up, and report latency statistics."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.service.segmentation_baseline import (
        Detectron2SegmentationBaseline,
        SegmentationBaselineSettings,
    )

    raw_config = _read_yaml(resolve_project_path(args.config))
    overrides = {
        "val_json": args.val_json,
        "weights": args.weights,
        "device": args.device,
        "score_threshold": args.score_threshold,
        "min_size_test": args.min_size_test,
        "max_size_test": args.max_size_test,
    }
    raw_config.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    settings = SegmentationBaselineSettings.model_validate(raw_config)
    image_paths = _resolve_benchmark_image_paths(
        explicit_images=args.images,
        val_json=settings.val_json,
        image_root=settings.image_root,
        image_limit=args.image_limit,
        resolve_path=resolve_project_path,
    )
    results = Detectron2SegmentationBaseline(settings).benchmark_latency(
        image_paths=image_paths,
        warmup_runs=args.warmup_runs,
        measured_runs=args.runs,
        precision=args.precision,
    )
    results["model_family"] = settings.model_family
    output_json = json.dumps(results, ensure_ascii=False, indent=2)
    print(output_json)

    if args.output:
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json + "\n", encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


def _resolve_benchmark_image_paths(
    explicit_images: list[str],
    val_json: str,
    image_root: str,
    image_limit: int,
    resolve_path: Callable[[str | Path], Path],
) -> list[Path]:
    """Resolve explicit images or select a deterministic validation sample."""
    if explicit_images:
        return [
            _resolve_path_allowing_absolute(image_path, resolve_path)
            for image_path in explicit_images
        ]
    if image_limit < 1:
        raise ValueError("--image-limit must be at least one.")

    validation_path = _resolve_path_allowing_absolute(val_json, resolve_path)
    with validation_path.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)
    images = coco_data.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"No images found in validation COCO file: {validation_path}")

    selected_images = sorted(
        images,
        key=lambda image: (image.get("id", 0), image.get("file_name", "")),
    )[:image_limit]
    root_path = Path(image_root)
    resolved_paths = []
    for image in selected_images:
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"Invalid COCO image file_name: {image}")
        image_path = Path(file_name)
        if not image_path.is_absolute():
            image_path = root_path / image_path
        resolved_paths.append(
            image_path if image_path.is_absolute() else resolve_path(image_path)
        )
    return resolved_paths


def _resolve_path_allowing_absolute(
    path: str | Path,
    resolve_path: Callable[[str | Path], Path],
) -> Path:
    """Use an absolute path directly or resolve a project-relative path."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_path(candidate)


if __name__ == "__main__":
    main()
