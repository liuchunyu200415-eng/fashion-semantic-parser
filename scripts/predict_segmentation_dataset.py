"""Generate COCO predictions for full-image or automatic-ROI inference."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


def add_src_to_python_path() -> None:
    """Add the local src directory when the package is not installed yet."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse dataset prediction arguments."""
    parser = argparse.ArgumentParser(
        description="Generate COCO predictions with optional automatic person ROI."
    )
    parser.add_argument(
        "--config",
        default="configs/segmentation_mask2former_deployment.yaml",
    )
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--roi-mode",
        choices=["full", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--subject-roi-margin",
        type=float,
        default=None,
        help=(
            "Override the fractional context around automatic person ROIs; "
            "otherwise use the config value."
        ),
    )
    parser.add_argument("--image-limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    """Run one reusable model service across a COCO image set."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.service.segmentation_runtime import (
        GarmentSegmentationService,
    )

    if args.image_limit is not None and args.image_limit < 1:
        raise ValueError("--image-limit must be at least one.")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least one.")
    settings_overrides = _build_settings_overrides(
        roi_mode=args.roi_mode,
        subject_roi_margin=args.subject_roi_margin,
    )

    validation_path = _resolve_path(args.val_json, resolve_project_path)
    output_path = resolve_project_path(args.output)
    source = _read_coco(validation_path)
    images = source["images"]
    if args.image_limit is not None:
        images = images[: args.image_limit]

    service = GarmentSegmentationService(
        args.config,
        settings_overrides=settings_overrides,
    )
    predictions: list[dict[str, Any]] = []
    roi_sources: Counter[str] = Counter()
    started_at = time.perf_counter()

    for index, image in enumerate(images, start=1):
        prediction = service.segment(
            str(image["file_name"]),
            auto_subject_roi=args.roi_mode == "auto",
        )
        predictions.extend(
            prediction_to_coco_results(
                prediction,
                image_id=int(image["id"]),
            )
        )
        roi_sources[prediction.subject_roi_source or "full_image"] += 1
        if index % args.progress_every == 0 or index == len(images):
            _print_progress(
                completed=index,
                total=len(images),
                prediction_count=len(predictions),
                roi_sources=roi_sources,
                started_at=started_at,
            )

    elapsed_seconds = time.perf_counter() - started_at
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(predictions, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "config": args.config,
        "validation_json": str(validation_path),
        "predictions_json": str(output_path),
        "roi_mode": args.roi_mode,
        "subject_roi_margin_override": args.subject_roi_margin,
        "image_count": len(images),
        "prediction_count": len(predictions),
        "roi_source_counts": dict(sorted(roi_sources.items())),
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": (
            len(images) / elapsed_seconds if elapsed_seconds > 0.0 else None
        ),
    }
    summary_path = _summary_path(output_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def prediction_to_coco_results(
    prediction: Any,
    *,
    image_id: int,
) -> list[dict[str, Any]]:
    """Convert typed polygon predictions to flat COCO result records."""
    results: list[dict[str, Any]] = []
    for instance in prediction.instances:
        polygons = [polygon for polygon in instance.mask if len(polygon) >= 6]
        if not polygons:
            continue
        width = instance.box.x_max - instance.box.x_min
        height = instance.box.y_max - instance.box.y_min
        if width <= 0.0 or height <= 0.0:
            continue
        results.append(
            {
                "image_id": image_id,
                "category_id": instance.category_id,
                "bbox": [
                    instance.box.x_min,
                    instance.box.y_min,
                    width,
                    height,
                ],
                "score": instance.confidence,
                "segmentation": polygons,
            }
        )
    return results


def _read_coco(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read a COCO mapping with a stable image order."""
    with path.open("r", encoding="utf-8") as file:
        source = json.load(file)
    if not isinstance(source, dict) or not isinstance(source.get("images"), list):
        raise ValueError(f"Expected a COCO mapping with images: {path}")
    source["images"] = sorted(
        source["images"],
        key=lambda image: int(image["id"]),
    )
    return source


def _print_progress(
    *,
    completed: int,
    total: int,
    prediction_count: int,
    roi_sources: Counter[str],
    started_at: float,
) -> None:
    """Print one concise progress line suitable for tail -f."""
    elapsed_seconds = time.perf_counter() - started_at
    rate = completed / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    remaining = total - completed
    eta_seconds = remaining / rate if rate > 0.0 else 0.0
    print(
        f"[{completed}/{total}] predictions={prediction_count} "
        f"elapsed={elapsed_seconds:.1f}s eta={eta_seconds:.1f}s "
        f"roi_sources={dict(sorted(roi_sources.items()))}",
        flush=True,
    )


def _summary_path(output_path: Path) -> Path:
    """Place the run summary beside its COCO prediction list."""
    return output_path.with_name(f"{output_path.stem}_summary.json")


def _build_settings_overrides(
    *,
    roi_mode: str,
    subject_roi_margin: float | None,
) -> dict[str, float]:
    """Validate and build optional segmentation config overrides."""
    if subject_roi_margin is None:
        return {}
    if roi_mode != "auto":
        raise ValueError("--subject-roi-margin requires --roi-mode auto.")
    if not 0.0 <= subject_roi_margin <= 1.0:
        raise ValueError("--subject-roi-margin must be between 0 and 1.")
    return {"subject_roi_margin": subject_roi_margin}


def _resolve_path(path: str, resolver: Any) -> Path:
    """Use absolute paths directly and resolve project-relative paths."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolver(candidate)


if __name__ == "__main__":
    main()
