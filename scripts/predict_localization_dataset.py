"""Generate saved PRD 3.1.2 predictions for one Fashionpedia part category."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_VALIDATION_JSON = (
    "data/processed/autodl/localization/fashionpedia_parts_validation.json"
)


def add_src_to_python_path() -> None:
    """Add the local package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse category-specific localization prediction arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one reusable Grounding DINO + SAM-HQ service on images that "
            "contain a selected Fashionpedia part category."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/localization_grounded_sam_hq.yaml",
    )
    parser.add_argument(
        "--val-json",
        default=DEFAULT_VALIDATION_JSON,
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--roi-mode",
        choices=["full", "auto"],
        default="full",
    )
    parser.add_argument("--image-limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--box-threshold", type=float, default=None)
    parser.add_argument("--text-threshold", type=float, default=None)
    parser.add_argument("--max-regions", type=int, default=None)
    parser.add_argument("--subject-roi-margin", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    """Run reusable-model inference and save flat COCO prediction records."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.taxonomy import (
        resolve_localization_prompt,
    )
    from fashion_semantic_parser.service.region_localization import (
        GroundedSAMHQRegionLocalizationService,
    )

    if args.image_limit is not None and args.image_limit < 1:
        raise ValueError("--image-limit must be at least one.")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least one.")
    prompt = resolve_localization_prompt(args.query)
    if prompt.region_label != args.category:
        raise ValueError(
            f"Query resolves to {prompt.region_label!r}, not "
            f"--category {args.category!r}."
        )

    validation_path = _resolve_path(args.val_json, resolve_project_path)
    source = _read_coco(validation_path)
    category_id, images = _select_category_images(
        source,
        category_name=args.category,
        image_limit=args.image_limit,
    )
    output_path = resolve_project_path(args.output)
    settings_overrides = _build_settings_overrides(
        roi_mode=args.roi_mode,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_regions=args.max_regions,
        subject_roi_margin=args.subject_roi_margin,
    )
    service = GroundedSAMHQRegionLocalizationService(
        args.config,
        settings_overrides=settings_overrides,
    )

    predictions: list[dict[str, Any]] = []
    roi_sources: Counter[str] = Counter()
    started_at = time.perf_counter()
    for index, image in enumerate(images, start=1):
        prediction = service.localize(
            str(image["file_name"]),
            args.query,
            auto_subject_roi=args.roi_mode == "auto",
        )
        predictions.extend(
            prediction_to_coco_results(
                prediction,
                image_id=int(image["id"]),
                category_id=category_id,
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
        "category": args.category,
        "category_id": category_id,
        "query": args.query,
        "roi_mode": args.roi_mode,
        "settings_overrides": settings_overrides,
        "image_ids": [int(image["id"]) for image in images],
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
    category_id: int,
) -> list[dict[str, Any]]:
    """Convert typed localization regions to flat COCO result records."""
    results: list[dict[str, Any]] = []
    for region in prediction.regions:
        polygons = [polygon for polygon in region.mask if len(polygon) >= 6]
        if not polygons:
            continue
        width = region.box.x_max - region.box.x_min
        height = region.box.y_max - region.box.y_min
        if width <= 0.0 or height <= 0.0:
            continue
        results.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [
                    region.box.x_min,
                    region.box.y_min,
                    width,
                    height,
                ],
                "score": region.confidence,
                "segmentation": polygons,
            }
        )
    return results


def _read_coco(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the COCO fields required for category-specific image selection."""
    with path.open("r", encoding="utf-8") as file:
        source = json.load(file)
    required_fields = ("images", "annotations", "categories")
    if not isinstance(source, dict) or not all(
        isinstance(source.get(field), list) for field in required_fields
    ):
        raise ValueError(f"Expected a complete COCO mapping: {path}")
    return source


def _select_category_images(
    source: dict[str, list[dict[str, Any]]],
    *,
    category_name: str,
    image_limit: int | None,
) -> tuple[int, list[dict[str, Any]]]:
    """Select sorted images with non-crowd ground truth for one exact category."""
    category_ids = [
        int(category["id"])
        for category in source["categories"]
        if category.get("name") == category_name
    ]
    if len(category_ids) != 1:
        raise ValueError(
            f"Expected one COCO category named {category_name!r}, "
            f"found {len(category_ids)}."
        )
    category_id = category_ids[0]
    image_ids = {
        int(annotation["image_id"])
        for annotation in source["annotations"]
        if int(annotation["category_id"]) == category_id
        and int(annotation.get("iscrowd", 0)) == 0
    }
    images = sorted(
        (image for image in source["images"] if int(image["id"]) in image_ids),
        key=lambda image: int(image["id"]),
    )
    if image_limit is not None:
        images = images[:image_limit]
    if not images:
        raise ValueError(f"No validation images contain category {category_name!r}.")
    return category_id, images


def _build_settings_overrides(
    *,
    roi_mode: str,
    box_threshold: float | None,
    text_threshold: float | None,
    max_regions: int | None,
    subject_roi_margin: float | None,
) -> dict[str, Any]:
    """Build optional model overrides while preserving the committed config."""
    if subject_roi_margin is not None and roi_mode != "auto":
        raise ValueError("--subject-roi-margin requires --roi-mode auto.")
    overrides = {
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "max_regions": max_regions,
        "subject_roi_margin": subject_roi_margin,
    }
    return {key: value for key, value in overrides.items() if value is not None}


def _print_progress(
    *,
    completed: int,
    total: int,
    prediction_count: int,
    roi_sources: Counter[str],
    started_at: float,
) -> None:
    """Print concise progress and ETA suitable for a nohup log."""
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
    """Place the run summary beside the flat prediction list."""
    return output_path.with_name(f"{output_path.stem}_summary.json")


def _resolve_path(path: str, resolver: Any) -> Path:
    """Use absolute paths directly and resolve project-relative paths."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolver(candidate)


if __name__ == "__main__":
    main()
