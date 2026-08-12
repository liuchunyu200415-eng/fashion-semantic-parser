"""Run a reusable open-language Grounding DINO + SAM-HQ smoke benchmark."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np

DEFAULT_CONFIG = "configs/localization_grounded_sam_hq.yaml"
DEFAULT_PART_CONFIG = "configs/localization_mask2former_parts_deployment.yaml"
DEFAULT_GARMENT_CONFIG = "configs/segmentation_mask2former_deployment.yaml"
DEFAULT_OUTPUT_DIR = "outputs/localization/referring_smoke"


def add_src_to_python_path() -> None:
    """Add the project, local package, and optional Mask2Former checkout."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    mask2former_path = project_root / "external" / "Mask2Former"
    for path in (project_root, src_path, mask2former_path):
        if path == mask2former_path and not path.is_dir():
            continue
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    """Parse one manifest-driven referring-expression prediction run."""
    parser = argparse.ArgumentParser(
        description=(
            "Run full referring expressions through a reusable open-vocabulary "
            "or hybrid candidate service and save one response per case."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--backend",
        choices=("grounded_sam_hq", "hybrid"),
        default="grounded_sam_hq",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--part-config", default=DEFAULT_PART_CONFIG)
    parser.add_argument(
        "--part-score-threshold",
        type=float,
        default=None,
        help=(
            "Optional Mask2Former known-part candidate threshold override. "
            "This does not change the Grounding DINO --box-threshold."
        ),
    )
    parser.add_argument("--garment-config", default=DEFAULT_GARMENT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roi-mode", choices=("auto", "full"), default="auto")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--box-threshold", type=float, default=None)
    parser.add_argument("--text-threshold", type=float, default=None)
    parser.add_argument("--max-regions", type=int, default=None)
    parser.add_argument("--subject-roi-margin", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    """Load one model bundle, execute all manifest cases, and save responses."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import (
        resolve_project_path,
        to_project_relative_path,
    )
    from fashion_semantic_parser.dao.localization.referring_smoke import (
        load_referring_smoke_manifest,
    )
    from fashion_semantic_parser.service.region_localization import (
        GroundedSAMHQRegionLocalizationService,
        HybridRegionLocalizationService,
        Mask2FormerPartLocalizationService,
    )
    from fashion_semantic_parser.service.segmentation_runtime import (
        GarmentSegmentationService,
    )

    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least one.")
    settings_overrides = _build_settings_overrides(
        roi_mode=args.roi_mode,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_regions=args.max_regions,
        subject_roi_margin=args.subject_roi_margin,
    )
    part_settings_overrides = _build_part_settings_overrides(
        score_threshold=args.part_score_threshold,
    )
    manifest_path = resolve_project_path(args.manifest)
    output_dir = resolve_project_path(args.output_dir)
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_referring_smoke_manifest(manifest_path)
    missing_images = [
        case.image_path
        for case in manifest.cases
        if not resolve_project_path(case.image_path).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"Referring smoke images are missing: {sorted(set(missing_images))}"
        )
    fallback_service = GroundedSAMHQRegionLocalizationService(
        args.config,
        settings_overrides=settings_overrides,
    )
    if args.backend == "hybrid":
        service: Any = HybridRegionLocalizationService(
            Mask2FormerPartLocalizationService(
                args.part_config,
                settings_overrides=part_settings_overrides,
            ),
            fallback_service,
            garment_segmentation_service=GarmentSegmentationService(
                args.garment_config
            ),
        )
    else:
        service = fallback_service

    response_files: dict[str, str] = {}
    latencies: list[float] = []
    roi_sources: Counter[str] = Counter()
    empty_case_count = 0
    run_started_at = time.perf_counter()
    total = len(manifest.cases)
    for index, case in enumerate(manifest.cases, start=1):
        started_at = time.perf_counter()
        try:
            prediction = service.localize_with_grounding_prompt(
                case.image_path,
                case.query,
                case.grounding_prompt,
                auto_subject_roi=args.roi_mode == "auto",
            )
        except Exception as error:
            failure_path = responses_dir / f"{case.id}.json"
            failure_path.write_text(
                json.dumps(
                    {
                        "case_id": case.id,
                        "query": case.query,
                        "grounding_prompt": case.grounding_prompt,
                        "regions": [],
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"Referring smoke inference failed for case {case.id!r}."
            ) from error
        elapsed_seconds = time.perf_counter() - started_at
        response = build_case_response(
            case=case,
            prediction=prediction,
            elapsed_seconds=elapsed_seconds,
            includes_model_load=index == 1,
        )
        response_path = responses_dir / f"{case.id}.json"
        response_path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        response_files[case.id] = to_project_relative_path(response_path)
        latencies.append(elapsed_seconds)
        roi_sources[prediction.subject_roi_source or "full_image"] += 1
        if not prediction.regions:
            empty_case_count += 1
        if index % args.progress_every == 0 or index == total:
            print(
                f"[{index}/{total}] id={case.id} "
                f"dimensions={','.join(case.dimensions)} "
                f"regions={len(prediction.regions)} "
                f"elapsed={elapsed_seconds:.2f}s",
                flush=True,
            )

    total_elapsed_seconds = time.perf_counter() - run_started_at
    summary = {
        "schema_version": 1,
        "benchmark_scope": "open_language_feasibility_smoke",
        "manifest": to_project_relative_path(manifest_path),
        "manifest_name": manifest.name,
        "backend": args.backend,
        "config": args.config,
        "part_config": args.part_config if args.backend == "hybrid" else None,
        "part_settings_overrides": (
            part_settings_overrides if args.backend == "hybrid" else None
        ),
        "garment_config": args.garment_config if args.backend == "hybrid" else None,
        "roi_mode": args.roi_mode,
        "settings_overrides": settings_overrides,
        "case_count": total,
        "empty_case_count": empty_case_count,
        "empty_case_rate_percent": _percent(empty_case_count, total),
        "roi_source_counts": dict(sorted(roi_sources.items())),
        "latency_seconds": _latency_summary(latencies),
        "total_elapsed_seconds": total_elapsed_seconds,
        "responses": response_files,
        "prd_accuracy_passed": None,
    }
    summary_path = output_dir / "predictions_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def build_case_response(
    *,
    case: Any,
    prediction: Any,
    elapsed_seconds: float,
    includes_model_load: bool,
) -> dict[str, Any]:
    """Attach benchmark metadata to one typed localization prediction."""
    response = cast(dict[str, Any], prediction.model_dump(mode="json"))
    response.update(
        {
            "case_id": case.id,
            "grounding_prompt": case.grounding_prompt,
            "dimensions": case.dimensions,
            "novelty": case.novelty,
            "reference_frame": case.reference_frame,
            "annotation_status": case.annotation_status,
            "expected_count": case.expected_count,
            "contrast_set_id": case.contrast_set_id,
            "elapsed_seconds": float(elapsed_seconds),
            "includes_model_load": includes_model_load,
        }
    )
    return response


def _build_settings_overrides(
    *,
    roi_mode: str,
    box_threshold: float | None,
    text_threshold: float | None,
    max_regions: int | None,
    subject_roi_margin: float | None,
) -> dict[str, Any]:
    """Validate optional smoke-test settings without changing deployment YAML."""
    if roi_mode == "full" and subject_roi_margin is not None:
        raise ValueError("--subject-roi-margin cannot be used with full ROI mode.")
    for name, value in (
        ("box_threshold", box_threshold),
        ("text_threshold", text_threshold),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if max_regions is not None and max_regions < 1:
        raise ValueError("--max-regions must be at least one.")
    if subject_roi_margin is not None and not 0.0 <= subject_roi_margin <= 1.0:
        raise ValueError("--subject-roi-margin must be between 0 and 1.")
    values = {
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "max_regions": max_regions,
        "subject_roi_margin": subject_roi_margin,
    }
    return {key: value for key, value in values.items() if value is not None}


def _build_part_settings_overrides(
    *,
    score_threshold: float | None,
) -> dict[str, Any]:
    """Validate known-part candidate overrides separately from grounding."""
    if score_threshold is not None and not 0.0 <= score_threshold <= 1.0:
        raise ValueError("--part-score-threshold must be between 0 and 1.")
    if score_threshold is None:
        return {}
    return {"score_threshold": score_threshold}


def _latency_summary(values: list[float]) -> dict[str, Any]:
    """Separate first-load wall time from comparable warm-case latency."""
    if not values:
        return {
            "first_case_including_model_load": None,
            "warm_case_count": 0,
            "warm_mean": None,
            "warm_p95": None,
            "all_wall_clock_mean": None,
            "all_wall_clock_p95": None,
        }
    warm_values = values[1:]
    return {
        "first_case_including_model_load": float(values[0]),
        "warm_case_count": len(warm_values),
        "warm_mean": float(np.mean(warm_values)) if warm_values else None,
        "warm_p95": float(np.percentile(warm_values, 95)) if warm_values else None,
        "all_wall_clock_mean": float(np.mean(values)),
        "all_wall_clock_p95": float(np.percentile(values, 95)),
    }


def _percent(numerator: int, denominator: int) -> float | None:
    """Return one bounded percentage without NaN or division by zero."""
    return 100.0 * numerator / denominator if denominator else None


if __name__ == "__main__":
    main()
