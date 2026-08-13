"""Benchmark warm PRD 3.1.2 service latency across deployed query routes."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

DEFAULT_VALIDATION_JSON = (
    "data/processed/autodl/localization/fashionpedia_parts_validation.json"
)
DEFAULT_QUERIES = (
    "衣领",
    "口袋",
    "肩部",
    "装饰",
    "袖口",
    "下摆",
    "腰部",
    "图案",
)
LATENCY_TARGET_MS = 30.0


def add_src_to_python_path() -> None:
    """Add the local package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse warm localization latency arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure warm service-call latency. Model loading is excluded; path "
            "validation, image decode, person ROI, inference, and postprocessing "
            "are included."
        )
    )
    parser.add_argument("images", nargs="*")
    parser.add_argument("--val-json", default=DEFAULT_VALIDATION_JSON)
    parser.add_argument("--image-limit", type=int, default=20)
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Repeat to benchmark selected queries; defaults to all eight PRD regions.",
    )
    parser.add_argument(
        "--backend",
        choices=["dense_local_reencoding", "supervised", "hybrid"],
        default="dense_local_reencoding",
    )
    parser.add_argument(
        "--part-config",
        default="configs/localization_mask2former_parts_deployment.yaml",
    )
    parser.add_argument(
        "--garment-config",
        default="configs/segmentation_mask2former_deployment.yaml",
    )
    parser.add_argument(
        "--fallback-config",
        default="configs/localization_grounded_sam_hq.yaml",
    )
    parser.add_argument(
        "--dense-config",
        default="configs/localization_dense_local_reencoding.yaml",
    )
    parser.add_argument(
        "--roi-mode",
        choices=["full", "auto"],
        default="auto",
    )
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    """Load one service, warm each route, and report central and tail latency."""
    args = parse_args()
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.dao.localization.taxonomy import (
        resolve_localization_prompt,
    )
    from fashion_semantic_parser.service.dense_local_reencoding import (
        DenseLocalReencodingRegionLocalizationService,
    )
    from fashion_semantic_parser.service.region_localization import (
        GroundedSAMHQRegionLocalizationService,
        HybridRegionLocalizationService,
        Mask2FormerPartLocalizationService,
    )
    from fashion_semantic_parser.service.segmentation_runtime import (
        GarmentSegmentationService,
    )

    if args.image_limit < 1:
        raise ValueError("--image-limit must be at least one.")
    if args.warmup_runs < 1:
        raise ValueError("--warmup-runs must be at least one.")
    if args.runs < 1:
        raise ValueError("--runs must be at least one.")

    image_paths = _resolve_benchmark_images(
        explicit_images=args.images,
        validation_path=_resolve_path(args.val_json, resolve_project_path),
        image_limit=args.image_limit,
    )
    service: Any
    if args.backend == "dense_local_reencoding":
        service = DenseLocalReencodingRegionLocalizationService(args.dense_config)
    else:
        supervised = Mask2FormerPartLocalizationService(args.part_config)
        service = supervised
        if args.backend == "hybrid":
            service = HybridRegionLocalizationService(
                supervised,
                GroundedSAMHQRegionLocalizationService(args.fallback_config),
                garment_segmentation_service=GarmentSegmentationService(
                    args.garment_config
                ),
            )

    import torch

    queries = args.query or list(DEFAULT_QUERIES)
    route_results = []
    all_latencies: list[float] = []
    for query in queries:
        prompt = resolve_localization_prompt(query)
        operation = _localization_operation(
            service=service,
            image_paths=image_paths,
            query=query,
            auto_subject_roi=args.roi_mode == "auto",
        )
        for run_index in range(args.warmup_runs):
            operation(run_index)
        _synchronize_cuda(torch)

        latencies = []
        for run_index in range(args.runs):
            _synchronize_cuda(torch)
            started_at = time.perf_counter()
            prediction = operation(run_index)
            _synchronize_cuda(torch)
            latencies.append((time.perf_counter() - started_at) * 1000.0)
        all_latencies.extend(latencies)
        summary = _latency_summary(latencies)
        route_results.append(
            {
                "query": query,
                "resolved_region": prompt.region_label,
                "last_region_count": len(prediction.regions),
                "latency_ms": summary,
                "p95_target_ms": LATENCY_TARGET_MS,
                "passed": summary["p95"] <= LATENCY_TARGET_MS,
            }
        )
        print(
            f"{prompt.region_label}: mean={summary['mean']:.2f}ms "
            f"p95={summary['p95']:.2f}ms passed={summary['p95'] <= LATENCY_TARGET_MS}",
            flush=True,
        )

    aggregate = _latency_summary(all_latencies)
    result = {
        "benchmark": "prd_3_1_2_warm_service_latency",
        "backend": args.backend,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "gpu_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "image_count": len(image_paths),
        "warmup_runs_per_query": args.warmup_runs,
        "measured_runs_per_query": args.runs,
        "roi_mode": args.roi_mode,
        "measurement_boundary": {
            "included": [
                "project path validation",
                "image decode",
                "person ROI when enabled",
                "model inference",
                "mask and box postprocessing",
            ],
            "excluded": [
                "model and weight loading",
                "HTTP transport",
                "JSON serialization",
            ],
        },
        "target_ms": LATENCY_TARGET_MS,
        "all_routes_passed": all(route["passed"] for route in route_results),
        "aggregate_latency_ms": aggregate,
        "routes": route_results,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        output_path = _resolve_path(args.output, resolve_project_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")


def _resolve_benchmark_images(
    *,
    explicit_images: list[str],
    validation_path: Path,
    image_limit: int,
) -> list[str]:
    """Return explicit paths or a deterministic validation sample."""
    if explicit_images:
        return explicit_images
    with validation_path.open("r", encoding="utf-8") as file:
        source = json.load(file)
    images = source.get("images") if isinstance(source, dict) else None
    if not isinstance(images, list) or not images:
        raise ValueError(f"No validation images found: {validation_path}")
    selected = sorted(images, key=lambda image: int(image["id"]))[:image_limit]
    paths = [str(image["file_name"]) for image in selected]
    if not all(paths):
        raise ValueError(f"Invalid image file_name in {validation_path}")
    return paths


def _localization_operation(
    *,
    service: Any,
    image_paths: list[str],
    query: str,
    auto_subject_roi: bool,
) -> Callable[[int], Any]:
    """Build a stable round-robin service operation for warm timing."""

    def operation(run_index: int) -> Any:
        return service.localize(
            image_paths[run_index % len(image_paths)],
            query,
            auto_subject_roi=auto_subject_roi,
        )

    return operation


def _synchronize_cuda(torch: Any) -> None:
    """Wait for queued kernels when CUDA is active."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    """Summarize measured milliseconds with central and tail latency."""
    if not latencies_ms:
        raise ValueError("At least one latency sample is required.")
    values = np.asarray(latencies_ms, dtype=float)
    mean_ms = float(np.mean(values))
    return {
        "mean": mean_ms,
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "requests_per_second_from_mean": float(1000.0 / mean_ms),
    }


def _resolve_path(path: str, resolver: Any) -> Path:
    """Use absolute paths directly and resolve project-relative paths."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolver(candidate)


if __name__ == "__main__":
    main()
