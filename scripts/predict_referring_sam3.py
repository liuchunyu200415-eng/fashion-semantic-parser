"""Run a manifest-driven SAM 3 text-prompt localization smoke benchmark."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL_ID = "facebook/sam3"
DEFAULT_OUTPUT_DIR = "outputs/localization/referring_smoke/sam3"


def add_src_to_python_path() -> None:
    """Add the project sources when this script runs in an isolated environment."""
    project_root = Path(__file__).resolve().parents[1]
    for path in (project_root, project_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    """Parse one direct text-prompt SAM 3 benchmark run."""
    parser = argparse.ArgumentParser(
        description=(
            "Run full referring expressions through SAM 3 and save responses "
            "compatible with evaluate_referring_localization.py."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--max-regions", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Load one SAM 3 model, run every expression, and retain misses."""
    args = parse_args()
    add_src_to_python_path()
    _validate_args(args)

    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor, __version__

    from fashion_semantic_parser.common.paths import (
        resolve_project_path,
        to_project_relative_path,
    )
    from fashion_semantic_parser.dao.localization.referring_smoke import (
        load_referring_smoke_manifest,
    )
    from scripts.predict_referring_localization import build_case_response

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

    dtype = _resolve_torch_dtype(args.dtype, torch)
    model_started_at = time.perf_counter()
    processor = Sam3Processor.from_pretrained(
        args.model_id,
        local_files_only=args.offline,
    )
    model = Sam3Model.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        local_files_only=args.offline,
    ).to(args.device)
    model.eval()
    model_load_seconds = time.perf_counter() - model_started_at

    response_files: dict[str, str] = {}
    latencies: list[float] = []
    empty_case_count = 0
    total = len(manifest.cases)
    for index, case in enumerate(manifest.cases, start=1):
        started_at = time.perf_counter()
        try:
            image = Image.open(resolve_project_path(case.image_path)).convert("RGB")
            inputs = processor(
                images=image,
                text=case.grounding_prompt,
                return_tensors="pt",
            )
            target_sizes = inputs.get("original_sizes").tolist()
            model_inputs = _move_inputs(
                inputs, device=args.device, dtype=dtype, torch=torch
            )
            with torch.inference_mode():
                outputs = model(**model_inputs)
            result = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=target_sizes,
            )[0]
            prediction = _result_to_prediction(
                result,
                image_path=case.image_path,
                query=case.query,
                grounding_prompt=case.grounding_prompt,
                image_size=image.size,
                max_regions=args.max_regions,
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
                f"SAM 3 referring inference failed for case {case.id!r}."
            ) from error

        elapsed_seconds = time.perf_counter() - started_at
        response = build_case_response(
            case=case,
            prediction=prediction,
            elapsed_seconds=elapsed_seconds,
            includes_model_load=False,
        )
        response_path = responses_dir / f"{case.id}.json"
        response_path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        response_files[case.id] = to_project_relative_path(response_path)
        latencies.append(elapsed_seconds)
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

    summary = {
        "schema_version": 1,
        "benchmark_scope": "open_language_feasibility_smoke",
        "manifest": to_project_relative_path(manifest_path),
        "manifest_name": manifest.name,
        "backend": "sam3_text_prompt",
        "model_id": args.model_id,
        "device": args.device,
        "dtype": args.dtype,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "max_regions": args.max_regions,
        "offline": args.offline,
        "torch_version": torch.__version__,
        "transformers_version": __version__,
        "model_load_seconds": model_load_seconds,
        "latency_seconds": _latency_summary(latencies),
        "case_count": total,
        "empty_case_count": empty_case_count,
        "empty_case_rate_percent": _percent(empty_case_count, total),
        "responses": response_files,
        "prd_accuracy_passed": None,
    }
    summary_path = output_dir / "predictions_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate inference thresholds before loading a gated heavyweight model."""
    for name in ("score_threshold", "mask_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.max_regions < 1:
        raise ValueError("--max-regions must be at least one.")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least one.")


def _resolve_torch_dtype(name: str, torch: Any) -> Any:
    """Map a stable CLI value to one torch dtype without importing torch globally."""
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def _move_inputs(inputs: Any, *, device: str, dtype: Any, torch: Any) -> dict[str, Any]:
    """Move tensors while casting only floating-point model inputs."""
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _result_to_prediction(
    result: dict[str, Any],
    *,
    image_path: str,
    query: str,
    grounding_prompt: str,
    image_size: tuple[int, int],
    max_regions: int,
) -> Any:
    """Convert one post-processed SAM 3 result to the existing API schema."""
    add_src_to_python_path()
    from fashion_semantic_parser.models.localization import (
        LocalizationBoundingBox,
        LocalizedRegion,
        RegionLocalizationPrediction,
    )
    from fashion_semantic_parser.service.region_localization import _mask_to_polygons

    masks = _to_numpy(result.get("masks"))
    boxes = _to_numpy(result.get("boxes"))
    scores = _to_numpy(result.get("scores"))
    masks = _normalize_masks(masks)
    boxes = (
        np.asarray(boxes, dtype=float).reshape((-1, 4))
        if boxes.size
        else np.empty((0, 4))
    )
    scores = np.asarray(scores, dtype=float).reshape(-1)
    count = min(len(masks), len(boxes), len(scores))
    order = sorted(range(count), key=lambda index: -float(scores[index]))[:max_regions]
    width, height = image_size
    regions: list[LocalizedRegion] = []
    for index in order:
        score = float(scores[index])
        if not math.isfinite(score):
            continue
        x_min, y_min, x_max, y_max = _clamp_box(
            boxes[index],
            width=width,
            height=height,
        )
        if x_max <= x_min or y_max <= y_min:
            continue
        mask = np.asarray(masks[index], dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                "SAM 3 post-processed mask dimensions do not match the image."
            )
        regions.append(
            LocalizedRegion(
                region_label="sam3_text",
                matched_text=grounding_prompt,
                confidence=max(0.0, min(1.0, score)),
                box=LocalizationBoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                ),
                mask=_mask_to_polygons(mask, coordinate_offset=(0.0, 0.0)),
            )
        )
    return RegionLocalizationPrediction(
        image_path=image_path,
        query=query,
        regions=regions,
    )


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors, lists, or absent result fields to NumPy."""
    if value is None:
        return np.asarray([])
    detached = value.detach() if hasattr(value, "detach") else value
    cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
    return np.asarray(cpu_value)


def _normalize_masks(masks: np.ndarray) -> np.ndarray:
    """Normalize SAM 3 masks to N,H,W while retaining empty results."""
    masks = np.asarray(masks)
    if masks.size == 0:
        return np.empty((0, 0, 0), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]
    if masks.ndim != 3:
        raise ValueError("SAM 3 masks must have shape N,H,W or N,1,H,W.")
    return masks.astype(bool)


def _clamp_box(
    box: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Clamp an absolute xyxy result to valid image coordinates."""
    values = np.asarray(box, dtype=float).reshape(-1)
    if len(values) != 4 or not np.isfinite(values).all():
        return 0.0, 0.0, 0.0, 0.0
    return (
        max(0.0, min(float(width), float(values[0]))),
        max(0.0, min(float(height), float(values[1]))),
        max(0.0, min(float(width), float(values[2]))),
        max(0.0, min(float(height), float(values[3]))),
    )


def _latency_summary(values: list[float]) -> dict[str, Any]:
    """Summarize inference latency after model loading has completed."""
    if not values:
        return {"count": 0, "mean": None, "p95": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _percent(numerator: int, denominator: int) -> float | None:
    """Return a JSON-safe percentage."""
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


if __name__ == "__main__":
    main()
