"""Export DINOv2 patch tokens and verify CUDA/TensorRT numerical parity."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ONNX_PATH = "models/onnx/localization/dinov2_vits14_728.onnx"
DEFAULT_METRICS_PATH = "outputs/localization/dinov2_onnx_export/metrics.json"
DEFAULT_ENGINE_CACHE = "outputs/localization/dinov2_onnx_export/trt_cache"


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse the fixed DINOv2 export and parity paths.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export fixed-spatial, dynamic-batch DINOv2 patch tokens."
    )
    parser.add_argument(
        "--dinov2-config",
        default="configs/localization_dinov2_region_728.yaml",
    )
    parser.add_argument("--output", default=DEFAULT_ONNX_PATH)
    parser.add_argument("--metrics", default=DEFAULT_METRICS_PATH)
    parser.add_argument("--engine-cache", default=DEFAULT_ENGINE_CACHE)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    """Export one model and require CUDA and TensorRT parity.

    Raises:
        RuntimeError: If export, provider execution, or parity validation fails.
        ValueError: If the configured export contract is invalid.
    """
    args = parse_args()
    if args.opset != 17:
        raise ValueError("The PRD DINOv2 export contract requires ONNX opset 17.")
    add_src_to_python_path()
    try:
        import onnx  # type: ignore[import-not-found]
        import onnxruntime as ort  # type: ignore[import-not-found]
        import tensorrt  # type: ignore[import-not-found]  # noqa: F401
        import torch  # type: ignore[import-not-found]
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "The pinned ONNX/TensorRT environment is required."
        ) from error

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.service.dinov2_region_encoder import (
        DinoV2RegionEncoder,
        load_dinov2_region_settings,
    )

    settings = load_dinov2_region_settings(args.dinov2_config)
    if settings.input_size != 728 or settings.device != "cuda":
        raise ValueError("Export requires the frozen 728 CUDA DINOv2 profile.")
    encoder = DinoV2RegionEncoder(settings)
    encoder.load()
    model = encoder._model  # pylint: disable=protected-access
    if model is None:
        raise RuntimeError("DINOv2 model did not initialize for export.")

    class PatchTokenModel(torch.nn.Module):
        """Expose normalized patch tokens as the only ONNX output."""

        def __init__(self, backbone: Any) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, pixel_values: Any) -> Any:
            """Return production-equivalent normalized float patch tokens."""
            output = self.backbone.forward_features(pixel_values)
            tokens = output["x_norm_patchtokens"]
            return torch.nn.functional.normalize(tokens.float(), dim=2)

    wrapper = PatchTokenModel(model).eval().cuda()
    output_path = resolve_project_path(args.output)
    metrics_path = resolve_project_path(args.metrics)
    engine_cache = resolve_project_path(args.engine_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    engine_cache.mkdir(parents=True, exist_ok=True)

    sample = torch.zeros(
        (1, 3, settings.input_size, settings.input_size),
        device="cuda",
        dtype=torch.float32,
    )
    export_started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            sample,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["patch_tokens"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "patch_tokens": {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )
    torch.cuda.synchronize()
    export_seconds = time.perf_counter() - export_started
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    available_providers = ort.get_available_providers()
    required_providers = {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
    if not required_providers.issubset(set(available_providers)):
        raise RuntimeError("Required ONNX Runtime GPU providers are unavailable.")
    cuda_session = ort.InferenceSession(
        str(output_path),
        providers=["CUDAExecutionProvider"],
    )
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = str(metrics_path.parent / "trt_profile")
    spatial_shape = f"3x{settings.input_size}x{settings.input_size}"
    trt_session = ort.InferenceSession(
        str(output_path),
        sess_options=session_options,
        providers=[
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(engine_cache),
                    "trt_profile_min_shapes": f"pixel_values:1x{spatial_shape}",
                    "trt_profile_opt_shapes": f"pixel_values:1x{spatial_shape}",
                    "trt_profile_max_shapes": f"pixel_values:3x{spatial_shape}",
                },
            ),
            "CUDAExecutionProvider",
        ],
    )

    cases: list[dict[str, object]] = []
    all_passed = True
    for batch_size in (1, 3):
        values = np.random.default_rng(batch_size).standard_normal(
            (batch_size, 3, settings.input_size, settings.input_size),
            dtype=np.float32,
        )
        tensor = torch.from_numpy(values).cuda()
        with torch.inference_mode():
            torch_reference = np.asarray(wrapper(tensor).cpu().numpy())
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ),
        ):
            fp16_reference = np.asarray(wrapper(tensor).cpu().numpy())
        cuda_output = np.asarray(
            cuda_session.run(["patch_tokens"], {"pixel_values": values})[0]
        )
        trt_output = np.asarray(
            trt_session.run(["patch_tokens"], {"pixel_values": values})[0]
        )
        cuda_parity = _parity_metrics(torch_reference, cuda_output)
        trt_parity = _parity_metrics(fp16_reference, trt_output)
        passed = (
            cuda_parity["max_abs_error"] <= 5e-4
            and cuda_parity["mean_cosine_similarity"] >= 0.99999
            and trt_parity["max_abs_error"] <= 0.05
            and trt_parity["mean_cosine_similarity"] >= 0.999
        )
        all_passed = all_passed and passed
        cases.append(
            {
                "batch_size": batch_size,
                "output_shape": list(trt_output.shape),
                "cuda_parity": cuda_parity,
                "tensorrt_fp16_parity": trt_parity,
                "passed": passed,
            }
        )
        print(
            f"batch={batch_size} cuda_max={cuda_parity['max_abs_error']:.6f} "
            + f"trt_max={trt_parity['max_abs_error']:.6f} "
            + f"trt_cos={trt_parity['mean_cosine_similarity']:.6f} "
            + f"passed={passed}",
            flush=True,
        )
    profile_path = Path(trt_session.end_profiling())
    tensorrt_executed = _tensorrt_profile_executed(profile_path)
    all_passed = all_passed and tensorrt_executed
    report = {
        "model": settings.model_name,
        "input_size": settings.input_size,
        "dynamic_batch_range": [1, 3],
        "opset": args.opset,
        "onnx_path": str(output_path),
        "onnx_size_bytes": output_path.stat().st_size,
        "export_seconds": export_seconds,
        "available_providers": available_providers,
        "cuda_session_providers": cuda_session.get_providers(),
        "tensorrt_session_providers": trt_session.get_providers(),
        "tensorrt_profile_path": str(profile_path),
        "tensorrt_executed": tensorrt_executed,
        "cases": cases,
        "parity_passed": all_passed,
        "scope": (
            "DINOv2 model export and provider parity only; no complete-request "
            + "accuracy or latency claim."
        ),
    }
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"tensorrt_executed: {tensorrt_executed}", flush=True)
    print(f"parity_passed: {all_passed}", flush=True)
    print(f"onnx_path: {output_path}", flush=True)
    if not all_passed:
        raise RuntimeError("DINOv2 ONNX/TensorRT parity did not pass.")


def _parity_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    """Return absolute error and token-wise cosine similarity.

    Args:
        reference: PyTorch reference patch tokens.
        candidate: Provider patch tokens with the same shape.

    Returns:
        JSON-safe numerical-parity measurements.

    Raises:
        ValueError: If output shapes differ or contain non-finite values.
    """
    if reference.shape != candidate.shape:
        raise ValueError("Provider output shape differs from PyTorch reference.")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Provider parity output contains non-finite values.")
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    reference_rows = reference.reshape(-1, reference.shape[-1]).astype(np.float64)
    candidate_rows = candidate.reshape(-1, candidate.shape[-1]).astype(np.float64)
    denominator = np.linalg.norm(reference_rows, axis=1) * np.linalg.norm(
        candidate_rows,
        axis=1,
    )
    cosine = np.sum(reference_rows * candidate_rows, axis=1) / np.maximum(
        denominator,
        1e-12,
    )
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "mean_cosine_similarity": float(cosine.mean()),
    }


def _tensorrt_profile_executed(profile_path: Path) -> bool:
    """Return whether ORT profiling attributes any node to TensorRT.

    Args:
        profile_path: ONNX Runtime JSON profiling output.

    Returns:
        Whether a TensorRT provider event was recorded.
    """
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, dict):
            continue
        arguments = event.get("args", {})
        provider = arguments.get("provider") if isinstance(arguments, dict) else None
        if provider == "TensorrtExecutionProvider" or "TRTKernel" in str(
            event.get("name", "")
        ):
            return True
    return False


if __name__ == "__main__":
    main()
