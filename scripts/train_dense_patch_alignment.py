"""Train calibrated query-to-patch similarity on Fashionpedia Masks."""

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

DEFAULT_INITIAL_CHECKPOINT = (
    "outputs/localization/dinov2_bge_alignment_train_images300_global/"
    + "alignment_head_smoke.pt"
)


@dataclass
class DenseTrainingRuntime:
    """Tensors and models shared by dense training and audit evaluation."""

    projection: Any
    log_scale: Any
    logit_bias: Any
    text_tensor: Any
    cache: Any
    settings: Any
    device: str


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one image-complete dense patch training run.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train query-calibrated DINOv2 patch similarity."
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--image-limit", type=int, default=100)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--initial-checkpoint", default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        default="outputs/localization/dinov2_dense_patch_alignment_train",
    )
    return parser.parse_args()


def main() -> None:
    """Extract frozen features, train dense calibration, and save a checkpoint.

    Raises:
        ValueError: If selection, feature geometry, or training values are invalid.
        RuntimeError: If PyTorch is unavailable or training becomes non-finite.
    """
    args = parse_args()
    if args.image_limit < 1:
        raise ValueError("--image-limit must be at least one")
    if args.image_offset < 0:
        raise ValueError("--image-offset cannot be negative")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be at least one")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least one")
    add_src_to_python_path()
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for dense patch training.") from error

    from fashion_semantic_parser.common.paths import PROJECT_ROOT, resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.localization.referring_dataset import (
        FashionpediaReferringDataset,
    )
    from fashion_semantic_parser.service.bge_m3_text_encoder import (
        BgeM3TextEncoder,
        load_bge_m3_text_settings,
    )
    from fashion_semantic_parser.service.dense_patch_alignment import (
        balanced_patch_mask_loss,
        build_dense_patch_training_cache,
        dense_patch_logits,
        load_dense_patch_alignment_settings,
    )
    from fashion_semantic_parser.service.dinov2_region_encoder import (
        DinoV2RegionEncoder,
        load_dinov2_region_settings,
    )
    from fashion_semantic_parser.service.region_text_alignment import (
        load_text_projection_checkpoint,
    )

    dense_settings = load_dense_patch_alignment_settings(
        args.config or "configs/localization_dense_patch_alignment.yaml"
    )
    steps = args.steps or dense_settings.training_steps
    batch_size = args.batch_size or dense_settings.batch_size
    torch.manual_seed(dense_settings.seed)
    np.random.seed(dense_settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dense_settings.seed)

    project_settings = load_settings()
    index_path = args.index or (
        "data/processed/autodl/localization/"
        + f"fashionpedia_referring_{args.split}.jsonl"
    )
    annotation_name = (
        "instances_attributes_train2020.json"
        if args.split == "train"
        else "instances_attributes_val2020.json"
    )
    annotations = args.annotations or (
        f"{project_settings.datasets.fashionpedia_root}/annotations/{annotation_name}"
    )
    dataset = FashionpediaReferringDataset(
        index_path=resolve_project_path(index_path),
        annotation_path=resolve_project_path(annotations),
        project_root=PROJECT_ROOT,
        max_images=args.image_limit,
        image_offset=args.image_offset,
    )
    items = [dataset[index] for index in range(len(dataset))]
    if not items:
        raise ValueError("Dense patch training loaded no queries.")

    extraction_started = time.perf_counter()
    bge_encoder = BgeM3TextEncoder(load_bge_m3_text_settings())
    text_embeddings = bge_encoder.encode([item.sample.query for item in items])
    bge_encoder.synchronize()
    dinov2_encoder = DinoV2RegionEncoder(load_dinov2_region_settings())
    cache = build_dense_patch_training_cache(items, dinov2_encoder)
    dinov2_encoder.synchronize()
    feature_extraction_seconds = time.perf_counter() - extraction_started
    query_count = len(items)
    selected_image_count = len(cache.image_ids)
    del items, dataset, bge_encoder, dinov2_encoder
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    projection, alignment_settings = load_text_projection_checkpoint(
        args.initial_checkpoint,
        device=device,
    )
    if text_embeddings.shape[1] != alignment_settings.text_dimension:
        raise ValueError("Text dimension does not match the initial checkpoint.")
    if cache.image_features.shape[2] != alignment_settings.region_dimension:
        raise ValueError("Patch dimension does not match the initial checkpoint.")
    if dense_settings.initial_logit_scale > dense_settings.max_logit_scale:
        raise ValueError("Initial logit scale exceeds its configured maximum.")
    text_tensor = torch.from_numpy(text_embeddings).to(device=device)
    log_scale = torch.nn.Parameter(
        torch.tensor(
            math.log(dense_settings.initial_logit_scale),
            dtype=torch.float32,
            device=device,
        )
    )
    logit_bias = torch.nn.Parameter(
        torch.tensor(
            dense_settings.initial_logit_bias,
            dtype=torch.float32,
            device=device,
        )
    )
    runtime = DenseTrainingRuntime(
        projection=projection,
        log_scale=log_scale,
        logit_bias=logit_bias,
        text_tensor=text_tensor,
        cache=cache,
        settings=dense_settings,
        device=device,
    )
    initial_summary = _evaluate_training_cache(runtime)
    optimizer = torch.optim.AdamW(
        [*projection.parameters(), log_scale, logit_bias],
        lr=dense_settings.learning_rate,
        weight_decay=dense_settings.weight_decay,
    )
    rng = np.random.default_rng(dense_settings.seed)
    training_started = time.perf_counter()
    projection.train()
    for _ in range(steps):
        batch_indices = rng.choice(
            query_count,
            size=min(batch_size, query_count),
            replace=False,
        )
        patch_tensor, target_tensor, query_indices = _training_batch(
            runtime,
            batch_indices,
        )
        optimizer.zero_grad(set_to_none=True)
        projected = projection(text_tensor[query_indices])
        logits = dense_patch_logits(
            patch_tensor,
            projected,
            log_scale,
            logit_bias,
            max_logit_scale=dense_settings.max_logit_scale,
        )
        loss = balanced_patch_mask_loss(logits, target_tensor)
        if not torch.isfinite(loss):
            raise RuntimeError("Dense patch training produced a non-finite loss.")
        loss.backward()
        optimizer.step()
    projection.eval()
    if device == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    final_summary = _evaluate_training_cache(runtime)
    if cast(float, final_summary["mean_loss"]) >= cast(
        float,
        initial_summary["mean_loss"],
    ):
        raise RuntimeError("Dense patch loss did not decrease during training.")

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "dense_patch_alignment.pt"
    final_scale = min(
        dense_settings.max_logit_scale,
        float(log_scale.detach().exp().cpu().item()),
    )
    final_bias = float(logit_bias.detach().cpu().item())
    torch.save(
        {
            "schema_version": 1,
            "alignment_settings": alignment_settings.model_dump(mode="json"),
            "dense_settings": dense_settings.model_dump(mode="json"),
            "projection_state_dict": projection.state_dict(),
            "logit_scale": final_scale,
            "logit_bias": final_bias,
            "base_encoders_frozen": True,
            "dinov2_model": "dinov2_vits14",
            "text_model": "BAAI/bge-m3",
            "initial_checkpoint": str(resolve_project_path(args.initial_checkpoint)),
        },
        checkpoint_path,
    )
    metrics = {
        "split": args.split,
        "query_count": query_count,
        "selected_image_count": selected_image_count,
        "image_offset": args.image_offset,
        "selection_scope": "complete_image_prefix",
        "steps": steps,
        "batch_size": min(batch_size, query_count),
        "patch_count_per_image": int(cache.image_features.shape[1]),
        "initial_mean_loss": initial_summary["mean_loss"],
        "final_mean_loss": final_summary["mean_loss"],
        "initial_patch_recall50": initial_summary["patch_recall50"],
        "final_patch_recall50": final_summary["patch_recall50"],
        "initial_mean_patch_iou": initial_summary["mean_patch_iou"],
        "final_mean_patch_iou": final_summary["mean_patch_iou"],
        "probability_threshold": dense_settings.probability_threshold,
        "logit_scale": final_scale,
        "logit_bias": final_bias,
        "feature_extraction_seconds": feature_extraction_seconds,
        "projection_training_seconds": training_seconds,
        "base_encoders_frozen": True,
        "training_patch_grid_metrics_only": True,
        "full_image_mask_localization_evaluated": False,
        "independent_manual_test_set": False,
        "prd_accuracy_92_passed": None,
        "prd_localization_30ms_passed": None,
        "checkpoint_path": str(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")


def _training_batch(
    runtime: DenseTrainingRuntime,
    batch_indices: np.ndarray,
) -> tuple[Any, Any, Any]:
    """Move one sampled query batch and its source-image patches to the device."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for dense patch batching.") from error
    query_indices = torch.from_numpy(batch_indices).to(
        device=runtime.device,
        dtype=torch.long,
    )
    image_indices = runtime.cache.query_image_indices[batch_indices]
    patch_tensor = torch.from_numpy(runtime.cache.image_features[image_indices]).to(
        device=runtime.device
    )
    target_tensor = torch.from_numpy(
        runtime.cache.target_patch_fractions[batch_indices]
    ).to(device=runtime.device)
    return patch_tensor, target_tensor, query_indices


def _evaluate_training_cache(
    runtime: DenseTrainingRuntime,
) -> dict[str, float | int]:
    """Measure loss and patch-grid IoU over every retained training query."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for dense patch evaluation.") from error
    from fashion_semantic_parser.service.dense_patch_alignment import (
        balanced_patch_mask_loss,
        dense_patch_logits,
    )

    runtime.projection.eval()
    losses: list[float] = []
    ious: list[float] = []
    query_count = len(runtime.cache.query_image_indices)
    with torch.inference_mode():
        for start in range(0, query_count, runtime.settings.batch_size):
            stop = min(query_count, start + runtime.settings.batch_size)
            batch_indices = np.arange(start, stop, dtype=np.int64)
            patch_tensor, target_tensor, query_indices = _training_batch(
                runtime,
                batch_indices,
            )
            projected = runtime.projection(runtime.text_tensor[query_indices])
            logits = dense_patch_logits(
                patch_tensor,
                projected,
                runtime.log_scale,
                runtime.logit_bias,
                max_logit_scale=runtime.settings.max_logit_scale,
            )
            loss = balanced_patch_mask_loss(logits, target_tensor)
            losses.extend([float(loss.item())] * len(batch_indices))
            predicted = torch.sigmoid(logits) >= runtime.settings.probability_threshold
            target = target_tensor > 0.0
            intersection = torch.logical_and(predicted, target).sum(dim=1)
            union = torch.logical_or(predicted, target).sum(dim=1)
            batch_ious = intersection.float() / union.clamp(min=1).float()
            ious.extend(float(value) for value in batch_ious.cpu().tolist())
    return {
        "query_count": query_count,
        "mean_loss": float(np.mean(losses)),
        "patch_recall50_count": sum(value >= 0.50 for value in ious),
        "patch_recall50": float(np.mean(np.asarray(ious) >= 0.50)),
        "mean_patch_iou": float(np.mean(ious)),
    }


if __name__ == "__main__":
    main()
