"""Train calibrated query-to-patch similarity on Fashionpedia Masks."""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np

DEFAULT_INITIAL_CHECKPOINT = (
    "outputs/localization/dinov2_bge_alignment_train_images300_global/"
    + "alignment_head_smoke.pt"
)


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
    parser.add_argument("--dinov2-config", default=None)
    parser.add_argument("--initial-checkpoint", default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument(
        "--model-type",
        choices=(
            "cosine_calibration",
            "multiscale_decoder",
            "multiscale_area_decoder",
        ),
        default="cosine_calibration",
    )
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
        build_dense_patch_training_cache,
        load_dense_patch_alignment_settings,
    )
    from fashion_semantic_parser.service.dense_patch_area import (
        build_query_area_predictor,
    )
    from fashion_semantic_parser.service.dense_patch_decoder import (
        build_multiscale_patch_decoder,
    )
    from fashion_semantic_parser.service.dense_patch_metrics import (
        patch_probability_metrics,
        select_patch_probability_threshold,
    )
    from fashion_semantic_parser.service.dense_patch_training import (
        DenseTrainingRuntime,
        evaluate_training_cache,
        runtime_loss,
        training_batch,
        training_cache_outputs,
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
    print(
        f"dataset_ready: queries={len(items)} "
        + f"images={len({item.sample.source_image_id for item in items})}"
    )

    extraction_started = time.perf_counter()
    bge_encoder = BgeM3TextEncoder(load_bge_m3_text_settings())
    text_embeddings = bge_encoder.encode([item.sample.query for item in items])
    bge_encoder.synchronize()
    print(f"text_features_ready: shape={tuple(text_embeddings.shape)}")
    dinov2_encoder = DinoV2RegionEncoder(
        load_dinov2_region_settings(
            args.dinov2_config or "configs/localization_dinov2_region.yaml"
        )
    )
    dinov2_input_size = dinov2_encoder.settings.input_size
    cache = build_dense_patch_training_cache(items, dinov2_encoder)
    dinov2_encoder.synchronize()
    feature_extraction_seconds = time.perf_counter() - extraction_started
    print(
        "dense_features_ready: "
        + f"shape={tuple(cache.image_features.shape)} "
        + f"seconds={feature_extraction_seconds:.3f}"
    )
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
    decoder = None
    area_predictor = None
    initial_logit_scale = dense_settings.initial_logit_scale
    if args.model_type in {"multiscale_decoder", "multiscale_area_decoder"}:
        decoder = build_multiscale_patch_decoder(
            alignment_settings.region_dimension,
            dense_settings,
        ).to(device)
        initial_logit_scale = 1.0
    if args.model_type == "multiscale_area_decoder":
        area_predictor = build_query_area_predictor(
            alignment_settings.region_dimension,
            dense_settings,
        ).to(device)
    log_scale = torch.nn.Parameter(
        torch.tensor(
            math.log(initial_logit_scale),
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
        decoder=decoder,
        area_predictor=area_predictor,
        model_type=args.model_type,
        log_scale=log_scale,
        logit_bias=logit_bias,
        text_tensor=text_tensor,
        cache=cache,
        settings=dense_settings,
        device=device,
    )
    initial_summary = evaluate_training_cache(runtime)
    optimization_parameters = [*projection.parameters(), log_scale, logit_bias]
    if decoder is not None:
        optimization_parameters.extend(decoder.parameters())
    if area_predictor is not None:
        optimization_parameters.extend(area_predictor.parameters())
    optimizer = torch.optim.AdamW(
        optimization_parameters,
        lr=dense_settings.learning_rate,
        weight_decay=dense_settings.weight_decay,
    )
    rng = np.random.default_rng(dense_settings.seed)
    training_started = time.perf_counter()
    projection.train()
    if decoder is not None:
        decoder.train()
    if area_predictor is not None:
        area_predictor.train()
    for step in range(1, steps + 1):
        batch_indices = rng.choice(
            query_count,
            size=min(batch_size, query_count),
            replace=False,
        )
        patch_tensor, target_tensor, query_indices = training_batch(
            runtime,
            batch_indices,
        )
        optimizer.zero_grad(set_to_none=True)
        projected = projection(text_tensor[query_indices])
        loss = runtime_loss(runtime, patch_tensor, projected, target_tensor)
        if not torch.isfinite(loss):
            raise RuntimeError("Dense patch training produced a non-finite loss.")
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"[train {step}/{steps}] loss={float(loss.item()):.6f}")
    projection.eval()
    if decoder is not None:
        decoder.eval()
    if area_predictor is not None:
        area_predictor.eval()
    if device == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    final_default_summary = evaluate_training_cache(runtime)
    if cast(float, final_default_summary["mean_loss"]) >= cast(
        float,
        initial_summary["mean_loss"],
    ):
        raise RuntimeError("Dense patch loss did not decrease during training.")
    training_probabilities, training_targets, area_fractions = training_cache_outputs(
        runtime
    )
    selected_threshold = dense_settings.probability_threshold
    threshold_metrics: dict[str, dict[str, float | int]] = {}
    selected_summary = final_default_summary
    mask_selection_mode = "query_area_topk"
    if area_fractions is None:
        selected_threshold, threshold_metrics = select_patch_probability_threshold(
            training_probabilities,
            training_targets,
            dense_settings.calibration_thresholds,
        )
        selected_summary = patch_probability_metrics(
            training_probabilities,
            training_targets,
            threshold=selected_threshold,
        )
        mask_selection_mode = "global_probability_threshold"
    calibrated_settings = dense_settings.model_copy(
        update={"probability_threshold": selected_threshold}
    )
    print(
        "mask_selection_ready: "
        + f"mode={mask_selection_mode} "
        + f"R50={selected_summary['patch_recall50']:.6f} "
        + f"meanIoU={selected_summary['mean_patch_iou']:.6f}"
    )

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "dense_patch_alignment.pt"
    final_scale = min(
        dense_settings.max_logit_scale,
        float(log_scale.detach().exp().cpu().item()),
    )
    final_bias = float(logit_bias.detach().cpu().item())
    checkpoint_payload: dict[str, object] = {
        "schema_version": 3 if area_predictor is not None else 2 if decoder else 1,
        "alignment_settings": alignment_settings.model_dump(mode="json"),
        "dense_settings": calibrated_settings.model_dump(mode="json"),
        "projection_state_dict": projection.state_dict(),
        "logit_scale": final_scale,
        "logit_bias": final_bias,
        "base_encoders_frozen": True,
        "dinov2_model": "dinov2_vits14",
        "text_model": "BAAI/bge-m3",
        "dinov2_input_size": dinov2_input_size,
        "model_type": args.model_type,
        "initial_checkpoint": str(resolve_project_path(args.initial_checkpoint)),
    }
    if decoder is not None:
        checkpoint_payload["decoder_state_dict"] = decoder.state_dict()
    if area_predictor is not None:
        checkpoint_payload["area_predictor_state_dict"] = area_predictor.state_dict()
    torch.save(checkpoint_payload, checkpoint_path)
    metrics = {
        "split": args.split,
        "query_count": query_count,
        "selected_image_count": selected_image_count,
        "image_offset": args.image_offset,
        "selection_scope": "complete_image_prefix",
        "model_type": args.model_type,
        "trainable_parameter_count": sum(
            int(parameter.numel()) for parameter in optimization_parameters
        ),
        "steps": steps,
        "batch_size": min(batch_size, query_count),
        "patch_count_per_image": int(cache.image_features.shape[1]),
        "dinov2_input_size": dinov2_input_size,
        "initial_mean_loss": initial_summary["mean_loss"],
        "final_mean_loss": final_default_summary["mean_loss"],
        "initial_patch_recall50": initial_summary["patch_recall50"],
        "final_default_patch_recall50": final_default_summary["patch_recall50"],
        "final_patch_recall50": selected_summary["patch_recall50"],
        "initial_mean_patch_iou": initial_summary["mean_patch_iou"],
        "final_default_mean_patch_iou": final_default_summary["mean_patch_iou"],
        "final_mean_patch_iou": selected_summary["mean_patch_iou"],
        "default_probability_threshold": dense_settings.probability_threshold,
        "probability_threshold": selected_threshold,
        "mask_selection_mode": mask_selection_mode,
        "threshold_calibration_scope": (
            None if area_fractions is not None else "training_patch_grid"
        ),
        "threshold_metrics": threshold_metrics,
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


if __name__ == "__main__":
    main()
