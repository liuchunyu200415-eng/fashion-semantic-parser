"""Fine-tune final DINOv2 blocks with small-part and Copy-Paste supervision."""

# Direct execution adds ``src`` before importing the local package.
# pylint: disable=import-outside-toplevel,duplicate-code

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

DEFAULT_INITIAL_CHECKPOINT = (
    "models/checkpoints/localization/" + "dinov2_multiscale_728_train1000_steps1500.pt"
)


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one bounded backbone fine-tuning run."""
    parser = argparse.ArgumentParser(
        description=(
            "Adapt the final DINOv2 blocks using weighted Fashionpedia Mask "
            "supervision and semantics-safe same-label Copy-Paste."
        )
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--image-limit", type=int, default=20)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--index",
        default=None,
    )
    parser.add_argument("--annotations", default=None)
    parser.add_argument(
        "--config",
        default="configs/localization_dense_patch_finetuning.yaml",
    )
    parser.add_argument(
        "--dinov2-config",
        default="configs/localization_dinov2_region_728.yaml",
    )
    parser.add_argument("--initial-checkpoint", default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        default="outputs/localization/dinov2_backbone_finetune_smoke20",
    )
    parser.add_argument("--disable-copy-paste", action="store_true")
    return parser.parse_args()


# The script keeps all provenance and metric inputs explicit for auditability.
# pylint: disable-next=too-many-locals,too-many-statements,too-many-branches
def main() -> None:
    """Train a schema-four checkpoint without claiming validation accuracy."""
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
        raise RuntimeError("PyTorch is required for DINOv2 fine-tuning.") from error

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
        load_dense_patch_alignment_checkpoint,
        mask_to_patch_fractions,
    )
    from fashion_semantic_parser.service.dense_patch_finetuning import (
        build_copy_paste_donor_groups,
        copy_paste_same_label_instance,
        load_dense_patch_finetuning_settings,
        query_loss_weight,
        select_copy_paste_donor,
    )
    from fashion_semantic_parser.service.dense_patch_training import (
        DenseTrainingRuntime,
        runtime_loss,
    )
    from fashion_semantic_parser.service.dinov2_region_encoder import (
        DinoV2RegionEncoder,
        load_dinov2_region_settings,
    )

    settings = load_dense_patch_finetuning_settings(args.config)
    steps = args.steps or settings.training_steps
    batch_size = args.batch_size or settings.batch_size
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)

    project_settings = load_settings()
    annotation_name = (
        "instances_attributes_train2020.json"
        if args.split == "train"
        else "instances_attributes_val2020.json"
    )
    annotations = args.annotations or (
        f"{project_settings.datasets.fashionpedia_root}/annotations/{annotation_name}"
    )
    index_path = args.index or (
        "data/processed/autodl/localization/"
        + (
            "fashionpedia_referring_train_balanced_100k.jsonl"
            if args.split == "train"
            else "fashionpedia_referring_validation.jsonl"
        )
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
        raise ValueError("DINOv2 fine-tuning loaded no queries.")
    selected_image_count = len({item.sample.source_image_id for item in items})
    print(f"dataset_ready: queries={len(items)} images={selected_image_count}")

    text_encoder = BgeM3TextEncoder(load_bge_m3_text_settings())
    text_embeddings = text_encoder.encode([item.sample.query for item in items])
    text_encoder.synchronize()
    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = load_dense_patch_alignment_checkpoint(
        args.initial_checkpoint,
        device=device,
    )
    if checkpoint.model_type != "multiscale_decoder" or checkpoint.decoder is None:
        raise ValueError(
            "Backbone fine-tuning requires a multiscale decoder checkpoint."
        )
    if checkpoint.dinov2_trainable_state_dict is not None:
        raise ValueError("Initial checkpoint already contains fine-tuned DINOv2 state.")
    if text_embeddings.shape[1] != checkpoint.alignment_settings.text_dimension:
        raise ValueError("Text dimension does not match the initial checkpoint.")

    encoder = DinoV2RegionEncoder(load_dinov2_region_settings(args.dinov2_config))
    if checkpoint.training_input_size != encoder.settings.input_size:
        raise ValueError("Initial checkpoint and DINOv2 input sizes differ.")
    if (
        checkpoint.alignment_settings.region_dimension
        != encoder.settings.feature_dimension
    ):
        raise ValueError("Initial checkpoint and DINOv2 feature dimensions differ.")
    trainable_backbone = encoder.configure_finetuning(settings.unfreeze_last_blocks)
    projection = checkpoint.projection.train()
    decoder = checkpoint.decoder.train()
    text_tensor = torch.from_numpy(text_embeddings).to(device=device)
    log_scale = torch.nn.Parameter(
        torch.tensor(math.log(checkpoint.logit_scale), device=device)
    )
    logit_bias = torch.nn.Parameter(torch.tensor(checkpoint.logit_bias, device=device))
    runtime = DenseTrainingRuntime(
        projection=projection,
        decoder=decoder,
        area_predictor=None,
        model_type="multiscale_decoder",
        log_scale=log_scale,
        logit_bias=logit_bias,
        text_tensor=text_tensor,
        cache=SimpleNamespace(),
        settings=checkpoint.dense_settings,
        device=device,
    )
    head_parameters = [
        *projection.parameters(),
        *decoder.parameters(),
        log_scale,
        logit_bias,
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": settings.head_learning_rate},
            {
                "params": [parameter for _, parameter in trainable_backbone],
                "lr": settings.backbone_learning_rate,
            },
        ],
        weight_decay=settings.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device == "cuda" and encoder.settings.precision == "fp16")
    )
    donor_groups = build_copy_paste_donor_groups(items)
    query_weights = np.asarray(
        [query_loss_weight(item, settings) for item in items],
        dtype=np.float32,
    )
    target_area_fractions = np.asarray(
        [item.target_masks.any(axis=0).mean() for item in items],
        dtype=np.float32,
    )
    rng = np.random.default_rng(settings.seed)
    losses: list[float] = []
    copy_paste_count = 0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch_indices = rng.choice(
            len(items),
            size=min(batch_size, len(items)),
            replace=False,
        )
        images: list[np.ndarray] = []
        target_masks: list[np.ndarray] = []
        for raw_index in batch_indices:
            receiver_index = int(raw_index)
            receiver = items[receiver_index]
            donor_index = None
            if not args.disable_copy_paste:
                donor_index = select_copy_paste_donor(
                    receiver_index,
                    items,
                    donor_groups,
                    settings,
                    rng,
                )
            if donor_index is None:
                images.append(receiver.image_rgb)
                target_masks.append(receiver.target_masks.any(axis=0))
            else:
                image, target_mask = copy_paste_same_label_instance(
                    receiver,
                    items[donor_index],
                    rng,
                )
                images.append(image)
                target_masks.append(target_mask)
                copy_paste_count += 1

        patch_tensor, geometries = encoder.encode_dense_trainable_batch(images)
        target_fractions = np.stack(
            [
                mask_to_patch_fractions(
                    target_mask,
                    geometry,
                    patch_size=encoder.settings.patch_size,
                ).reshape(-1)
                for target_mask, geometry in zip(
                    target_masks,
                    geometries,
                    strict=True,
                )
            ]
        )
        target_tensor = torch.from_numpy(target_fractions).to(device=device)
        query_indices = torch.from_numpy(batch_indices).to(
            device=device,
            dtype=torch.long,
        )
        weight_tensor = torch.from_numpy(query_weights[batch_indices]).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        projected = projection(text_tensor[query_indices])
        loss = runtime_loss(
            runtime,
            patch_tensor,
            projected,
            target_tensor,
            weight_tensor,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("DINOv2 fine-tuning produced a non-finite loss.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [*head_parameters, *[parameter for _, parameter in trainable_backbone]],
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.item()))
        if step == 1 or step % 20 == 0 or step == steps:
            print(f"[train {step}/{steps}] loss={losses[-1]:.6f}")
    encoder.synchronize()
    training_seconds = time.perf_counter() - started

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "dense_patch_alignment.pt"
    final_scale = min(
        checkpoint.dense_settings.max_logit_scale,
        float(log_scale.detach().exp().cpu().item()),
    )
    payload: dict[str, object] = {
        "schema_version": 4,
        "alignment_settings": checkpoint.alignment_settings.model_dump(mode="json"),
        "dense_settings": checkpoint.dense_settings.model_dump(mode="json"),
        "projection_state_dict": projection.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "logit_scale": final_scale,
        "logit_bias": float(logit_bias.detach().cpu().item()),
        "base_encoders_frozen": False,
        "dinov2_model": "dinov2_vits14",
        "text_model": "BAAI/bge-m3",
        "dinov2_input_size": encoder.settings.input_size,
        "model_type": "multiscale_decoder",
        "dinov2_unfrozen_block_count": settings.unfreeze_last_blocks,
        "dinov2_trainable_state_dict": encoder.trainable_state_dict(),
        "initial_checkpoint": str(resolve_project_path(args.initial_checkpoint)),
        "fine_tuning_settings": settings.model_dump(mode="json"),
    }
    torch.save(payload, checkpoint_path)
    metrics: dict[str, Any] = {
        "split": args.split,
        "query_count": len(items),
        "selected_image_count": selected_image_count,
        "image_offset": args.image_offset,
        "selection_scope": "complete_image_prefix",
        "steps": steps,
        "batch_size": min(batch_size, len(items)),
        "initial_step_loss": losses[0],
        "final_step_loss": losses[-1],
        "final_20_step_mean_loss": float(np.mean(losses[-20:])),
        "copy_paste_applied_count": copy_paste_count,
        "small_target_weighted_query_count": int(
            np.sum(target_area_fractions < settings.small_target_area_threshold)
        ),
        "weak_or_small_weighted_query_count": int(np.sum(query_weights > 1.0)),
        "query_loss_weight_min": float(query_weights.min()),
        "query_loss_weight_max": float(query_weights.max()),
        "dinov2_unfrozen_block_count": settings.unfreeze_last_blocks,
        "dinov2_trainable_parameter_count": sum(
            int(parameter.numel()) for _, parameter in trainable_backbone
        ),
        "head_trainable_parameter_count": sum(
            int(parameter.numel()) for parameter in head_parameters
        ),
        "training_seconds": training_seconds,
        "training_loss_only": True,
        "independent_validation_evaluated": False,
        "prd_accuracy_92_passed": None,
        "checkpoint_path": str(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for key in (
        "query_count",
        "selected_image_count",
        "initial_step_loss",
        "final_20_step_mean_loss",
        "copy_paste_applied_count",
        "weak_or_small_weighted_query_count",
        "dinov2_unfrozen_block_count",
        "dinov2_trainable_parameter_count",
        "checkpoint_path",
    ):
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
