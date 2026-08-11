"""Smoke-train a BGE-M3-to-DINOv2 projection on Fashionpedia regions."""

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse a deliberately bounded alignment training smoke."""
    parser = argparse.ArgumentParser(
        description="Train only the BGE-M3-to-DINOv2 alignment projection."
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/localization/dinov2_bge_alignment_smoke",
    )
    return parser.parse_args()


def main() -> None:
    """Extract frozen features once, train one small head, and save evidence."""
    args = parse_args()
    if args.limit < 2:
        raise ValueError("--limit must be at least two")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be at least one")
    add_src_to_python_path()
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for alignment training.") from error

    from fashion_semantic_parser.common.paths import PROJECT_ROOT, resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.localization.referring_dataset import (
        FashionpediaReferringDataset,
    )
    from fashion_semantic_parser.service.bge_m3_text_encoder import (
        BgeM3TextEncoder,
        load_bge_m3_text_settings,
    )
    from fashion_semantic_parser.service.dinov2_region_encoder import (
        DinoV2RegionEncoder,
        load_dinov2_region_settings,
    )
    from fashion_semantic_parser.service.region_text_alignment import (
        build_positive_region_mask,
        build_text_projection,
        load_region_text_alignment_settings,
        multi_positive_contrastive_loss,
        positive_top1_accuracy,
    )

    alignment_settings = load_region_text_alignment_settings()
    steps = args.steps or alignment_settings.training_steps
    torch.manual_seed(alignment_settings.seed)
    np.random.seed(alignment_settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(alignment_settings.seed)

    project_settings = load_settings()
    index = args.index or (
        "data/processed/autodl/localization/"
        f"fashionpedia_referring_{args.split}.jsonl"
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
        index_path=resolve_project_path(index),
        annotation_path=resolve_project_path(annotations),
        project_root=PROJECT_ROOT,
        max_samples=args.limit,
    )
    items = [dataset[index] for index in range(len(dataset))]
    if len(items) < 2:
        raise ValueError("Alignment smoke requires at least two dataset samples.")

    extraction_started = time.perf_counter()
    bge_encoder = BgeM3TextEncoder(load_bge_m3_text_settings())
    text_embeddings = bge_encoder.encode([item.sample.query for item in items])
    bge_encoder.synchronize()

    dinov2_encoder = DinoV2RegionEncoder(load_dinov2_region_settings())
    region_features_by_id = extract_unique_region_features(items, dinov2_encoder)
    dinov2_encoder.synchronize()
    extraction_seconds = time.perf_counter() - extraction_started

    region_annotation_ids = sorted(region_features_by_id)
    region_embeddings = np.stack(
        [
            region_features_by_id[annotation_id]
            for annotation_id in region_annotation_ids
        ]
    )
    if text_embeddings.shape[1] != alignment_settings.text_dimension:
        raise ValueError(
            "BGE-M3 feature dimension does not match the alignment config: "
            f"features={text_embeddings.shape[1]} "
            f"config={alignment_settings.text_dimension}"
        )
    if region_embeddings.shape[1] != alignment_settings.region_dimension:
        raise ValueError(
            "DINOv2 feature dimension does not match the alignment config: "
            f"features={region_embeddings.shape[1]} "
            f"config={alignment_settings.region_dimension}"
        )
    query_annotation_ids = [item.source_annotation_ids for item in items]
    positive_mask_array = build_positive_region_mask(
        query_annotation_ids,
        region_annotation_ids,
    )
    negative_pair_count = int((~positive_mask_array).sum())
    if negative_pair_count == 0:
        raise ValueError(
            "Alignment smoke requires at least one non-matching query-region pair."
        )

    del bge_encoder, dinov2_encoder
    gc.collect()
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    projection = build_text_projection(alignment_settings).to(device)
    optimizer = torch.optim.AdamW(
        projection.parameters(),
        lr=alignment_settings.learning_rate,
        weight_decay=alignment_settings.weight_decay,
    )
    text_tensor = torch.from_numpy(text_embeddings).to(device=device)
    region_tensor = torch.from_numpy(region_embeddings).to(device=device)
    positive_mask = torch.from_numpy(positive_mask_array).to(device=device)

    projection.train()
    with torch.no_grad():
        initial_loss_tensor, initial_logits = multi_positive_contrastive_loss(
            projection(text_tensor),
            region_tensor,
            positive_mask,
            temperature=alignment_settings.temperature,
        )
        initial_loss = float(initial_loss_tensor.item())
        initial_top1 = positive_top1_accuracy(initial_logits, positive_mask)

    training_started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = multi_positive_contrastive_loss(
            projection(text_tensor),
            region_tensor,
            positive_mask,
            temperature=alignment_settings.temperature,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Alignment smoke produced a non-finite loss.")
        loss.backward()
        optimizer.step()
    projection.eval()
    with torch.no_grad():
        final_loss_tensor, final_logits = multi_positive_contrastive_loss(
            projection(text_tensor),
            region_tensor,
            positive_mask,
            temperature=alignment_settings.temperature,
        )
        final_loss = float(final_loss_tensor.item())
    if device == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    final_top1 = positive_top1_accuracy(final_logits, positive_mask)
    loss_decreased = final_loss < initial_loss
    if not loss_decreased:
        raise RuntimeError(
            "Alignment loss did not decrease during the bounded smoke: "
            f"initial={initial_loss:.6f} final={final_loss:.6f}"
        )

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "alignment_head_smoke.pt"
    torch.save(
        {
            "schema_version": 1,
            "alignment_settings": alignment_settings.model_dump(mode="json"),
            "state_dict": projection.state_dict(),
            "base_encoders_frozen": True,
            "dinov2_model": "dinov2_vits14",
            "text_model": "BAAI/bge-m3",
        },
        checkpoint_path,
    )
    metrics = {
        "split": args.split,
        "sample_count": len(items),
        "unique_region_count": len(region_annotation_ids),
        "positive_pair_count": int(positive_mask_array.sum()),
        "negative_pair_count": negative_pair_count,
        "text_dimension": int(text_embeddings.shape[1]),
        "region_dimension": int(region_embeddings.shape[1]),
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_query_top1": initial_top1,
        "final_query_top1": final_top1,
        "loss_decreased": loss_decreased,
        "feature_extraction_seconds": extraction_seconds,
        "projection_training_seconds": training_seconds,
        "base_encoders_frozen": True,
        "prd_accuracy_92_passed": None,
        "prd_localization_30ms_passed": None,
        "checkpoint_path": str(checkpoint_path),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")


def extract_unique_region_features(
    items: list[Any],
    encoder: Any,
) -> dict[int, np.ndarray]:
    """Encode each unique source Mask once, grouped by source image."""
    grouped: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"image_rgb": None, "masks": {}}
    )
    for item in items:
        image_id = item.sample.source_image_id
        group = grouped[image_id]
        if group["image_rgb"] is None:
            group["image_rgb"] = item.image_rgb
        elif not np.array_equal(group["image_rgb"], item.image_rgb):
            raise ValueError(f"Image {image_id} decoded inconsistently.")
        for annotation_id, mask in zip(
            item.source_annotation_ids,
            item.target_masks,
        ):
            existing = group["masks"].get(annotation_id)
            if existing is not None and not np.array_equal(existing, mask):
                raise ValueError(f"Annotation {annotation_id} decoded inconsistently.")
            group["masks"][annotation_id] = mask

    features_by_id: dict[int, np.ndarray] = {}
    for group in grouped.values():
        annotation_ids = sorted(group["masks"])
        masks = np.stack([group["masks"][value] for value in annotation_ids])
        features = encoder.encode(group["image_rgb"], masks)
        for annotation_id, feature in zip(annotation_ids, features):
            features_by_id[annotation_id] = np.asarray(feature, dtype=np.float32)
    return features_by_id


if __name__ == "__main__":
    main()
