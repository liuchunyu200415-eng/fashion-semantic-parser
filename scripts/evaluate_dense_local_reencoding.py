"""Evaluate category-free coarse-to-fine DINOv2 local re-encoding."""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np

DEFAULT_CHECKPOINT = (
    "outputs/localization/dinov2_multiscale_728_train1000_steps1500/"
    + "dense_patch_alignment.pt"
)


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one frozen coarse-to-fine local re-encoding evaluation.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate query-driven local DINOv2 re-encoding."
    )
    parser.add_argument(
        "--split", choices=("train", "validation"), default="validation"
    )
    parser.add_argument("--image-limit", type=int, default=2)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--index", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dinov2-config",
        default="configs/localization_dinov2_region_728.yaml",
    )
    parser.add_argument("--crop-fraction", type=float, default=0.30)
    parser.add_argument("--max-crops", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="outputs/localization/dense_local_reencoding_eval",
    )
    return parser.parse_args()


def main() -> None:
    """Run frozen full-query coarse and local score fusion without GT tuning.

    Raises:
        ValueError: If data selection, model, crop, or feature geometry is invalid.
        RuntimeError: If PyTorch is unavailable.
    """
    args = parse_args()
    if (
        args.image_limit < 1
        or args.image_offset < 0
        or not 0.0 < args.crop_fraction <= 1.0
        or args.max_crops < 1
    ):
        raise ValueError("Local re-encoding selection or crop settings are invalid.")
    add_src_to_python_path()
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for local re-encoding.") from error

    from fashion_semantic_parser.common.paths import PROJECT_ROOT, resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.localization.referring_dataset import (
        FashionpediaReferringDataset,
    )
    from fashion_semantic_parser.service.bge_m3_text_encoder import (
        BgeM3TextEncoder,
        load_bge_m3_text_settings,
    )
    from fashion_semantic_parser.service.dense_crop_audit import (
        extract_crop_image,
        fuse_crop_score_maps,
        restore_crop_score_map,
        select_query_peak_crops,
    )
    from fashion_semantic_parser.service.dense_patch_alignment import (
        apply_finetuned_dinov2_checkpoint,
        load_dense_patch_alignment_checkpoint,
    )
    from fashion_semantic_parser.service.dense_patch_inference import (
        predict_patch_outputs,
    )
    from fashion_semantic_parser.service.dense_patch_metrics import write_dense_json
    from fashion_semantic_parser.service.dense_region_localization import (
        binary_mask_iou,
        box_iou,
        mask_box,
    )
    from fashion_semantic_parser.service.dinov2_region_encoder import (
        DinoV2RegionEncoder,
        load_dinov2_region_settings,
        patch_scores_to_image,
    )

    settings = load_settings()
    index_path = args.index or (
        "data/processed/autodl/localization/"
        + f"fashionpedia_referring_{args.split}.jsonl"
    )
    annotation_name = (
        "instances_attributes_train2020.json"
        if args.split == "train"
        else "instances_attributes_val2020.json"
    )
    annotation_path = args.annotations or (
        f"{settings.datasets.fashionpedia_root}/annotations/{annotation_name}"
    )
    dataset = FashionpediaReferringDataset(
        index_path=resolve_project_path(index_path),
        annotation_path=resolve_project_path(annotation_path),
        project_root=PROJECT_ROOT,
        max_images=args.image_limit,
        image_offset=args.image_offset,
    )
    items = [dataset[index] for index in range(len(dataset))]
    if not items:
        raise ValueError("Local re-encoding evaluation loaded no queries.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = load_dense_patch_alignment_checkpoint(
        args.checkpoint,
        device=device,
    )
    text_encoder = BgeM3TextEncoder(load_bge_m3_text_settings())
    embeddings = text_encoder.encode([item.sample.query for item in items])
    text_encoder.synchronize()
    with torch.inference_mode():
        projected = checkpoint.projection(
            torch.from_numpy(embeddings).to(device=device)
        )
        projected = torch.nn.functional.normalize(projected.float(), dim=1)
        projected_text = np.asarray(projected.cpu().numpy(), dtype=np.float32)
    image_encoder = DinoV2RegionEncoder(load_dinov2_region_settings(args.dinov2_config))
    apply_finetuned_dinov2_checkpoint(image_encoder, checkpoint)
    groups: dict[int, list[int]] = defaultdict(list)
    for item_index, item in enumerate(items):
        groups[item.sample.source_image_id].append(item_index)
    cases: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    threshold = checkpoint.dense_settings.probability_threshold
    for image_number, (image_id, item_indices) in enumerate(groups.items(), start=1):
        image_started = time.perf_counter()
        source_image = items[item_indices[0]].image_rgb
        dense = image_encoder.encode_dense(source_image)
        coarse_probabilities, _ = predict_patch_outputs(
            checkpoint,
            dense.features,
            projected_text[item_indices],
            device,
        )
        for local_index, item_index in enumerate(item_indices):
            item = items[item_index]
            coarse_map = patch_scores_to_image(
                coarse_probabilities[local_index],
                dense.geometry,
            )
            crops = select_query_peak_crops(
                coarse_probabilities[local_index],
                dense.geometry,
                crop_fraction=args.crop_fraction,
                max_crops=args.max_crops,
            )
            local_maps: list[np.ndarray] = []
            for crop in crops:
                crop_image = extract_crop_image(source_image, crop)
                crop_dense = image_encoder.encode_dense(crop_image)
                crop_probabilities, _ = predict_patch_outputs(
                    checkpoint,
                    crop_dense.features,
                    projected_text[item_index : item_index + 1],
                    device,
                )
                crop_map = patch_scores_to_image(
                    crop_probabilities[0],
                    crop_dense.geometry,
                )
                local_maps.append(
                    restore_crop_score_map(
                        crop_map,
                        crop,
                        source_image.shape[:2],
                    )
                )
            local_map = fuse_crop_score_maps(local_maps)
            fused_map = np.maximum(coarse_map, local_map)
            target_mask = np.asarray(item.target_masks.any(axis=0), dtype=bool)
            row: dict[str, object] = {
                "query_id": item.sample.id,
                "query": item.sample.query,
                "language": item.sample.language,
                "dimensions": list(item.sample.dimensions),
                "target_label": item.sample.target_label,
                "source_image_id": image_id,
                "target_annotation_ids": list(item.source_annotation_ids),
                "target_count": len(item.source_annotation_ids),
                "target_area": int(target_mask.sum()),
                "crop_count": len(crops),
                "crops": [
                    {
                        "x_min": crop.x_min,
                        "y_min": crop.y_min,
                        "x_max": crop.x_max,
                        "y_max": crop.y_max,
                    }
                    for crop in crops
                ],
            }
            for name, score_map in (
                ("coarse", coarse_map),
                ("local_only", local_map),
                ("coarse_local_max", fused_map),
            ):
                predicted_mask = np.asarray(score_map >= threshold, dtype=bool)
                row[f"{name}_mask_iou"] = binary_mask_iou(
                    target_mask,
                    predicted_mask,
                )
                row[f"{name}_box_iou"] = box_iou(
                    mask_box(target_mask),
                    mask_box(predicted_mask),
                )
                row[f"{name}_predicted_area"] = int(predicted_mask.sum())
            cases.append(row)
        elapsed_seconds = time.perf_counter() - image_started
        image_rows.append(
            {
                "source_image_id": image_id,
                "query_count": len(item_indices),
                "elapsed_seconds": elapsed_seconds,
            }
        )
        print(
            f"[{image_number}/{len(groups)}] image={image_id} "
            + f"queries={len(item_indices)} elapsed={elapsed_seconds:.3f}s"
        )
    summary = _summarize(cases)
    summary.update(
        {
            "split": args.split,
            "image_offset": args.image_offset,
            "selected_image_count": len(groups),
            "checkpoint_path": str(resolve_project_path(args.checkpoint)),
            "checkpoint_training_input_size": checkpoint.training_input_size,
            "dinov2_input_size": image_encoder.settings.input_size,
            "crop_fraction": args.crop_fraction,
            "max_crops": args.max_crops,
            "crop_generation_uses_ground_truth": False,
            "local_reencoding_uses_ground_truth": False,
            "uses_fixed_part_categories": False,
            "full_language_query_used": True,
            "threshold_source": "frozen_training_checkpoint",
            "latency_scope": "offline_batched_audit_not_complete_request_latency",
            "first_image_including_image_model_load_seconds": cast(
                float,
                image_rows[0]["elapsed_seconds"],
            ),
            "warm_image_count": max(0, len(image_rows) - 1),
            "warm_mean_image_seconds": _warm_mean_seconds(image_rows),
            "independent_manual_test_set": False,
            "prd_accuracy_92_passed": None,
            "prd_localization_30ms_passed": None,
        }
    )
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_dense_json(output_dir / "metrics.json", summary)
    write_dense_json(output_dir / "cases.json", cases)
    print(f"query_count: {summary['query_count']}")
    for name in ("coarse", "local_only", "coarse_local_max"):
        row = cast(dict[str, object], summary[name])
        print(
            f"{name}: R50={cast(float, row['mask_recall50']):.6f} "
            + f"R75={cast(float, row['mask_recall75']):.6f} "
            + f"IoU={cast(float, row['mean_mask_iou']):.6f} "
            + f"BoxR50={cast(float, row['box_recall50']):.6f}"
        )


def _summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate all fixed coarse/local branches with retained misses."""
    if not cases:
        raise ValueError("Local re-encoding summary requires at least one case.")
    result: dict[str, object] = {"query_count": len(cases)}
    for name in ("coarse", "local_only", "coarse_local_max"):
        mask_ious = np.asarray(
            [cast(float, case[f"{name}_mask_iou"]) for case in cases]
        )
        box_ious = np.asarray([cast(float, case[f"{name}_box_iou"]) for case in cases])
        area_ratios = np.asarray(
            [
                cast(int, case[f"{name}_predicted_area"])
                / cast(int, case["target_area"])
                for case in cases
            ],
            dtype=np.float64,
        )
        result[name] = {
            "query_count": len(cases),
            "mask_recall50_count": int(np.sum(mask_ious >= 0.50)),
            "mask_recall50": float(np.mean(mask_ious >= 0.50)),
            "mask_recall75_count": int(np.sum(mask_ious >= 0.75)),
            "mask_recall75": float(np.mean(mask_ious >= 0.75)),
            "mean_mask_iou": float(mask_ious.mean()),
            "box_recall50_count": int(np.sum(box_ious >= 0.50)),
            "box_recall50": float(np.mean(box_ious >= 0.50)),
            "median_predicted_to_target_area_ratio": float(np.median(area_ratios)),
        }
    return result


def _warm_mean_seconds(image_rows: list[dict[str, object]]) -> float | None:
    """Return warm per-image audit time without treating it as API latency."""
    if len(image_rows) < 2:
        return None
    return float(
        np.mean([cast(float, row["elapsed_seconds"]) for row in image_rows[1:]])
    )


if __name__ == "__main__":
    main()
