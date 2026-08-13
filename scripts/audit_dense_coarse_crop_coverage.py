"""Audit category-free coarse query crops before local re-encoding."""

import argparse
import sys
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
    """Parse one GT-dependent coarse-crop coverage audit.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Audit full-query peak crops before local re-encoding."
    )
    parser.add_argument(
        "--split", choices=("train", "validation"), default="validation"
    )
    parser.add_argument("--image-limit", type=int, default=50)
    parser.add_argument("--image-offset", type=int, default=52)
    parser.add_argument("--index", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument(
        "--dinov2-config",
        default="configs/localization_dinov2_region_728.yaml",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--crop-fractions",
        type=float,
        nargs="+",
        default=(0.20, 0.30, 0.40),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/localization/dense_coarse_crop_audit",
    )
    return parser.parse_args()


def main() -> None:
    """Score query-only peak crops against retained GT Masks.

    Raises:
        ValueError: If selection, crop settings, or model dimensions are invalid.
        RuntimeError: If PyTorch is unavailable.
    """
    args = parse_args()
    if args.image_limit < 1 or args.image_offset < 0:
        raise ValueError("Image limit must be positive and offset non-negative.")
    crop_fractions = tuple(float(value) for value in args.crop_fractions)
    if (
        not crop_fractions
        or tuple(sorted(set(crop_fractions))) != crop_fractions
        or any(not 0.0 < value <= 1.0 for value in crop_fractions)
    ):
        raise ValueError("Crop fractions must be unique ascending values in (0, 1].")
    add_src_to_python_path()
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for crop coverage audit.") from error

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
        crop_target_coverage,
        select_query_peak_crops,
    )
    from fashion_semantic_parser.service.dense_patch_alignment import (
        load_dense_patch_alignment_checkpoint,
    )
    from fashion_semantic_parser.service.dense_patch_inference import (
        predict_patch_outputs,
    )
    from fashion_semantic_parser.service.dense_patch_metrics import write_dense_json
    from fashion_semantic_parser.service.dinov2_region_encoder import (
        DinoV2RegionEncoder,
        load_dinov2_region_settings,
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
        raise ValueError("Coarse crop audit loaded no queries.")
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
    groups: dict[int, list[int]] = defaultdict(list)
    for item_index, item in enumerate(items):
        groups[item.sample.source_image_id].append(item_index)
    cases: list[dict[str, object]] = []
    for image_number, (image_id, item_indices) in enumerate(groups.items(), start=1):
        dense = image_encoder.encode_dense(items[item_indices[0]].image_rgb)
        probabilities, _ = predict_patch_outputs(
            checkpoint,
            dense.features,
            projected_text[item_indices],
            device,
        )
        for local_index, item_index in enumerate(item_indices):
            item = items[item_index]
            target_mask = np.asarray(item.target_masks.any(axis=0), dtype=bool)
            audits: dict[str, object] = {}
            for fraction in crop_fractions:
                top_three = select_query_peak_crops(
                    probabilities[local_index],
                    dense.geometry,
                    crop_fraction=fraction,
                    max_crops=3,
                )
                top_one_coverage, top_one_area = crop_target_coverage(
                    target_mask,
                    top_three[:1],
                )
                top_three_coverage, top_three_area = crop_target_coverage(
                    target_mask,
                    top_three,
                )
                audits[f"{fraction:.2f}"] = {
                    "top1_target_coverage": top_one_coverage,
                    "top1_image_area_fraction": top_one_area,
                    "top3_target_coverage": top_three_coverage,
                    "top3_image_area_fraction": top_three_area,
                    "crop_count": len(top_three),
                }
            cases.append(
                {
                    "query_id": item.sample.id,
                    "query": item.sample.query,
                    "language": item.sample.language,
                    "dimensions": list(item.sample.dimensions),
                    "target_label": item.sample.target_label,
                    "source_image_id": image_id,
                    "target_area": int(target_mask.sum()),
                    "crop_audits": audits,
                }
            )
        print(
            f"[{image_number}/{len(groups)}] image={image_id} "
            + f"queries={len(item_indices)}"
        )
    summary = _summarize(cases, crop_fractions)
    summary.update(
        {
            "split": args.split,
            "image_offset": args.image_offset,
            "selected_image_count": len(groups),
            "checkpoint_path": str(resolve_project_path(args.checkpoint)),
            "checkpoint_training_input_size": checkpoint.training_input_size,
            "dinov2_input_size": image_encoder.settings.input_size,
            "crop_generation_uses_ground_truth": False,
            "coverage_evaluation_uses_ground_truth": True,
            "uses_fixed_part_categories": False,
            "full_language_query_used": True,
            "mask_localization_evaluated": False,
            "prd_accuracy_92_passed": None,
            "prd_localization_30ms_passed": None,
        }
    )
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_dense_json(output_dir / "metrics.json", summary)
    write_dense_json(output_dir / "cases.json", cases)
    print(f"query_count: {summary['query_count']}")
    for fraction in crop_fractions:
        by_fraction = cast(dict[str, object], summary["by_crop_fraction"])
        row = cast(dict[str, object], by_fraction[f"{fraction:.2f}"])
        print(
            f"crop_fraction={fraction:.2f} "
            + f"top1_C90={cast(float, row['top1_target_coverage90']):.6f} "
            + f"top3_C90={cast(float, row['top3_target_coverage90']):.6f} "
            + "top3_area="
            + f"{cast(float, row['mean_top3_image_area_fraction']):.6f}"
        )


def _summarize(
    cases: list[dict[str, object]],
    crop_fractions: tuple[float, ...],
) -> dict[str, object]:
    """Aggregate crop coverage with explicit denominators."""
    by_fraction: dict[str, object] = {}
    for fraction in crop_fractions:
        key = f"{fraction:.2f}"
        rows = [
            cast(dict[str, object], cast(dict[str, object], case["crop_audits"])[key])
            for case in cases
        ]
        top_one = np.asarray([cast(float, row["top1_target_coverage"]) for row in rows])
        top_three = np.asarray(
            [cast(float, row["top3_target_coverage"]) for row in rows]
        )
        top_three_area = np.asarray(
            [cast(float, row["top3_image_area_fraction"]) for row in rows]
        )
        by_fraction[key] = {
            "query_count": len(rows),
            "top1_target_coverage50_count": int(np.sum(top_one >= 0.50)),
            "top1_target_coverage50": float(np.mean(top_one >= 0.50)),
            "top1_target_coverage90_count": int(np.sum(top_one >= 0.90)),
            "top1_target_coverage90": float(np.mean(top_one >= 0.90)),
            "top3_target_coverage50_count": int(np.sum(top_three >= 0.50)),
            "top3_target_coverage50": float(np.mean(top_three >= 0.50)),
            "top3_target_coverage90_count": int(np.sum(top_three >= 0.90)),
            "top3_target_coverage90": float(np.mean(top_three >= 0.90)),
            "mean_top3_image_area_fraction": float(top_three_area.mean()),
        }
    return {
        "query_count": len(cases),
        "by_crop_fraction": by_fraction,
    }


if __name__ == "__main__":
    main()
