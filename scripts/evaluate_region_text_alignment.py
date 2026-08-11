"""Evaluate one learned text projection on an independent referring split."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_CHECKPOINT = (
    "outputs/localization/dinov2_bge_alignment_smoke/alignment_head_smoke.pt"
)


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one bounded, independent-split retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate BGE-M3-to-DINOv2 region retrieval by source image."
    )
    parser.add_argument(
        "--split", choices=("train", "validation"), default="validation"
    )
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument("--limit", type=int, default=None)
    limit_group.add_argument("--image-limit", type=int, default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--spatial-weight", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/localization/dinov2_bge_alignment_validation_smoke",
    )
    return parser.parse_args()


def main() -> None:
    """Load frozen features, rank same-image candidates, and save audit records."""
    args = parse_args()
    sample_limit = args.limit if args.image_limit is None else None
    if sample_limit is None and args.image_limit is None:
        sample_limit = 8
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("--limit must be at least one")
    if args.image_limit is not None and args.image_limit < 1:
        raise ValueError("--image-limit must be at least one")
    if not 0.0 <= args.spatial_weight <= 1.0:
        raise ValueError("--spatial-weight must be between zero and one")
    add_src_to_python_path()
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for alignment evaluation.") from error

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
        build_spatial_score_adjustments,
        evaluate_image_candidate_retrieval,
        extract_unique_region_features,
        load_text_projection_checkpoint,
    )

    project_settings = load_settings()
    using_default_index = args.index is None
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
        max_samples=sample_limit,
        max_images=args.image_limit,
    )
    items = [dataset[index] for index in range(len(dataset))]
    if not items:
        raise ValueError("Alignment evaluation loaded no samples.")

    extraction_started = time.perf_counter()
    bge_encoder = BgeM3TextEncoder(load_bge_m3_text_settings())
    text_embeddings = bge_encoder.encode([item.sample.query for item in items])
    bge_encoder.synchronize()
    dinov2_encoder = DinoV2RegionEncoder(load_dinov2_region_settings())
    region_features_by_id = extract_unique_region_features(items, dinov2_encoder)
    dinov2_encoder.synchronize()
    feature_extraction_seconds = time.perf_counter() - extraction_started

    device = "cuda" if torch.cuda.is_available() else "cpu"
    projection, alignment_settings = load_text_projection_checkpoint(
        args.checkpoint,
        device=device,
    )
    if text_embeddings.shape[1] != alignment_settings.text_dimension:
        raise ValueError("Text feature dimension does not match the checkpoint.")
    text_tensor = torch.from_numpy(text_embeddings).to(device=device)
    with torch.inference_mode():
        projected_text = projection(text_tensor).float().cpu().numpy()

    region_annotation_ids = sorted(region_features_by_id)
    region_features = np.stack(
        [region_features_by_id[value] for value in region_annotation_ids]
    )
    region_image_by_id: dict[int, int] = {}
    region_box_by_id: dict[int, np.ndarray] = {}
    image_sizes: dict[int, tuple[int, int]] = {}
    for item in items:
        image_id = item.sample.source_image_id
        image_sizes[image_id] = item.image_rgb.shape[:2]
        for annotation_id, box in zip(
            item.source_annotation_ids,
            item.target_boxes,
        ):
            image_id = item.sample.source_image_id
            previous = region_image_by_id.setdefault(annotation_id, image_id)
            if previous != image_id:
                raise ValueError(
                    f"Annotation {annotation_id} belongs to multiple source images."
                )
            previous_box = region_box_by_id.setdefault(annotation_id, box)
            if not np.allclose(previous_box, box, atol=1e-6):
                raise ValueError(
                    f"Annotation {annotation_id} has inconsistent target boxes."
                )

    query_texts = [item.sample.query for item in items]
    region_image_ids = [region_image_by_id[value] for value in region_annotation_ids]
    spatial_adjustments, spatial_modifiers = build_spatial_score_adjustments(
        queries=query_texts,
        query_image_ids=[item.sample.source_image_id for item in items],
        region_image_ids=region_image_ids,
        region_boxes_xyxy=np.stack(
            [region_box_by_id[value] for value in region_annotation_ids]
        ),
        image_sizes=image_sizes,
        weight=args.spatial_weight,
    )

    summary, cases = evaluate_image_candidate_retrieval(
        query_ids=[item.sample.id for item in items],
        projected_text_features=projected_text,
        query_image_ids=[item.sample.source_image_id for item in items],
        query_target_ids=[item.source_annotation_ids for item in items],
        query_dimensions=[tuple(item.sample.dimensions) for item in items],
        query_languages=[item.sample.language for item in items],
        region_annotation_ids=region_annotation_ids,
        region_image_ids=region_image_ids,
        region_features=region_features,
        score_adjustments=spatial_adjustments,
    )
    for case, item, modifier in zip(cases, items, spatial_modifiers):
        case["query"] = item.sample.query
        case["target_label"] = item.sample.target_label
        case["template_id"] = item.sample.template_id
        case["spatial_modifier"] = modifier
    spatial_summary = summary["by_dimension"].get("spatial", {})
    annotated_part_coverage = args.image_limit is not None and using_default_index
    summary.update(
        {
            "split": args.split,
            "selected_image_count": len(
                {item.sample.source_image_id for item in items}
            ),
            "unique_region_count": len(region_annotation_ids),
            "feature_extraction_seconds": feature_extraction_seconds,
            "spatial_rerank_weight": args.spatial_weight,
            "spatial_modifier_query_count": sum(
                modifier is not None for modifier in spatial_modifiers
            ),
            "spatial_competitive_query_count": spatial_summary.get(
                "competitive_query_count", 0
            ),
            "spatial_competitive_top1_accuracy": spatial_summary.get(
                "competitive_top1_accuracy"
            ),
            "spatial_competitive_exact_set_at_target_count_rate": (
                spatial_summary.get("competitive_exact_set_at_target_count_rate")
            ),
            "checkpoint_path": str(resolve_project_path(args.checkpoint)),
            "selection_scope": (
                "complete_image_prefix"
                if args.image_limit is not None
                else "query_prefix"
            ),
            "candidate_region_scope": (
                "all_fashionpedia_part_masks_per_selected_image"
                if annotated_part_coverage
                else "selected_query_target_union_per_image"
            ),
            "fashionpedia_annotated_part_candidate_coverage": annotated_part_coverage,
            "full_image_candidate_coverage": False,
            "mask_localization_evaluated": False,
            "prd_accuracy_92_passed": None,
            "prd_localization_30ms_passed": None,
        }
    )
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    keys = (
        "split",
        "query_count",
        "selected_image_count",
        "unique_region_count",
        "top1_correct_count",
        "top1_accuracy",
        "exact_set_correct_count",
        "exact_set_at_target_count_rate",
        "mean_reciprocal_rank",
        "competitive_query_count",
        "competitive_top1_correct_count",
        "competitive_top1_accuracy",
        "competitive_exact_set_correct_count",
        "competitive_exact_set_at_target_count_rate",
        "spatial_rerank_weight",
        "spatial_modifier_query_count",
        "spatial_competitive_query_count",
        "spatial_competitive_top1_accuracy",
        "spatial_competitive_exact_set_at_target_count_rate",
        "selection_scope",
        "candidate_region_scope",
        "fashionpedia_annotated_part_candidate_coverage",
        "full_image_candidate_coverage",
        "mask_localization_evaluated",
        "prd_accuracy_92_passed",
        "prd_localization_30ms_passed",
    )
    for key in keys:
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
