"""Evaluate full-query DINOv2 dense similarity Masks without oracle candidates."""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

DEFAULT_CHECKPOINT = (
    "outputs/localization/dinov2_bge_alignment_train_images300_global/"
    + "alignment_head_smoke.pt"
)


@dataclass(frozen=True)
class DenseRunMetadata:
    """Immutable configuration and startup timing for one dense smoke run."""

    split: str
    image_offset: int
    checkpoint: str
    text_seconds: float
    projection_seconds: float


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one image-complete dense localization smoke.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate full-image DINOv2-to-text dense similarity Masks."
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument("--image-limit", type=int, default=2)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--index", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--score-quantiles",
        type=_parse_quantiles,
        default=(0.90, 0.95, 0.98, 0.99, 0.995),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/localization/dinov2_dense_localization_smoke",
    )
    return parser.parse_args()


def main() -> None:
    """Score dense patches against full language queries and retain every miss.

    Raises:
        ValueError: If selection arguments or loaded data are invalid.
    """
    args = parse_args()
    if args.image_limit < 1:
        raise ValueError("--image-limit must be at least one")
    if args.image_offset < 0:
        raise ValueError("--image-offset cannot be negative")
    add_src_to_python_path()
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for dense localization.") from error

    from fashion_semantic_parser.common.paths import PROJECT_ROOT, resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.localization.referring_dataset import (
        FashionpediaReferringDataset,
    )
    from fashion_semantic_parser.service.bge_m3_text_encoder import (
        BgeM3TextEncoder,
        load_bge_m3_text_settings,
    )
    from fashion_semantic_parser.service.dense_region_localization import (
        binary_mask_iou,
        box_iou,
        dense_similarity_scores,
        mask_box,
        quantile_mask_candidates,
    )
    from fashion_semantic_parser.service.dinov2_region_encoder import (
        DinoV2RegionEncoder,
        load_dinov2_region_settings,
        patch_scores_to_image,
    )
    from fashion_semantic_parser.service.region_text_alignment import (
        load_text_projection_checkpoint,
    )

    project_settings = load_settings()
    index = args.index or (
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
        index_path=resolve_project_path(index),
        annotation_path=resolve_project_path(annotations),
        project_root=PROJECT_ROOT,
        max_images=args.image_limit,
        image_offset=args.image_offset,
    )
    items = [dataset[index] for index in range(len(dataset))]
    if not items:
        raise ValueError("Dense localization smoke loaded no queries.")

    bge_encoder = BgeM3TextEncoder(load_bge_m3_text_settings())
    text_started = time.perf_counter()
    text_embeddings = bge_encoder.encode([item.sample.query for item in items])
    bge_encoder.synchronize()
    text_seconds = time.perf_counter() - text_started

    device = "cuda" if torch.cuda.is_available() else "cpu"
    projection, alignment_settings = load_text_projection_checkpoint(
        args.checkpoint,
        device=device,
    )
    if text_embeddings.shape[1] != alignment_settings.text_dimension:
        raise ValueError("Text feature dimension does not match the checkpoint.")
    projection_started = time.perf_counter()
    with torch.inference_mode():
        text_tensor = torch.from_numpy(text_embeddings).to(device=device)
        projected = projection(text_tensor)
        projected = torch.nn.functional.normalize(projected.float(), dim=1)
        projected_text = np.asarray(projected.cpu().numpy(), dtype=np.float32)
    projection_seconds = time.perf_counter() - projection_started

    groups: dict[int, list[int]] = defaultdict(list)
    for item_index, item in enumerate(items):
        groups[item.sample.source_image_id].append(item_index)
    dinov2_encoder = DinoV2RegionEncoder(load_dinov2_region_settings())
    cases: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    for image_number, (image_id, item_indices) in enumerate(groups.items(), start=1):
        image_rgb = items[item_indices[0]].image_rgb
        started = time.perf_counter()
        dense_features = dinov2_encoder.encode_dense(image_rgb)
        dinov2_encoder.synchronize()
        encode_seconds = time.perf_counter() - started
        scoring_started = time.perf_counter()
        score_grids = dense_similarity_scores(
            dense_features.features,
            projected_text[item_indices],
        )
        for local_index, item_index in enumerate(item_indices):
            item = items[item_index]
            image_scores = patch_scores_to_image(
                score_grids[local_index],
                dense_features.geometry,
            )
            target_mask = np.asarray(item.target_masks.any(axis=0), dtype=bool)
            target_box = mask_box(target_mask)
            candidates = quantile_mask_candidates(
                image_scores,
                args.score_quantiles,
            )
            cases.append(
                {
                    "query_id": item.sample.id,
                    "query": item.sample.query,
                    "language": item.sample.language,
                    "dimensions": list(item.sample.dimensions),
                    "target_label": item.sample.target_label,
                    "source_image_id": image_id,
                    "target_annotation_ids": list(item.source_annotation_ids),
                    "target_count": len(item.source_annotation_ids),
                    "mask_iou_by_quantile": {
                        _quantile_key(candidate.quantile): binary_mask_iou(
                            target_mask,
                            candidate.mask,
                        )
                        for candidate in candidates
                    },
                    "box_iou_by_quantile": {
                        _quantile_key(candidate.quantile): box_iou(
                            target_box,
                            candidate.box,
                        )
                        for candidate in candidates
                    },
                    "threshold_by_quantile": {
                        _quantile_key(candidate.quantile): candidate.threshold
                        for candidate in candidates
                    },
                    "predicted_area_by_quantile": {
                        _quantile_key(candidate.quantile): int(candidate.mask.sum())
                        for candidate in candidates
                    },
                }
            )
        scoring_seconds = time.perf_counter() - scoring_started
        image_rows.append(
            {
                "source_image_id": image_id,
                "query_count": len(item_indices),
                "dinov2_encode_seconds": encode_seconds,
                "dense_scoring_seconds": scoring_seconds,
            }
        )
        print(
            f"[{image_number}/{len(groups)}] image={image_id} "
            + f"queries={len(item_indices)} encode={encode_seconds:.3f}s "
            + f"score={scoring_seconds:.3f}s"
        )

    metadata = DenseRunMetadata(
        split=args.split,
        image_offset=args.image_offset,
        checkpoint=str(resolve_project_path(args.checkpoint)),
        text_seconds=text_seconds,
        projection_seconds=projection_seconds,
    )
    summary = _summarize(
        cases,
        image_rows,
        args.score_quantiles,
        metadata,
    )
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", summary)
    _write_json(output_dir / "cases.json", cases)
    _write_json(output_dir / "images.json", image_rows)
    print(f"query_count: {summary['query_count']}")
    for quantile, metrics in cast(
        dict[str, dict[str, object]],
        summary["by_score_quantile"],
    ).items():
        print(
            f"quantile={quantile} "
            + f"mask_R50={metrics['mask_recall50']} "
            + f"mask_R75={metrics['mask_recall75']} "
            + f"mean_mask_iou={metrics['mean_mask_iou']} "
            + f"box_R50={metrics['box_recall50']}"
        )


def _parse_quantiles(value: str) -> tuple[float, ...]:
    """Parse unique ascending score quantiles."""
    try:
        quantiles = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Score quantiles must be comma-separated numbers."
        ) from error
    if (
        not quantiles
        or tuple(sorted(set(quantiles))) != quantiles
        or any(not 0.0 < quantile < 1.0 for quantile in quantiles)
    ):
        raise argparse.ArgumentTypeError(
            "Score quantiles must be unique, ascending, and in (0, 1)."
        )
    return quantiles


def _quantile_key(value: float) -> str:
    """Return one stable JSON key for a score quantile."""
    return f"{value:.3f}"


def _summarize(
    cases: list[dict[str, object]],
    image_rows: list[dict[str, object]],
    quantiles: tuple[float, ...],
    metadata: DenseRunMetadata,
) -> dict[str, object]:
    """Aggregate query-level dense Mask and Box metrics with denominators."""
    by_quantile = {
        _quantile_key(quantile): _quantile_metrics(cases, quantile)
        for quantile in quantiles
    }
    return {
        "split": metadata.split,
        "selected_image_count": len(image_rows),
        "image_offset": metadata.image_offset,
        "query_count": len(cases),
        "score_quantiles": quantiles,
        "by_score_quantile": by_quantile,
        "text_encoding_seconds": metadata.text_seconds,
        "text_projection_seconds": metadata.projection_seconds,
        "dinov2_encode_seconds": sum(
            cast(float, row["dinov2_encode_seconds"]) for row in image_rows
        ),
        "dense_scoring_seconds": sum(
            cast(float, row["dense_scoring_seconds"]) for row in image_rows
        ),
        "checkpoint_path": metadata.checkpoint,
        "candidate_region_scope": "full_image_dinov2_dense_patch_similarity",
        "full_image_candidate_coverage": True,
        "uses_oracle_candidates": False,
        "full_language_query_used": True,
        "mask_localization_evaluated": True,
        "independent_manual_test_set": False,
        "selected_score_quantile": None,
        "prd_accuracy_92_passed": None,
        "prd_localization_30ms_passed": None,
    }


def _quantile_metrics(
    cases: list[dict[str, object]],
    quantile: float,
) -> dict[str, object]:
    """Return numerator-aware Mask and Box metrics at one quantile."""
    key = _quantile_key(quantile)
    mask_ious = np.asarray(
        [
            float(cast(dict[str, float], case["mask_iou_by_quantile"])[key])
            for case in cases
        ],
        dtype=float,
    )
    box_ious = np.asarray(
        [
            float(cast(dict[str, float], case["box_iou_by_quantile"])[key])
            for case in cases
        ],
        dtype=float,
    )
    return {
        "query_count": len(cases),
        "mask_recall50_count": int(np.sum(mask_ious >= 0.50)),
        "mask_recall50": float(np.mean(mask_ious >= 0.50)),
        "mask_recall75_count": int(np.sum(mask_ious >= 0.75)),
        "mask_recall75": float(np.mean(mask_ious >= 0.75)),
        "mean_mask_iou": float(mask_ious.mean()),
        "box_recall50_count": int(np.sum(box_ious >= 0.50)),
        "box_recall50": float(np.mean(box_ious >= 0.50)),
        "mean_box_iou": float(box_ious.mean()),
    }


def _write_json(path: Path, value: object) -> None:
    """Write one deterministic UTF-8 JSON artifact."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
