"""Evaluate supervised full-query DINOv2 patch Masks on complete images."""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np

DEFAULT_CHECKPOINT = (
    "outputs/localization/dinov2_dense_patch_alignment_train_images100/"
    + "dense_patch_alignment.pt"
)


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one image-complete dense patch localization evaluation.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated DINOv2 query-to-patch Masks."
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
        "--output-dir",
        default="outputs/localization/dinov2_dense_patch_localization",
    )
    return parser.parse_args()


def main() -> None:
    """Predict full-image Masks with one frozen learned probability threshold.

    Raises:
        ValueError: If selection or model feature dimensions are invalid.
        RuntimeError: If PyTorch is unavailable.
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
        raise RuntimeError("PyTorch is required for dense patch evaluation.") from error

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
    )
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
        raise ValueError("Dense patch evaluation loaded no queries.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = load_dense_patch_alignment_checkpoint(
        args.checkpoint,
        device=device,
    )
    text_encoder = BgeM3TextEncoder(load_bge_m3_text_settings())
    text_started = time.perf_counter()
    text_embeddings = text_encoder.encode([item.sample.query for item in items])
    text_encoder.synchronize()
    text_seconds = time.perf_counter() - text_started
    if text_embeddings.shape[1] != checkpoint.alignment_settings.text_dimension:
        raise ValueError("Text dimension does not match the dense checkpoint.")
    projection_started = time.perf_counter()
    with torch.inference_mode():
        text_tensor = torch.from_numpy(text_embeddings).to(device=device)
        projected = checkpoint.projection(text_tensor)
        projected = torch.nn.functional.normalize(projected.float(), dim=1)
        projected_text = np.asarray(projected.cpu().numpy(), dtype=np.float32)
    projection_seconds = time.perf_counter() - projection_started

    groups: dict[int, list[int]] = defaultdict(list)
    for item_index, item in enumerate(items):
        groups[item.sample.source_image_id].append(item_index)
    image_encoder = DinoV2RegionEncoder(load_dinov2_region_settings())
    cases: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    threshold = checkpoint.dense_settings.probability_threshold
    for image_number, (image_id, item_indices) in enumerate(groups.items(), start=1):
        image_started = time.perf_counter()
        dense = image_encoder.encode_dense(items[item_indices[0]].image_rgb)
        image_encoder.synchronize()
        encode_seconds = time.perf_counter() - image_started
        scoring_started = time.perf_counter()
        probabilities, predicted_area_fractions = _predict_patch_outputs(
            checkpoint,
            dense.features,
            projected_text[item_indices],
            device,
        )
        selected_patch_masks = None
        if predicted_area_fractions is not None:
            from fashion_semantic_parser.service.dense_patch_area import (
                topk_patch_masks,
            )

            selected_patch_masks = topk_patch_masks(
                probabilities.reshape(len(item_indices), -1),
                predicted_area_fractions,
            ).reshape(probabilities.shape)
        for local_index, item_index in enumerate(item_indices):
            item = items[item_index]
            if selected_patch_masks is None:
                patch_selection = probabilities[local_index] >= threshold
            else:
                patch_selection = selected_patch_masks[local_index]
            predicted_mask = np.asarray(
                patch_scores_to_image(
                    np.asarray(patch_selection, dtype=np.float32),
                    dense.geometry,
                )
                >= 0.5,
                dtype=bool,
            )
            target_mask = np.asarray(item.target_masks.any(axis=0), dtype=bool)
            mask_iou = binary_mask_iou(target_mask, predicted_mask)
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
                    "mask_iou": mask_iou,
                    "box_iou": box_iou(
                        mask_box(target_mask),
                        mask_box(predicted_mask),
                    ),
                    "mask_recall50_passed": mask_iou >= 0.50,
                    "target_area": int(target_mask.sum()),
                    "predicted_area": int(predicted_mask.sum()),
                    "predicted_patch_count": int(patch_selection.sum()),
                    "predicted_area_fraction": (
                        None
                        if predicted_area_fractions is None
                        else float(predicted_area_fractions[local_index])
                    ),
                }
            )
        scoring_seconds = time.perf_counter() - scoring_started
        total_seconds = time.perf_counter() - image_started
        image_rows.append(
            {
                "source_image_id": image_id,
                "query_count": len(item_indices),
                "dinov2_encode_seconds": encode_seconds,
                "dense_scoring_seconds": scoring_seconds,
                "total_image_seconds": total_seconds,
            }
        )
        print(
            f"[{image_number}/{len(groups)}] image={image_id} "
            + f"queries={len(item_indices)} encode={encode_seconds:.3f}s "
            + f"score={scoring_seconds:.3f}s"
        )

    summary = _summarize(cases, image_rows)
    summary.update(
        {
            "split": args.split,
            "selected_image_count": len(image_rows),
            "image_offset": args.image_offset,
            "selection_scope": "complete_image_prefix",
            "probability_threshold": threshold,
            "mask_selection_mode": (
                "query_area_topk"
                if checkpoint.area_predictor is not None
                else "global_probability_threshold"
            ),
            "model_type": checkpoint.model_type,
            "logit_scale": checkpoint.logit_scale,
            "logit_bias": checkpoint.logit_bias,
            "text_encoding_seconds": text_seconds,
            "text_projection_seconds": projection_seconds,
            "first_image_including_dinov2_load_seconds": image_rows[0][
                "total_image_seconds"
            ],
            "warm_image_count": max(0, len(image_rows) - 1),
            "warm_mean_image_seconds": _warm_mean(image_rows),
            "checkpoint_path": str(resolve_project_path(args.checkpoint)),
            "candidate_region_scope": "full_image_supervised_dinov2_patch_similarity",
            "full_image_candidate_coverage": True,
            "uses_oracle_candidates": False,
            "full_language_query_used": True,
            "mask_localization_evaluated": True,
            "independent_manual_test_set": False,
            "prd_accuracy_92_passed": None,
            "prd_localization_30ms_passed": None,
        }
    )
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", summary)
    _write_json(output_dir / "cases.json", cases)
    _write_json(output_dir / "images.json", image_rows)
    for key in (
        "query_count",
        "mask_recall50_count",
        "mask_recall50",
        "mask_recall75_count",
        "mask_recall75",
        "mean_mask_iou",
        "box_recall50",
        "warm_mean_image_seconds",
    ):
        print(f"{key}: {summary[key]}")


def _summarize(
    cases: list[dict[str, object]],
    image_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate overall and diagnostic grouped Mask/Box metrics."""
    result = _score_cases(cases)
    dimensions = sorted(
        {
            dimension
            for case in cases
            for dimension in cast(list[str], case["dimensions"])
        }
    )
    result["by_dimension"] = {
        dimension: _score_cases(
            [case for case in cases if dimension in cast(list[str], case["dimensions"])]
        )
        for dimension in dimensions
    }
    for field in ("language", "target_label"):
        values = sorted({cast(str, case[field]) for case in cases})
        result[f"by_{field}"] = {
            value: _score_cases(
                [case for case in cases if cast(str, case[field]) == value]
            )
            for value in values
        }
    result["image_count"] = len(image_rows)
    return result


def _predict_patch_outputs(
    checkpoint: object,
    patch_features: np.ndarray,
    projected_text: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Dispatch frozen patch probabilities and optional predicted target areas."""
    from fashion_semantic_parser.service.dense_patch_alignment import (
        DensePatchAlignmentCheckpoint,
    )

    typed_checkpoint = cast(DensePatchAlignmentCheckpoint, checkpoint)
    if typed_checkpoint.model_type == "cosine_calibration":
        from fashion_semantic_parser.service.dense_region_localization import (
            calibrated_dense_probabilities,
            dense_similarity_scores,
        )

        similarities = dense_similarity_scores(patch_features, projected_text)
        calibrated: np.ndarray = calibrated_dense_probabilities(
            similarities,
            logit_scale=typed_checkpoint.logit_scale,
            logit_bias=typed_checkpoint.logit_bias,
        )
        return calibrated, None
    if (
        typed_checkpoint.model_type
        not in {"multiscale_decoder", "multiscale_area_decoder"}
        or typed_checkpoint.decoder is None
    ):
        raise ValueError("Dense checkpoint model type and decoder are inconsistent.")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for multiscale evaluation.") from error
    from fashion_semantic_parser.service.dense_patch_decoder import (
        multiscale_patch_decoder_logits,
    )

    height, width, feature_dimension = patch_features.shape
    query_count = len(projected_text)
    flattened = np.asarray(
        patch_features.reshape(1, height * width, feature_dimension),
        dtype=np.float32,
    )
    patch_tensor = (
        torch.from_numpy(flattened)
        .to(device=device)
        .expand(
            query_count,
            -1,
            -1,
        )
    )
    text_tensor = torch.from_numpy(projected_text).to(device=device)
    log_scale = torch.tensor(
        math.log(typed_checkpoint.logit_scale),
        dtype=torch.float32,
        device=device,
    )
    logit_bias = torch.tensor(
        typed_checkpoint.logit_bias,
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        logits = multiscale_patch_decoder_logits(
            typed_checkpoint.decoder,
            patch_tensor,
            text_tensor,
            (
                log_scale,
                logit_bias,
                typed_checkpoint.dense_settings.max_logit_scale,
            ),
        )
        probabilities = torch.sigmoid(logits).reshape(query_count, height, width)
        predicted_area_fractions = None
        if typed_checkpoint.model_type == "multiscale_area_decoder":
            if typed_checkpoint.area_predictor is None:
                raise ValueError("Area checkpoint is missing its area predictor.")
            from fashion_semantic_parser.service.dense_patch_area import (
                query_area_logits,
            )

            area_logits = query_area_logits(
                typed_checkpoint.area_predictor,
                patch_tensor,
                text_tensor,
            )
            predicted_area_fractions = np.asarray(
                torch.sigmoid(area_logits).cpu().numpy(),
                dtype=np.float32,
            )
        elif typed_checkpoint.area_predictor is not None:
            raise ValueError("Non-area checkpoint unexpectedly has an area predictor.")
    return (
        np.asarray(probabilities.cpu().numpy(), dtype=np.float32),
        predicted_area_fractions,
    )


def _score_cases(cases: list[dict[str, object]]) -> dict[str, object]:
    """Score all retained cases with explicit numerators and denominators."""
    mask_ious = np.asarray(
        [cast(float, case["mask_iou"]) for case in cases],
        dtype=float,
    )
    box_ious = np.asarray(
        [cast(float, case["box_iou"]) for case in cases],
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


def _warm_mean(image_rows: list[dict[str, object]]) -> float | None:
    """Return mean image time after excluding the first model-load image."""
    if len(image_rows) < 2:
        return None
    return float(
        np.mean([cast(float, row["total_image_seconds"]) for row in image_rows[1:]])
    )


def _write_json(path: Path, value: object) -> None:
    """Write one deterministic UTF-8 JSON artifact."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
