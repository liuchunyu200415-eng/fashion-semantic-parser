"""Measure SAM-HQ Mask quality when prompted by oracle Fashionpedia boxes."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, TypedDict

import numpy as np


class _TargetRow(TypedDict):
    """One unique Fashionpedia target used by the oracle diagnostic."""

    annotation_id: int
    label: str
    mask: np.ndarray
    box: tuple[float, float, float, float]


class _ImageGroup(TypedDict):
    """One decoded image and its unique annotation targets."""

    image_rgb: np.ndarray | None
    targets: dict[int, _TargetRow]


class _CaseRow(TypedDict):
    """One persisted target-level oracle refinement result."""

    source_image_id: int
    source_annotation_id: int
    target_label: str
    target_area_pixels: int
    target_area_ratio: float
    prompt_box: tuple[float, float, float, float]
    mask_box: tuple[float, float, float, float] | None
    mask_quality: float
    mask_iou: float
    box_expansion_ratio: float
    candidate_count: int
    score_selected_candidate_index: int
    oracle_best_candidate_index: int
    oracle_best_mask_iou: float


class _ImageRow(TypedDict):
    """One persisted image-level runtime result."""

    source_image_id: int
    target_count: int
    prompt_count: int
    elapsed_seconds: float


class _DatasetProtocol(Protocol):
    """Minimal map-style dataset contract used by the diagnostic."""

    def __len__(self) -> int:
        """Return the selected query count."""
        ...

    def __getitem__(self, index: int) -> Any:
        """Return one loaded referring item."""
        ...


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one image-complete oracle-Box refinement smoke.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Measure SAM-HQ refinement with exact Fashionpedia GT boxes."
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
    parser.add_argument(
        "--box-expansion-ratios",
        type=_parse_expansion_ratios,
        default=(0.0,),
        help="Comma-separated per-side Box expansion ratios; must include zero.",
    )
    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="Retain ambiguity-aware SAM-HQ Mask candidates for oracle analysis.",
    )
    parser.add_argument(
        "--config",
        default="configs/localization_sam_hq_proposals.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/localization/sam_hq_box_prompt_recall_smoke",
    )
    return parser.parse_args()


def main() -> None:
    """Run oracle-Box refinement while retaining every GT target.

    Raises:
        ValueError: If selection arguments or loaded target data are invalid.
    """
    args = parse_args()
    if args.image_limit < 1:
        raise ValueError("--image-limit must be at least one")
    if args.image_offset < 0:
        raise ValueError("--image-offset cannot be negative")
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import PROJECT_ROOT, resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.localization.referring_dataset import (
        FashionpediaReferringDataset,
    )
    from fashion_semantic_parser.service.sam_hq_proposals import (
        load_sam_hq_proposal_settings,
    )
    from fashion_semantic_parser.service.sam_hq_refinement import (
        SAMHQBoxPromptRefiner,
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
    groups = _load_unique_targets(dataset)
    if not groups:
        raise ValueError("SAM-HQ Box prompt smoke loaded no images.")

    refiner = SAMHQBoxPromptRefiner(load_sam_hq_proposal_settings(args.config))
    cases: list[_CaseRow] = []
    image_rows: list[_ImageRow] = []
    latencies: list[float] = []
    for image_number, (image_id, group) in enumerate(groups.items(), start=1):
        targets = [group["targets"][key] for key in sorted(group["targets"])]
        image_rgb = group["image_rgb"]
        if image_rgb is None:
            raise ValueError(f"Image {image_id} has no decoded pixels.")
        prompt_rows = [
            (
                target,
                ratio,
                _expand_box(
                    target["box"],
                    ratio,
                    image_width=int(image_rgb.shape[1]),
                    image_height=int(image_rgb.shape[0]),
                ),
            )
            for ratio in args.box_expansion_ratios
            for target in targets
        ]
        boxes = [row[2] for row in prompt_rows]
        started = time.perf_counter()
        candidate_groups = refiner.refine_candidates(
            image_rgb,
            boxes,
            multimask_output=args.multimask_output,
        )
        refiner.synchronize()
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        if len(candidate_groups) != len(prompt_rows):
            raise ValueError("SAM-HQ did not preserve the Box prompt count.")
        for (target, ratio, _box), refinements in zip(
            prompt_rows,
            candidate_groups,
            strict=True,
        ):
            target_mask = target["mask"]
            candidate_ious = [
                _mask_iou(target_mask, refinement.mask) for refinement in refinements
            ]
            selected_index = max(
                range(len(refinements)),
                key=lambda index: refinements[index].mask_quality,
            )
            oracle_index = max(
                range(len(refinements)),
                key=candidate_ious.__getitem__,
            )
            selected = refinements[selected_index]
            cases.append(
                {
                    "source_image_id": image_id,
                    "source_annotation_id": target["annotation_id"],
                    "target_label": target["label"],
                    "target_area_pixels": int(target_mask.sum()),
                    "target_area_ratio": float(target_mask.mean()),
                    "prompt_box": selected.prompt_box,
                    "mask_box": selected.mask_box,
                    "mask_quality": selected.mask_quality,
                    "mask_iou": candidate_ious[selected_index],
                    "box_expansion_ratio": ratio,
                    "candidate_count": len(refinements),
                    "score_selected_candidate_index": selected_index,
                    "oracle_best_candidate_index": oracle_index,
                    "oracle_best_mask_iou": candidate_ious[oracle_index],
                }
            )
        image_rows.append(
            {
                "source_image_id": image_id,
                "target_count": len(targets),
                "prompt_count": len(prompt_rows),
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"[{image_number}/{len(groups)}] image={image_id} "
            + f"targets={len(targets)} prompts={len(prompt_rows)} "
            + f"elapsed={elapsed:.3f}s"
        )

    summary = _summarize(cases, image_rows, args)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", summary)
    _write_json(output_dir / "cases.json", cases)
    _write_json(output_dir / "images.json", image_rows)
    for key, value in summary.items():
        print(f"{key}: {value}")


def _load_unique_targets(dataset: _DatasetProtocol) -> dict[int, _ImageGroup]:
    """Group unique annotation targets by image from repeated query rows."""
    groups: dict[int, _ImageGroup] = defaultdict(
        lambda: {"image_rgb": None, "targets": {}}
    )
    for item_index in range(len(dataset)):
        item = dataset[item_index]
        image_id = item.sample.source_image_id
        group = groups[image_id]
        if group["image_rgb"] is None:
            group["image_rgb"] = item.image_rgb
        elif not np.array_equal(group["image_rgb"], item.image_rgb):
            raise ValueError(f"Image {image_id} decoded inconsistently.")
        target_by_id = {
            target.source_annotation_id: target for target in item.sample.targets
        }
        for annotation_id, mask, box_values in zip(
            item.source_annotation_ids,
            item.target_masks,
            item.target_boxes,
            strict=True,
        ):
            target = target_by_id[annotation_id]
            row: _TargetRow = {
                "annotation_id": annotation_id,
                "label": target.label,
                "mask": np.asarray(mask, dtype=bool),
                "box": (
                    float(box_values[0]),
                    float(box_values[1]),
                    float(box_values[2]),
                    float(box_values[3]),
                ),
            }
            previous = group["targets"].get(annotation_id)
            if previous is not None and (
                previous["label"] != row["label"]
                or previous["box"] != row["box"]
                or not np.array_equal(previous["mask"], row["mask"])
            ):
                raise ValueError(f"Annotation {annotation_id} decoded inconsistently.")
            group["targets"][annotation_id] = row
    return dict(groups)


def _mask_iou(target_mask: np.ndarray, prediction_mask: np.ndarray) -> float:
    """Return binary Mask IoU, retaining empty predictions as zero."""
    target = np.asarray(target_mask, dtype=bool)
    prediction = np.asarray(prediction_mask, dtype=bool)
    if target.shape != prediction.shape or not target.any():
        raise ValueError("Mask IoU requires equal shapes and non-empty GT.")
    intersection = int(np.logical_and(target, prediction).sum())
    union = int(np.logical_or(target, prediction).sum())
    return intersection / union if union else 0.0


def _parse_expansion_ratios(value: str) -> tuple[float, ...]:
    """Parse unique bounded expansion ratios and require a zero baseline."""
    try:
        ratios = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Box expansion ratios must be comma-separated numbers."
        ) from error
    if not ratios or len(set(ratios)) != len(ratios):
        raise argparse.ArgumentTypeError(
            "Box expansion ratios must be non-empty and unique."
        )
    if any(not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0 for ratio in ratios):
        raise argparse.ArgumentTypeError("Box expansion ratios must be in [0, 1].")
    if 0.0 not in ratios:
        raise argparse.ArgumentTypeError("Box expansion ratios must include zero.")
    return ratios


def _expand_box(
    box: tuple[float, float, float, float],
    ratio: float,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Expand each Box side by a fraction of its width and height."""
    x_min, y_min, x_max, y_max = box
    margin_x = (x_max - x_min) * ratio
    margin_y = (y_max - y_min) * ratio
    return (
        max(0.0, x_min - margin_x),
        max(0.0, y_min - margin_y),
        min(float(image_width), x_max + margin_x),
        min(float(image_height), y_max + margin_y),
    )


def _summarize(
    cases: list[_CaseRow],
    image_rows: list[_ImageRow],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Build JSON-safe oracle-Box refinement metrics."""
    by_expansion_ratio: dict[str, dict[str, float | int]] = {}
    for ratio in args.box_expansion_ratios:
        ratio_cases = [row for row in cases if row["box_expansion_ratio"] == ratio]
        selected_ious = np.asarray(
            [row["mask_iou"] for row in ratio_cases],
            dtype=float,
        )
        oracle_ious = np.asarray(
            [row["oracle_best_mask_iou"] for row in ratio_cases],
            dtype=float,
        )
        by_expansion_ratio[f"{ratio:.3f}"] = {
            "target_count": len(ratio_cases),
            "score_selected_recall50": float(np.mean(selected_ious >= 0.50)),
            "score_selected_recall75": float(np.mean(selected_ious >= 0.75)),
            "score_selected_mean_mask_iou": float(selected_ious.mean()),
            "oracle_best_recall50": float(np.mean(oracle_ious >= 0.50)),
            "oracle_best_recall75": float(np.mean(oracle_ious >= 0.75)),
            "oracle_best_mean_mask_iou": float(oracle_ious.mean()),
        }
    baseline = by_expansion_ratio["0.000"]
    latencies = np.asarray(
        [row["elapsed_seconds"] for row in image_rows],
        dtype=float,
    )
    warm = latencies[1:]
    return {
        "split": args.split,
        "selected_image_count": len(image_rows),
        "image_offset": args.image_offset,
        "target_region_count": int(baseline["target_count"]),
        "box_prompt_recall50": baseline["score_selected_recall50"],
        "box_prompt_recall75": baseline["score_selected_recall75"],
        "all_gt_mean_mask_iou": baseline["score_selected_mean_mask_iou"],
        "oracle_multimask_recall50": baseline["oracle_best_recall50"],
        "oracle_multimask_recall75": baseline["oracle_best_recall75"],
        "oracle_multimask_mean_mask_iou": baseline["oracle_best_mean_mask_iou"],
        "multimask_output": args.multimask_output,
        "box_expansion_ratios": args.box_expansion_ratios,
        "by_box_expansion_ratio": by_expansion_ratio,
        "first_image_including_model_load_seconds": float(latencies[0]),
        "warm_image_count": len(warm),
        "warm_mean_seconds": float(warm.mean()) if len(warm) else None,
        "warm_p95_seconds": float(np.percentile(warm, 95)) if len(warm) else None,
        "candidate_region_scope": "fashionpedia_oracle_gt_boxes",
        "oracle_box_prompt_evaluated": True,
        "language_ranking_evaluated": False,
        "prd_accuracy_92_passed": None,
        "prd_localization_30ms_passed": None,
    }


def _write_json(path: Path, value: object) -> None:
    """Write one deterministic UTF-8 JSON artifact."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
