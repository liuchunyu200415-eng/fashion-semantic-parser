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


class _ImageRow(TypedDict):
    """One persisted image-level runtime result."""

    source_image_id: int
    target_count: int
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
        boxes = [target["box"] for target in targets]
        started = time.perf_counter()
        refinements = refiner.refine(group["image_rgb"], boxes)
        refiner.synchronize()
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        if len(refinements) != len(targets):
            raise ValueError("SAM-HQ did not preserve the Box prompt count.")
        for target, refinement in zip(targets, refinements, strict=True):
            target_mask = target["mask"]
            cases.append(
                {
                    "source_image_id": image_id,
                    "source_annotation_id": target["annotation_id"],
                    "target_label": target["label"],
                    "target_area_pixels": int(target_mask.sum()),
                    "target_area_ratio": float(target_mask.mean()),
                    "prompt_box": refinement.prompt_box,
                    "mask_box": refinement.mask_box,
                    "mask_quality": refinement.mask_quality,
                    "mask_iou": _mask_iou(target_mask, refinement.mask),
                }
            )
        image_rows.append(
            {
                "source_image_id": image_id,
                "target_count": len(targets),
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"[{image_number}/{len(groups)}] image={image_id} "
            + f"targets={len(targets)} elapsed={elapsed:.3f}s"
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


def _summarize(
    cases: list[_CaseRow],
    image_rows: list[_ImageRow],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Build JSON-safe oracle-Box refinement metrics."""
    ious = np.asarray([row["mask_iou"] for row in cases], dtype=float)
    latencies = np.asarray(
        [row["elapsed_seconds"] for row in image_rows],
        dtype=float,
    )
    warm = latencies[1:]
    return {
        "split": args.split,
        "selected_image_count": len(image_rows),
        "image_offset": args.image_offset,
        "target_region_count": len(cases),
        "box_prompt_recall50": float(np.mean(ious >= 0.50)),
        "box_prompt_recall75": float(np.mean(ious >= 0.75)),
        "all_gt_mean_mask_iou": float(ious.mean()),
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
