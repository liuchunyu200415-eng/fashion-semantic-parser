"""Measure the class-agnostic SAM-HQ proposal ceiling on Fashionpedia parts."""

import argparse
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
    """Parse one image-complete proposal-recall smoke."""
    parser = argparse.ArgumentParser(
        description="Measure SAM-HQ proposal Mask recall before language ranking."
    )
    parser.add_argument(
        "--split", choices=("train", "validation"), default="validation"
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
        default="outputs/localization/sam_hq_proposal_recall_smoke",
    )
    return parser.parse_args()


def main() -> None:
    """Generate proposals once per image and retain every GT miss in metrics."""
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
        SAMHQAutomaticProposalGenerator,
        best_proposal_mask_iou,
        load_sam_hq_proposal_settings,
    )

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
        max_images=args.image_limit,
        image_offset=args.image_offset,
    )
    groups: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"image_rgb": None, "masks": {}}
    )
    for item_index in range(len(dataset)):
        item = dataset[item_index]
        image_id = item.sample.source_image_id
        group = groups[image_id]
        if group["image_rgb"] is None:
            group["image_rgb"] = item.image_rgb
        elif not np.array_equal(group["image_rgb"], item.image_rgb):
            raise ValueError(f"Image {image_id} decoded inconsistently.")
        for annotation_id, mask in zip(
            item.source_annotation_ids,
            item.target_masks,
        ):
            previous = group["masks"].get(annotation_id)
            if previous is not None and not np.array_equal(previous, mask):
                raise ValueError(f"Annotation {annotation_id} decoded inconsistently.")
            group["masks"][annotation_id] = mask
    if not groups:
        raise ValueError("SAM-HQ proposal smoke loaded no images.")

    generator = SAMHQAutomaticProposalGenerator(
        load_sam_hq_proposal_settings(args.config)
    )
    cases: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for image_number, (image_id, group) in enumerate(groups.items(), start=1):
        started = time.perf_counter()
        proposals = generator.generate(group["image_rgb"])
        generator.synchronize()
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        for annotation_id, target_mask in sorted(group["masks"].items()):
            best_iou, best_index = best_proposal_mask_iou(target_mask, proposals)
            cases.append(
                {
                    "source_image_id": image_id,
                    "source_annotation_id": annotation_id,
                    "proposal_count": len(proposals),
                    "best_proposal_index": best_index,
                    "best_mask_iou": best_iou,
                }
            )
        image_rows.append(
            {
                "source_image_id": image_id,
                "target_count": len(group["masks"]),
                "proposal_count": len(proposals),
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"[{image_number}/{len(groups)}] image={image_id} "
            f"targets={len(group['masks'])} proposals={len(proposals)} "
            f"elapsed={elapsed:.2f}s"
        )

    best_ious = np.asarray([row["best_mask_iou"] for row in cases], dtype=float)
    warm = np.asarray(latencies[1:], dtype=float)
    summary = {
        "split": args.split,
        "selected_image_count": len(groups),
        "image_offset": args.image_offset,
        "target_region_count": len(cases),
        "proposal_count": sum(row["proposal_count"] for row in image_rows),
        "mean_proposals_per_image": float(
            np.mean([row["proposal_count"] for row in image_rows])
        ),
        "proposal_recall50": float(np.mean(best_ious >= 0.50)),
        "proposal_recall75": float(np.mean(best_ious >= 0.75)),
        "all_gt_mean_best_mask_iou": float(best_ious.mean()),
        "first_image_including_model_load_seconds": latencies[0],
        "warm_image_count": len(warm),
        "warm_mean_seconds": float(warm.mean()) if len(warm) else None,
        "warm_p95_seconds": float(np.percentile(warm, 95)) if len(warm) else None,
        "candidate_region_scope": "sam_hq_automatic_masks_full_image",
        "proposal_mask_recall_evaluated": True,
        "language_ranking_evaluated": False,
        "prd_accuracy_92_passed": None,
        "prd_localization_30ms_passed": None,
    }
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
    (output_dir / "images.json").write_text(
        json.dumps(image_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
