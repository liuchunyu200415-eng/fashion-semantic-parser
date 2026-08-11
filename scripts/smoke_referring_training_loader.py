"""Smoke-test Fashionpedia referring JSONL loading through PyTorch."""

import argparse
import sys
from pathlib import Path


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse a bounded DataLoader smoke run."""
    parser = argparse.ArgumentParser(
        description="Decode official Fashionpedia target Masks for referring samples."
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--index", default=None)
    parser.add_argument("--annotations", default=None)
    return parser.parse_args()


def main() -> None:
    """Load a few variable-size, multi-target batches and print bounded counts."""
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be at least one")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least one")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    add_src_to_python_path()

    try:
        from torch.utils.data import DataLoader  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the DataLoader smoke.") from error

    from fashion_semantic_parser.common.paths import PROJECT_ROOT, resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.localization.referring_dataset import (
        FashionpediaReferringDataset,
        collate_referring_training_items,
    )

    settings = load_settings()
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
        f"{settings.datasets.fashionpedia_root}/annotations/{annotation_name}"
    )
    dataset = FashionpediaReferringDataset(
        index_path=resolve_project_path(index),
        annotation_path=resolve_project_path(annotations),
        project_root=PROJECT_ROOT,
        max_samples=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_referring_training_items,
    )

    sample_count = 0
    target_count = 0
    mask_pixel_count = 0
    dimension_counts: dict[str, int] = {}
    for batch in loader:
        sample_count += len(batch["samples"])
        target_count += sum(masks.shape[0] for masks in batch["target_masks"])
        mask_pixel_count += sum(int(masks.sum()) for masks in batch["target_masks"])
        for dimensions in batch["dimensions"]:
            for dimension in dimensions:
                dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1

    print(f"split: {args.split}")
    print(f"sample_count: {sample_count}")
    print(f"target_count: {target_count}")
    print(f"mask_pixel_count: {mask_pixel_count}")
    print(f"dimension_counts: {dict(sorted(dimension_counts.items()))}")
    print(f"batch_size: {args.batch_size}")
    print(f"workers: {args.workers}")


if __name__ == "__main__":
    main()
