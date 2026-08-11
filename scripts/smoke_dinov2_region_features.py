"""Smoke-test official DINOv2 Mask-pooled Fashionpedia region features."""

import argparse
import statistics
import sys
import time
from pathlib import Path


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse one bounded DINOv2 region-feature smoke."""
    parser = argparse.ArgumentParser(
        description="Extract official DINOv2 features from Fashionpedia Masks."
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--config",
        default="configs/localization_dinov2_region.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Load the model once and report cold/warm feature shape and latency."""
    args = parse_args()
    if args.limit < 2:
        raise ValueError("--limit must be at least two for cold/warm timing.")
    add_src_to_python_path()

    import numpy as np

    from fashion_semantic_parser.common.paths import PROJECT_ROOT, resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.localization.referring_dataset import (
        FashionpediaReferringDataset,
    )
    from fashion_semantic_parser.service.dinov2_region_encoder import (
        DinoV2RegionEncoder,
        load_dinov2_region_settings,
    )

    settings = load_settings()
    annotation_name = (
        "instances_attributes_train2020.json"
        if args.split == "train"
        else "instances_attributes_val2020.json"
    )
    dataset = FashionpediaReferringDataset(
        index_path=resolve_project_path(
            "data/processed/autodl/localization/"
            f"fashionpedia_referring_{args.split}.jsonl"
        ),
        annotation_path=resolve_project_path(
            f"{settings.datasets.fashionpedia_root}/annotations/{annotation_name}"
        ),
        project_root=PROJECT_ROOT,
        max_samples=args.limit,
    )
    encoder = DinoV2RegionEncoder(load_dinov2_region_settings(args.config))

    load_started = time.perf_counter()
    encoder.load()
    encoder.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    elapsed_seconds: list[float] = []
    target_count = 0
    feature_rows = []
    for item in dataset:
        encoder.synchronize()
        started = time.perf_counter()
        features = encoder.encode(item.image_rgb, item.target_masks)
        encoder.synchronize()
        elapsed_seconds.append(time.perf_counter() - started)
        if features.shape[0] != len(item.sample.targets):
            raise ValueError("DINOv2 feature count does not match target count.")
        target_count += features.shape[0]
        feature_rows.append(features)

    all_features = np.concatenate(feature_rows, axis=0)
    norms = np.linalg.norm(all_features, axis=1)
    warm_seconds = elapsed_seconds[1:]
    print(f"model: {encoder.settings.model_name}")
    print(f"model_load_seconds: {model_load_seconds:.3f}")
    print(f"sample_count: {len(dataset)}")
    print(f"target_count: {target_count}")
    print(f"feature_shape: {tuple(all_features.shape)}")
    print(f"feature_norm_range: {norms.min():.6f}..{norms.max():.6f}")
    print(f"first_encode_ms: {elapsed_seconds[0] * 1000.0:.3f}")
    print(f"warm_mean_ms: {statistics.fmean(warm_seconds) * 1000.0:.3f}")
    print(f"warm_max_ms: {max(warm_seconds) * 1000.0:.3f}")
    print("prd_localization_30ms_passed: not_evaluated")


if __name__ == "__main__":
    main()
