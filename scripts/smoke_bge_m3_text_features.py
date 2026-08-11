"""Smoke-test complete-query BGE-M3 dense text features and latency."""

import statistics
import sys
import time
from pathlib import Path

QUERIES = (
    "这件衣服的领口",
    "衣服左侧的袖口",
    "the silver zipper on the jacket",
    "外套里面的内搭",
)


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    """Report cold/warm text embedding shape, norms, and bounded latency."""
    add_src_to_python_path()
    import numpy as np

    from fashion_semantic_parser.service.bge_m3_text_encoder import (
        BgeM3TextEncoder,
        load_bge_m3_text_settings,
    )

    encoder = BgeM3TextEncoder(load_bge_m3_text_settings())
    load_started = time.perf_counter()
    encoder.load()
    encoder.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    elapsed_seconds: list[float] = []
    embedding_rows = []
    for query in QUERIES:
        encoder.synchronize()
        started = time.perf_counter()
        embedding_rows.append(encoder.encode([query]))
        encoder.synchronize()
        elapsed_seconds.append(time.perf_counter() - started)

    embeddings = np.concatenate(embedding_rows, axis=0)
    norms = np.linalg.norm(embeddings, axis=1)
    warm_seconds = elapsed_seconds[1:]
    print(f"model: {encoder.settings.model_name}")
    print(f"model_load_seconds: {model_load_seconds:.3f}")
    print(f"query_count: {len(QUERIES)}")
    print(f"embedding_shape: {tuple(embeddings.shape)}")
    print(f"embedding_norm_range: {norms.min():.6f}..{norms.max():.6f}")
    print(f"first_encode_ms: {elapsed_seconds[0] * 1000.0:.3f}")
    print(f"warm_mean_ms: {statistics.fmean(warm_seconds) * 1000.0:.3f}")
    print(f"warm_max_ms: {max(warm_seconds) * 1000.0:.3f}")
    print("dinov2_text_alignment_trained: false")
    print("prd_localization_30ms_passed: not_evaluated")


if __name__ == "__main__":
    main()
