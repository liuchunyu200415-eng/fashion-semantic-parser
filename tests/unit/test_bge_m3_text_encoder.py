"""Tests for the complete-query BGE-M3 text feature path."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from fashion_semantic_parser.service.bge_m3_text_encoder import (
    BgeM3TextEncoder,
    BgeM3TextEncoderSettings,
    load_bge_m3_text_settings,
    validate_complete_queries,
)


def test_complete_queries_are_not_collapsed_or_rewritten() -> None:
    """Spatial, attribute, and relation modifiers must reach the text encoder."""
    queries = ["衣服左侧的袖口", "the silver zipper on the jacket"]

    assert validate_complete_queries(queries) == queries


def test_complete_queries_reject_blank_input() -> None:
    """Blank queries cannot create meaningless alignment supervision."""
    with pytest.raises(ValueError, match="non-empty"):
        validate_complete_queries(["   "])


def test_fp16_is_not_allowed_on_cpu() -> None:
    """CPU smoke cannot claim the CUDA fp16 text implementation path."""
    with pytest.raises(ValidationError, match="CUDA"):
        BgeM3TextEncoderSettings(device="cpu", precision="fp16")


def test_project_config_pins_multilingual_bge_m3() -> None:
    """The committed text smoke must retain fixed model identity and geometry."""
    settings = load_bge_m3_text_settings()

    assert settings.model_name == "BAAI/bge-m3"
    assert settings.model_revision == "3c06a359c08b8c49f1cab07e3eac8f846eb3a038"
    assert settings.embedding_dimension == 1024
    assert settings.max_length == 64
    assert settings.batch_size == 32


def test_local_assets_require_pinned_revision_and_weight_size(tmp_path: Path) -> None:
    """Runtime cannot accept a drifting snapshot or partial safetensors file."""
    model_path = tmp_path / "bge-m3"
    model_path.mkdir()
    (model_path / ".pinned_revision").write_text("a" * 40 + "\n", encoding="utf-8")
    weights_path = model_path / "model.safetensors"
    weights_path.write_bytes(b"1234")
    encoder = BgeM3TextEncoder(
        BgeM3TextEncoderSettings(
            model_revision="a" * 40,
            weights_size_bytes=4,
        )
    )

    encoder._validate_local_assets(model_path)

    weights_path.write_bytes(b"123")
    with pytest.raises(RuntimeError, match="size mismatch"):
        encoder._validate_local_assets(model_path)


def test_encode_caps_large_query_sets_to_configured_batch_size() -> None:
    """Large evaluation prefixes cannot become one unbounded CUDA batch."""
    observed: dict[str, int] = {}

    class FakeModel:
        def encode(self, queries, *, batch_size, **kwargs):
            observed["query_count"] = len(queries)
            observed["batch_size"] = batch_size
            return np.ones((len(queries), 1024), dtype=np.float32)

    encoder = BgeM3TextEncoder(
        BgeM3TextEncoderSettings(
            device="cpu",
            precision="fp32",
            batch_size=2,
        )
    )
    encoder._model = FakeModel()

    result = encoder.encode(["领口", "左侧袖口", "银色拉链"])

    assert result.shape == (3, 1024)
    assert observed == {"query_count": 3, "batch_size": 2}
