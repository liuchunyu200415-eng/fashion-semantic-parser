"""Encode complete multilingual referring expressions with pinned BGE-M3."""

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator

from fashion_semantic_parser.common.paths import resolve_project_path


class BgeM3TextEncoderSettings(BaseModel):
    """Validated dense BGE-M3 text-embedding smoke configuration."""

    model_name: Literal["BAAI/bge-m3"] = "BAAI/bge-m3"
    model_revision: str = Field(
        default="3c06a359c08b8c49f1cab07e3eac8f846eb3a038",
        pattern=r"^[0-9a-f]{40}$",
    )
    model_path: str = "models/checkpoints/localization/bge-m3"
    weights_size_bytes: int = Field(default=2271064456, ge=1)
    embedding_dimension: Literal[1024] = 1024
    max_length: int = Field(default=64, ge=4, le=512)
    batch_size: int = Field(default=32, ge=1, le=256)
    device: Literal["cuda", "cpu"] = "cuda"
    precision: Literal["fp16", "fp32"] = "fp16"

    @model_validator(mode="after")
    def validate_precision(self) -> "BgeM3TextEncoderSettings":
        """Keep fp16 on the PRD CUDA path only."""
        if self.device == "cpu" and self.precision == "fp16":
            raise ValueError("BGE-M3 fp16 smoke requires the CUDA device.")
        return self


class BgeM3TextEncoder:
    """Pinned dense BGE-M3 encoder that preserves complete query strings."""

    def __init__(self, settings: BgeM3TextEncoderSettings) -> None:
        self.settings = settings
        self._torch: Any | None = None
        self._model: Any | None = None

    def load(self) -> None:
        """Load local BGE-M3 files without any runtime network fallback."""
        if self._model is not None:
            return
        model_path = resolve_project_path(self.settings.model_path)
        self._validate_local_assets(model_path)
        try:
            import torch  # type: ignore[import-not-found]
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers and PyTorch are required for BGE-M3."
            ) from error
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("BGE-M3 CUDA smoke requested but CUDA is unavailable.")
        model_kwargs = {
            "torch_dtype": (
                torch.float16 if self.settings.precision == "fp16" else torch.float32
            )
        }
        model = SentenceTransformer(
            str(model_path),
            device=self.settings.device,
            local_files_only=True,
            trust_remote_code=False,
            model_kwargs=model_kwargs,
        )
        model.max_seq_length = self.settings.max_length
        self._model = model.eval()
        self._torch = torch

    def encode(self, queries: list[str]) -> np.ndarray:
        """Return one normalized 1024-D embedding per uncollapsed full query."""
        validated_queries = validate_complete_queries(queries)
        self.load()
        if self._model is None:
            raise RuntimeError("BGE-M3 model did not initialize.")
        embeddings = self._model.encode(
            validated_queries,
            batch_size=min(self.settings.batch_size, len(validated_queries)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        result: np.ndarray = np.asarray(embeddings, dtype=np.float32)
        if result.shape != (
            len(validated_queries),
            self.settings.embedding_dimension,
        ):
            raise ValueError(f"Unexpected BGE-M3 embedding shape: {result.shape}")
        return result

    def synchronize(self) -> None:
        """Synchronize CUDA so smoke timing includes completed text encoding."""
        if (
            self._torch is not None
            and self.settings.device == "cuda"
            and self._torch.cuda.is_available()
        ):
            self._torch.cuda.synchronize()

    def _validate_local_assets(self, model_path: Path) -> None:
        """Reject missing, drifting, or partial pinned BGE-M3 assets."""
        revision_path = model_path / ".pinned_revision"
        if not revision_path.is_file():
            raise RuntimeError(
                "Pinned BGE-M3 assets are missing; run "
                "scripts/setup_bge_m3_text_model.py."
            )
        actual_revision = revision_path.read_text(encoding="utf-8").strip()
        if actual_revision != self.settings.model_revision:
            raise RuntimeError(
                "BGE-M3 revision mismatch: "
                f"expected={self.settings.model_revision} actual={actual_revision}"
            )
        weights_path = model_path / "model.safetensors"
        if not weights_path.is_file():
            raise RuntimeError("BGE-M3 safetensors weights are missing.")
        actual_size = weights_path.stat().st_size
        if actual_size != self.settings.weights_size_bytes:
            raise RuntimeError(
                "BGE-M3 safetensors size mismatch: "
                f"expected={self.settings.weights_size_bytes} actual={actual_size}"
            )


def load_bge_m3_text_settings(
    config_path: str | Path = "configs/localization_bge_m3_text.yaml",
) -> BgeM3TextEncoderSettings:
    """Load one project-relative BGE-M3 text configuration."""
    path = resolve_project_path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cast(
        BgeM3TextEncoderSettings,
        BgeM3TextEncoderSettings.model_validate(raw),
    )


def validate_complete_queries(queries: list[str]) -> list[str]:
    """Reject blanks while preserving every complete query verbatim."""
    if not queries:
        raise ValueError("At least one complete query is required.")
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("BGE-M3 queries must be non-empty strings.")
    return list(queries)
