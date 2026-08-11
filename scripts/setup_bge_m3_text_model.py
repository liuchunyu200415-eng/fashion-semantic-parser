"""Download the pinned dense-only BGE-M3 text encoder assets."""

import argparse
import sys
from pathlib import Path

MODEL_NAME = "BAAI/bge-m3"
MODEL_REVISION = "3c06a359c08b8c49f1cab07e3eac8f846eb3a038"
WEIGHTS_SIZE_BYTES = 2271064456
REQUIRED_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse the optional Hugging Face endpoint."""
    parser = argparse.ArgumentParser(
        description="Download pinned BGE-M3 dense text-embedding assets."
    )
    parser.add_argument("--endpoint", default=None)
    return parser.parse_args()


def main() -> None:
    """Download a fixed snapshot and verify its required dense-model files."""
    args = parse_args()
    add_src_to_python_path()
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    from fashion_semantic_parser.common.paths import resolve_project_path

    model_path = resolve_project_path("models/checkpoints/localization/bge-m3")
    model_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REVISION,
        local_dir=model_path,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=list(REQUIRED_FILES),
        endpoint=args.endpoint,
    )
    missing = [name for name in REQUIRED_FILES if not (model_path / name).is_file()]
    if missing:
        raise RuntimeError(f"BGE-M3 snapshot is missing files: {missing}")
    weights_path = model_path / "model.safetensors"
    if weights_path.stat().st_size != WEIGHTS_SIZE_BYTES:
        raise RuntimeError(
            "BGE-M3 safetensors size mismatch: "
            f"expected={WEIGHTS_SIZE_BYTES} actual={weights_path.stat().st_size}"
        )
    revision_path = model_path / ".pinned_revision"
    revision_path.write_text(MODEL_REVISION + "\n", encoding="utf-8")
    print(f"model_name: {MODEL_NAME}")
    print(f"model_revision: {MODEL_REVISION}")
    print(f"weights_size_bytes: {weights_path.stat().st_size}")
    print(f"model_path: {model_path}")


if __name__ == "__main__":
    main()
