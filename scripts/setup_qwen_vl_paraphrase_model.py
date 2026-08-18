"""Download the pinned official PRD Qwen-VL-Chat INT4 checkpoint."""

# Direct execution adds ``src`` and optional model dependencies at runtime.
# pylint: disable=import-outside-toplevel

import argparse
import shutil
import sys
from pathlib import Path

MODEL_NAME = "Qwen/Qwen-VL-Chat-Int4"
MODEL_REVISION = "55acaf444e9f5adfd47105b875571a23d7f7fa30"
MINIMUM_FREE_BYTES = 11_000_000_000
REQUIRED_FILES = (
    "config.json",
    "configuration_qwen.py",
    "generation_config.json",
    "model.safetensors.index.json",
    "modeling_qwen.py",
    "quantize_config.json",
    "qwen.tiktoken",
    "qwen_generation_utils.py",
    "tokenization_qwen.py",
    "tokenizer_config.json",
    "visual.py",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
)


def add_src_to_python_path() -> None:
    """Add the local source package when the project is not installed."""
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    """Parse model storage and optional Hugging Face mirror settings."""
    parser = argparse.ArgumentParser(
        description="Download pinned Qwen-VL-Chat-Int4 for PRD data expansion."
    )
    parser.add_argument(
        "--model-path",
        default="models/checkpoints/localization/qwen-vl-chat-int4",
    )
    parser.add_argument("--endpoint", default=None)
    return parser.parse_args()


def main() -> None:
    """Download one immutable official snapshot and validate its files."""
    args = parse_args()
    add_src_to_python_path()
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    from fashion_semantic_parser.common.paths import resolve_project_path

    model_path = resolve_project_path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED_FILES if not (model_path / name).is_file()]
    existing_weights_size = sum(
        path.stat().st_size for path in model_path.glob("model-*.safetensors")
    )
    if missing or existing_weights_size < 9_000_000_000:
        free_bytes = shutil.disk_usage(model_path).free
        if free_bytes < MINIMUM_FREE_BYTES:
            raise RuntimeError(
                "Qwen-VL-Chat-Int4 needs at least 11 GB free on the selected "
                f"volume; available={free_bytes} path={model_path}"
            )
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
        raise RuntimeError(f"Qwen-VL snapshot is missing files: {missing}")
    weights_size = sum(
        path.stat().st_size for path in model_path.glob("model-*.safetensors")
    )
    if weights_size < 9_000_000_000:
        raise RuntimeError(f"Qwen-VL INT4 weight shards are incomplete: {weights_size}")
    (model_path / ".pinned_revision").write_text(
        MODEL_REVISION + "\n",
        encoding="utf-8",
    )
    print(f"model_name: {MODEL_NAME}")
    print(f"model_revision: {MODEL_REVISION}")
    print(f"weights_size_bytes: {weights_size}")
    print(f"model_path: {model_path}")


if __name__ == "__main__":
    main()
