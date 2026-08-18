#!/usr/bin/env bash
# Install Qwen-VL INT4 runtime dependencies and download its pinned snapshot.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_name="${QWEN_VL_CONDA_ENV:-fashion-prd-312}"
model_path="${QWEN_VL_MODEL_PATH:-$project_root/models/checkpoints/localization/qwen-vl-chat-int4}"
endpoint="${QWEN_VL_ENDPOINT:-https://hf-mirror.com}"

conda run --no-capture-output -n "$environment_name" \
  python -m pip install \
  accelerate==0.33.0 \
  datasets==2.21.0 \
  einops==0.8.0 \
  gekko==1.2.1 \
  optimum==1.22.0 \
  peft==0.12.0 \
  rouge==1.0.1 \
  sentencepiece==0.2.0 \
  tiktoken==0.7.0 \
  transformers-stream-generator==0.0.4

# Optimum 1.22 supports Transformers 4.44; restore the project pins explicitly.
conda run --no-capture-output -n "$environment_name" \
  python -m pip install --no-deps \
  transformers==4.44.2 \
  tokenizers==0.19.1 \
  huggingface-hub==0.24.6

conda run --no-capture-output -n "$environment_name" \
  python -m pip install --no-deps auto-gptq==0.7.1

conda run --no-capture-output -n "$environment_name" \
  python -m pip check

conda run --no-capture-output -n "$environment_name" \
  python "$project_root/scripts/setup_qwen_vl_paraphrase_model.py" \
  --model-path "$model_path" \
  --endpoint "$endpoint"

conda run --no-capture-output -n "$environment_name" \
  python -c \
  'from importlib.metadata import version; import auto_gptq, optimum, tiktoken, transformers_stream_generator; print("optimum:", version("optimum")); print("qwen_vl_runtime: ready")'
