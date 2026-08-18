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
  optimum==1.21.4 \
  peft==0.12.0 \
  rouge==1.0.1 \
  sentencepiece==0.2.0 \
  tiktoken==0.7.0 \
  transformers-stream-generator==0.0.4

# Keep the PRD environment's pinned Torch and Transformers versions intact.
conda run --no-capture-output -n "$environment_name" \
  python -m pip install --no-deps auto-gptq==0.7.1

conda run --no-capture-output -n "$environment_name" \
  python "$project_root/scripts/setup_qwen_vl_paraphrase_model.py" \
  --model-path "$model_path" \
  --endpoint "$endpoint"

conda run --no-capture-output -n "$environment_name" \
  python -c \
  'import auto_gptq, tiktoken, transformers_stream_generator; print("qwen_vl_runtime: ready")'
