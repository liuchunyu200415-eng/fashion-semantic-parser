#!/usr/bin/env bash
# Install the exact PRD 3.1.2 CUDA 12 deployment runtimes without pip caching.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_name="fashion-prd-312"
conda_executable="${CONDA_EXE:-/root/miniconda3/bin/conda}"
python_index="https://pypi.org/simple"
onnxruntime_cuda12_index="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
nvidia_index="https://pypi.nvidia.com"

if [[ ! -x "$conda_executable" ]]; then
  echo "Conda executable not found: $conda_executable" >&2
  exit 1
fi
if ! "$conda_executable" env list | awk '{print $1}' | grep -qx "$environment_name"; then
  echo "Conda environment not found: $environment_name" >&2
  exit 1
fi

cd "$project_root"

"$conda_executable" run --name "$environment_name" \
  python -c 'import platform; assert platform.python_version() == "3.10.12"'

"$conda_executable" run --name "$environment_name" \
  python -m pip install \
  --no-cache-dir \
  --index-url "$python_index" \
  'numpy==1.26.4' \
  'coloredlogs==15.0.1' \
  'flatbuffers==24.3.25' \
  'onnx==1.15.0' \
  'packaging==24.1' \
  'protobuf==4.25.3' \
  'sympy==1.12' \
  'wheel==0.43.0'

"$conda_executable" run --name "$environment_name" \
  python -m pip uninstall --yes onnxruntime onnxruntime-gpu

"$conda_executable" run --name "$environment_name" \
  python -m pip install \
  --no-cache-dir \
  --no-deps \
  --index-url "$onnxruntime_cuda12_index" \
  'onnxruntime-gpu==1.17.1'

"$conda_executable" run --name "$environment_name" \
  python -m pip install \
  --no-cache-dir \
  --no-deps \
  --index-url "$python_index" \
  --extra-index-url "$nvidia_index" \
  'tensorrt_bindings==8.6.1' \
  'tensorrt_libs==8.6.1' \
  'tensorrt==8.6.1.post1'

"$conda_executable" run --name "$environment_name" \
  python -m pip install \
  --no-cache-dir \
  --force-reinstall \
  --no-deps \
  --index-url "$python_index" \
  'nvidia-cublas-cu12==12.1.3.1' \
  'nvidia-cuda-nvrtc-cu12==12.1.105' \
  'nvidia-cuda-runtime-cu12==12.1.105' \
  'nvidia-cudnn-cu12==8.9.2.26'

"$conda_executable" run --name "$environment_name" \
  python -m pip check

"$conda_executable" run --name "$environment_name" \
  python scripts/check_prd_312_deployment_env.py
