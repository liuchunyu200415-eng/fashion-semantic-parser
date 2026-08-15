#!/usr/bin/env bash
# Build the pinned Detectron2 runtime inside the isolated PRD 3.1.2 environment.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_name="fashion-prd-312"
conda_executable="${CONDA_EXE:-/root/miniconda3/bin/conda}"
repository_path="$project_root/external/detectron2"
repository_url="git@github.com:facebookresearch/detectron2.git"
repository_commit="d1e04565d3bec8719335b88be9e9b961bf3ec464"
mask2former_path="$project_root/external/Mask2Former"
cuda_architecture="8.6"
wheel_directory="$(mktemp -d /tmp/fashion-prd-312-detectron2.XXXXXX)"
trap 'rm -rf "$wheel_directory"' EXIT

if [[ ! -x "$conda_executable" ]]; then
  echo "Conda executable not found: $conda_executable" >&2
  exit 1
fi
if ! "$conda_executable" env list | awk '{print $1}' | grep -qx "$environment_name"; then
  echo "Conda environment not found: $environment_name" >&2
  exit 1
fi
if [[ ! -d "$mask2former_path/mask2former" ]]; then
  echo "Mask2Former checkout is missing: $mask2former_path" >&2
  exit 1
fi

mkdir -p "$(dirname "$repository_path")"
if [[ ! -d "$repository_path/.git" ]]; then
  if [[ -e "$repository_path" ]]; then
    echo "Refusing to overwrite non-Git path: $repository_path" >&2
    exit 1
  fi
  git clone --filter=blob:none "$repository_url" "$repository_path"
fi

remote_url="$(git -C "$repository_path" remote get-url origin)"
if [[ "$remote_url" != *"facebookresearch/detectron2.git" ]]; then
  echo "Unexpected Detectron2 origin: $remote_url" >&2
  exit 1
fi
if [[ -n "$(git -C "$repository_path" status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to change a dirty Detectron2 checkout: $repository_path" >&2
  exit 1
fi

git -C "$repository_path" fetch --depth 1 origin "$repository_commit"
git -C "$repository_path" checkout --detach "$repository_commit"
if [[ "$(git -C "$repository_path" rev-parse HEAD)" != "$repository_commit" ]]; then
  echo "Detectron2 checkout does not match the pinned commit." >&2
  exit 1
fi

"$conda_executable" run --name "$environment_name" \
  python -m pip install --no-cache-dir \
  'setuptools==80.9.0' \
  'black==21.4b2' \
  'click==8.0.4' \
  'cloudpickle==3.0.0' \
  'future==0.18.3' \
  'fvcore==0.1.5.post20221221' \
  'hydra-core==1.3.2' \
  'iopath==0.1.9' \
  'matplotlib==3.9.2' \
  'omegaconf==2.3.0' \
  'pathspec==0.9.0' \
  'pydot==1.4.2' \
  'tabulate==0.9.0' \
  'tensorboard==2.17.1' \
  'termcolor==2.4.0' \
  'tqdm==4.66.5' \
  'yacs==0.1.8'

# PyTorch 2.1.2 imports ``pkg_resources`` while loading its C++/CUDA extension
# helper. Setuptools 81+ removes that compatibility module, so fail before the
# expensive Detectron2 build if the pinned build toolchain is not usable.
"$conda_executable" run --name "$environment_name" \
  python -c \
  'import pkg_resources; from torch.utils.cpp_extension import CUDA_HOME; print(f"detectron2_build_cuda_home: {CUDA_HOME}")'

env \
  FORCE_CUDA=1 \
  TORCH_CUDA_ARCH_LIST="$cuda_architecture" \
  MAX_JOBS="${MAX_JOBS:-2}" \
  "$conda_executable" run --name "$environment_name" \
  python -m pip wheel \
  --no-cache-dir \
  --no-build-isolation \
  --no-deps \
  --wheel-dir "$wheel_directory" \
  "$repository_path"

mapfile -t detectron2_wheels < <(
  find "$wheel_directory" -maxdepth 1 -type f -name 'detectron2-*.whl'
)
if [[ "${#detectron2_wheels[@]}" -ne 1 ]]; then
  echo "Expected one Detectron2 wheel, found ${#detectron2_wheels[@]}." >&2
  exit 1
fi

"$conda_executable" run --name "$environment_name" \
  python -m pip install \
  --no-cache-dir \
  --no-deps \
  --force-reinstall \
  "${detectron2_wheels[0]}"

"$conda_executable" run --name "$environment_name" python -m pip check

PYTHONPATH="$project_root/src" \
  "$conda_executable" run --name "$environment_name" \
  python "$project_root/scripts/check_prd_312_detectron2_env.py"

echo "detectron2_commit: $(git -C "$repository_path" rev-parse HEAD)"
echo "detectron2_repo: $repository_path"
