#!/usr/bin/env bash
# Install the pinned official SAM-HQ source used by automatic proposals.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_path="$project_root/external/sam-hq"
repo_url="git@github.com:SysCV/sam-hq.git"
repo_commit="e696978d60352dc9a26b12631cd91781502c6546"
weights_path="$project_root/models/checkpoints/localization/sam_hq_vit_b.pth"
weights_sha256="14a9d662cd6f5a9c2dba6d40ab0058d88d287e4a18fd6fdc6ad5fb1a3fdeaa57"
conda_executable="${CONDA_EXE:-/root/miniconda3/bin/conda}"

mkdir -p "$(dirname "$repo_path")"

if [[ ! -d "$repo_path/.git" ]]; then
  if [[ -e "$repo_path" ]]; then
    echo "Refusing to overwrite non-Git path: $repo_path" >&2
    exit 1
  fi
  git clone --filter=blob:none "$repo_url" "$repo_path"
fi

remote_url="$(git -C "$repo_path" remote get-url origin)"
if [[ "$remote_url" != *"SysCV/sam-hq.git" ]]; then
  echo "Unexpected SAM-HQ origin: $remote_url" >&2
  exit 1
fi
if [[ -n "$(git -C "$repo_path" status --porcelain)" ]]; then
  echo "Refusing to change a dirty SAM-HQ checkout: $repo_path" >&2
  exit 1
fi

git -C "$repo_path" fetch --depth 1 origin "$repo_commit"
git -C "$repo_path" checkout --detach "$repo_commit"
if [[ -n "$(git -C "$repo_path" status --porcelain)" ]]; then
  echo "SAM-HQ checkout is dirty after pinned checkout: $repo_path" >&2
  exit 1
fi

if [[ ! -s "$weights_path" ]]; then
  echo "SAM-HQ weights are missing: $weights_path" >&2
  echo "Restore the previously downloaded official ViT-B checkpoint first." >&2
  exit 1
fi
echo "$weights_sha256  $weights_path" | sha256sum -c -

PYTHONPATH="$repo_path${PYTHONPATH:+:$PYTHONPATH}" \
  "$conda_executable" run --name fashion-prd-312 \
  python -c \
  "from segment_anything import SamAutomaticMaskGenerator; print('sam_hq_import: ready')"

echo "sam_hq_commit: $(git -C "$repo_path" rev-parse HEAD)"
echo "sam_hq_repo: $repo_path"
echo "sam_hq_weights: $weights_path"
