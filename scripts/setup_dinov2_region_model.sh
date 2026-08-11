#!/usr/bin/env bash
# Install the pinned official DINOv2 source and ViT-S/14 weights locally.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_path="$project_root/external/dinov2"
repo_url="git@github.com:facebookresearch/dinov2.git"
repo_commit="7764ea0f912e53c92e82eb78a2a1631e92725fc8"
weights_path="$project_root/models/checkpoints/localization/dinov2_vits14_pretrain.pth"
partial_weights_path="$weights_path.partial"
weights_url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
weights_size_bytes="88283115"

mkdir -p "$(dirname "$repo_path")" "$(dirname "$weights_path")"

if [[ ! -d "$repo_path/.git" ]]; then
  if [[ -e "$repo_path" ]]; then
    echo "Refusing to overwrite non-Git path: $repo_path" >&2
    exit 1
  fi
  git clone --filter=blob:none "$repo_url" "$repo_path"
fi

remote_url="$(git -C "$repo_path" remote get-url origin)"
if [[ "$remote_url" != *"facebookresearch/dinov2.git" ]]; then
  echo "Unexpected DINOv2 origin: $remote_url" >&2
  exit 1
fi
worktree_has_files="false"
if find "$repo_path" -mindepth 1 -maxdepth 1 ! -name .git -print -quit \
  | grep -q .; then
  worktree_has_files="true"
fi
if [[ "$worktree_has_files" == "true" ]] \
  && [[ -n "$(git -C "$repo_path" status --porcelain)" ]]; then
  echo "Refusing to change a dirty DINOv2 checkout: $repo_path" >&2
  exit 1
fi

git -C "$repo_path" fetch --depth 1 origin "$repo_commit"
git -C "$repo_path" checkout --detach "$repo_commit"
if [[ -n "$(git -C "$repo_path" status --porcelain)" ]]; then
  echo "DINOv2 checkout is dirty after pinned checkout: $repo_path" >&2
  exit 1
fi

if [[ -f "$weights_path" ]]; then
  actual_size="$(wc -c < "$weights_path" | tr -d ' ')"
  if [[ "$actual_size" != "$weights_size_bytes" ]]; then
    echo "Existing DINOv2 weights have unexpected size: $actual_size" >&2
    exit 1
  fi
else
  curl --fail --location --retry 3 --continue-at - \
    --output "$partial_weights_path" \
    "$weights_url"
  actual_size="$(wc -c < "$partial_weights_path" | tr -d ' ')"
  if [[ "$actual_size" != "$weights_size_bytes" ]]; then
    echo "Downloaded DINOv2 weights have unexpected size: $actual_size" >&2
    exit 1
  fi
  mv "$partial_weights_path" "$weights_path"
fi

echo "dinov2_commit: $(git -C "$repo_path" rev-parse HEAD)"
echo "weights_path: $weights_path"
echo "weights_size_bytes: $(wc -c < "$weights_path" | tr -d ' ')"
