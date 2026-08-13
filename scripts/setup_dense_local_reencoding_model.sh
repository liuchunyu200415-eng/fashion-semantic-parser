#!/usr/bin/env bash
# Link the frozen AutoDL checkpoint into the production model path.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_checkpoint="${1:-${project_root}/outputs/localization/dinov2_multiscale_728_train1000_steps1500/dense_patch_alignment.pt}"
target_checkpoint="${project_root}/models/checkpoints/localization/dinov2_multiscale_728_train1000_steps1500.pt"

if [[ ! -s "${source_checkpoint}" ]]; then
  echo "Missing frozen checkpoint: ${source_checkpoint}" >&2
  exit 1
fi

mkdir -p "$(dirname "${target_checkpoint}")"
if [[ -e "${target_checkpoint}" && ! -L "${target_checkpoint}" ]]; then
  echo "Refusing to overwrite a non-symlink checkpoint: ${target_checkpoint}" >&2
  exit 1
fi
ln -sfn "${source_checkpoint}" "${target_checkpoint}"

test -s "${target_checkpoint}"
echo "dense_local_checkpoint: $(readlink -f "${target_checkpoint}")"
