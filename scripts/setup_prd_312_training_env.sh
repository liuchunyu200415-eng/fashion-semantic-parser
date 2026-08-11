#!/usr/bin/env bash
# Create or repair the isolated PRD 3.1.2 training environment.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_name="fashion-prd-312"
conda_executable="${CONDA_EXE:-/root/miniconda3/bin/conda}"
environment_file="$project_root/environment/prd_3_1_2_training.yaml"
temporary_condarc="$(mktemp /tmp/fashion-prd-312-condarc.XXXXXX.yaml)"

cleanup() {
  rm -f "$temporary_condarc"
}
trap cleanup EXIT

if [[ ! -x "$conda_executable" ]]; then
  echo "Conda executable not found: $conda_executable" >&2
  exit 1
fi

printf '{}\n' > "$temporary_condarc"
cd "$project_root"

if "$conda_executable" env list | awk '{print $1}' | grep -qx "$environment_name"; then
  echo "Repairing existing environment: $environment_name"
  CONDARC="$temporary_condarc" "$conda_executable" env update \
    --name "$environment_name" \
    --file "$environment_file" \
    --prune
else
  echo "Creating environment: $environment_name"
  CONDARC="$temporary_condarc" "$conda_executable" env create \
    --file "$environment_file"
fi

"$conda_executable" run --name "$environment_name" \
  python scripts/check_prd_312_training_env.py
