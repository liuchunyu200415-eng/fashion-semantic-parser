#!/usr/bin/env bash
# Run every repository quality gate with the active Python environment.

set -euo pipefail

python_command="${PYTHON_COMMAND:-python}"
quality_roots=(src scripts tests)

"$python_command" -m black --check "${quality_roots[@]}"
"$python_command" -m isort --check-only "${quality_roots[@]}"
"$python_command" -m flake8 \
  --max-line-length 88 \
  --extend-ignore E203,W503 \
  "${quality_roots[@]}"
"$python_command" -m mypy \
  --ignore-missing-imports \
  --explicit-package-bases \
  src scripts
"$python_command" -m pylint --errors-only src scripts
"$python_command" scripts/check_coding_standards.py
"$python_command" -m pytest -q \
  --cov=fashion_semantic_parser \
  --cov-report=term-missing \
  --cov-fail-under=75
