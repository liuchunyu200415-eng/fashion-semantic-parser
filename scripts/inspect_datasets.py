"""Inspect configured FashionAI and DeepFashion2 datasets."""

import json
import sys
from pathlib import Path


def add_src_to_python_path() -> None:
    """Add the local src directory when the package is not installed yet."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    """Print a JSON summary of configured datasets."""
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.datasets.summary import (
        inspect_project_datasets,
    )

    settings = load_settings()
    summary = inspect_project_datasets(
        resolve_project_path(settings.datasets.fashionai_root),
        resolve_project_path(settings.datasets.deepfashion2_root),
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
