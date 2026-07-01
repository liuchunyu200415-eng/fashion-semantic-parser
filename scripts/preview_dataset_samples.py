"""Preview normalized samples from configured datasets."""

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
    """Print a small JSON preview of normalized dataset samples."""
    add_src_to_python_path()

    from fashion_semantic_parser.common.paths import resolve_project_path
    from fashion_semantic_parser.config import load_settings
    from fashion_semantic_parser.dao.datasets.deepfashion2 import (
        load_deepfashion2_samples,
    )
    from fashion_semantic_parser.dao.datasets.fashionai import (
        load_fashionai_attribute_samples,
        load_fashionai_questions,
    )

    settings = load_settings()
    fashionai_root = resolve_project_path(settings.datasets.fashionai_root)
    deepfashion2_root = resolve_project_path(settings.datasets.deepfashion2_root)
    preview = {
        "fashionai_samples": [
            sample.model_dump()
            for sample in load_fashionai_attribute_samples(fashionai_root, limit=3)
        ],
        "fashionai_questions": [
            question.model_dump()
            for question in load_fashionai_questions(fashionai_root, limit=3)
        ],
        "deepfashion2_train_samples": [
            sample.model_dump()
            for sample in load_deepfashion2_samples(
                deepfashion2_root,
                split="train",
                limit=3,
            )
        ],
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
