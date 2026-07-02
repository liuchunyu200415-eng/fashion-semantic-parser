"""Tests for PRD 3.1.1 segmentation dataset preparation."""

import json
from pathlib import Path

from fashion_semantic_parser.dao.segmentation.coco import (
    convert_deepfashion2_to_coco,
)
from fashion_semantic_parser.dao.segmentation.taxonomy import (
    map_deepfashion2_category,
)


def test_map_deepfashion2_category_to_prd_taxonomy() -> None:
    """DeepFashion2 categories should map to PRD segmentation categories."""
    top = map_deepfashion2_category("short sleeve top")
    pants = map_deepfashion2_category("trousers")
    dress = map_deepfashion2_category("sling dress")

    assert top is not None
    assert top.english_name == "top"
    assert pants is not None
    assert pants.english_name == "pants"
    assert dress is not None
    assert dress.english_name == "dress"
    assert map_deepfashion2_category("unknown") is None


def test_convert_deepfashion2_to_coco_writes_segmentation_json(
    tmp_path: Path,
) -> None:
    """DeepFashion2 conversion should write compact COCO segmentation JSON."""
    root = tmp_path / "deepfashion2"
    image_root = root / "train" / "image"
    annotation_root = root / "train" / "annos"
    output_path = tmp_path / "processed" / "deepfashion2_train.json"
    image_root.mkdir(parents=True)
    annotation_root.mkdir(parents=True)
    (image_root / "000001.jpg").write_bytes(b"fake-image")
    annotation = {
        "item1": {
            "category_name": "short sleeve top",
            "bounding_box": [10, 20, 110, 220],
            "segmentation": [[10, 20, 110, 20, 110, 220, 10, 220]],
        },
        "item2": {
            "category_name": "trousers",
            "bounding_box": [1, 2, 3, 4],
        },
    }
    (annotation_root / "000001.json").write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )

    summary = convert_deepfashion2_to_coco(
        root=root,
        split="train",
        output_path=output_path,
    )
    coco = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary.image_count == 1
    assert summary.annotation_count == 1
    assert summary.skipped_item_count == 1
    assert coco["images"][0]["file_name"].endswith("train/image/000001.jpg")
    assert coco["annotations"][0]["category_id"] == 1
    assert coco["annotations"][0]["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert len(coco["categories"]) == 8
