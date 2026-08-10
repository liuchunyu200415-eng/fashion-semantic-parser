"""Tests for selecting the referring smoke cases from Fashionpedia."""

import json
from pathlib import Path

import pytest

from scripts.prepare_referring_smoke_fashionpedia import (
    prepare_referring_smoke_manifest,
)


def test_prepare_manifest_selects_images_without_inventing_cuff_gt(
    tmp_path: Path,
) -> None:
    """Direct masks may be imported, but sleeves must not become cuff masks."""
    annotation_path, image_root = _write_complete_fixture(tmp_path)
    template_path = (
        Path(__file__).resolve().parents[2]
        / "data/benchmarks/localization/referring_smoke_v1.template.json"
    )

    manifest, summary = prepare_referring_smoke_manifest(
        annotations_path=annotation_path,
        image_root=image_root,
        template_path=template_path,
    )

    cases = {case["id"]: case for case in manifest["cases"]}
    assert len(cases) == 20
    assert summary["available_annotated_image_count"] == 1
    assert summary["fashionpedia_gt_imported_count"] == 4
    assert summary["manual_review_required_count"] == 16
    assert summary["accuracy_ready"] is False
    assert all(case["image_path"].endswith("all.jpg") for case in cases.values())

    assert cases["basic_collar_001"]["annotation_status"] == "mask"
    assert cases["basic_collar_001"]["targets"][0]["label"] == "neckline"
    assert cases["basic_zipper_001"]["expected_count"] == 2

    right_pocket = cases["spatial_right_pocket_001"]["targets"][0]
    assert right_pocket["box"]["x_min"] == 60.0
    lower_zipper = cases["spatial_lower_zipper_001"]["targets"][0]
    assert lower_zipper["box"]["y_min"] == 70.0

    for case_id in (
        "basic_cuffs_001",
        "spatial_left_cuff_001",
        "spatial_right_cuff_001",
    ):
        assert cases[case_id]["annotation_status"] == "unlabelled"
        assert cases[case_id]["targets"] == []
        assert "expected_count" not in cases[case_id]


def test_prepare_manifest_rejects_missing_image_archive(tmp_path: Path) -> None:
    """Annotations alone must not produce a manifest with broken image paths."""
    annotation_path, image_root = _write_complete_fixture(tmp_path)
    (image_root / "all.jpg").unlink()
    template_path = (
        Path(__file__).resolve().parents[2]
        / "data/benchmarks/localization/referring_smoke_v1.template.json"
    )

    with pytest.raises(FileNotFoundError, match="No annotated Fashionpedia images"):
        prepare_referring_smoke_manifest(
            annotations_path=annotation_path,
            image_root=image_root,
            template_path=template_path,
        )


def _write_complete_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write one image containing every category combination used by the selector."""
    root = tmp_path / "fashionpedia"
    annotation_path = root / "annotations" / "instances_attributes_val2020.json"
    image_root = root / "test"
    annotation_path.parent.mkdir(parents=True)
    image_root.mkdir(parents=True)
    (image_root / "all.jpg").write_bytes(b"fixture")

    category_names = [
        "neckline",
        "sleeve",
        "zipper",
        "rivet",
        "jacket",
        "hood",
        "pocket",
        "flower",
        "collar",
        "applique",
        "top, t-shirt, sweatshirt",
        "bead",
    ]
    categories = [
        {"id": category_id, "name": name}
        for category_id, name in enumerate(category_names)
    ]
    category_id = {category["name"]: category["id"] for category in categories}
    annotations = []

    def add(name: str, x: float, y: float) -> None:
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": 1,
                "category_id": category_id[name],
                "bbox": [x, y, 10, 10],
                "segmentation": [[x, y, x + 10, y, x + 10, y + 10]],
            }
        )

    add("neckline", 40, 10)
    add("sleeve", 10, 20)
    add("sleeve", 80, 20)
    add("zipper", 45, 20)
    add("zipper", 45, 70)
    add("rivet", 45, 30)
    add("rivet", 45, 40)
    add("jacket", 10, 10)
    add("hood", 35, 0)
    add("pocket", 20, 60)
    add("pocket", 60, 60)
    add("flower", 15, 35)
    add("collar", 35, 10)
    add("applique", 40, 30)
    add("top, t-shirt, sweatshirt", 20, 20)
    add("bead", 40, 15)

    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "all.jpg",
                        "width": 100,
                        "height": 100,
                    }
                ],
                "categories": categories,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )
    return annotation_path, image_root
