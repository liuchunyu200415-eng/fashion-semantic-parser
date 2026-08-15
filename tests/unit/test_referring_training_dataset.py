"""Tests for loading referring JSONL records with official source Masks."""

import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest

from fashion_semantic_parser.dao.localization.referring_dataset import (
    FashionpediaReferringDataset,
    collate_referring_training_items,
)


def test_dataset_loads_independent_multi_target_masks(tmp_path: Path) -> None:
    """Broad part expressions must retain each referenced instance Mask."""
    index_path, annotation_path = _write_source(tmp_path)
    dataset = FashionpediaReferringDataset(
        index_path=index_path,
        annotation_path=annotation_path,
        project_root=tmp_path,
        mask_decoder=_polygon_decoder,
    )

    item = dataset[0]

    assert len(dataset) == 1
    assert item.sample.query == "这件衣服的口袋"
    assert item.image_rgb.shape == (12, 16, 3)
    assert item.image_rgb[0, 0].tolist() == [30, 20, 10]
    assert item.target_masks.shape == (2, 12, 16)
    assert np.all(item.target_masks.sum(axis=(1, 2)) > 0)
    assert item.target_boxes.tolist() == [
        [1.0, 2.0, 5.0, 7.0],
        [9.0, 2.0, 14.0, 7.0],
    ]
    assert item.source_annotation_ids == (101, 102)


def test_dataset_metadata_and_area_do_not_load_image_pixels(tmp_path: Path) -> None:
    """Large-run metadata scans must keep full images out of resident memory."""
    index_path, annotation_path = _write_source(tmp_path)
    decode_count = 0

    def counted_decoder(
        segmentation: object,
        height: int,
        width: int,
    ) -> np.ndarray:
        nonlocal decode_count
        decode_count += 1
        return _polygon_decoder(segmentation, height, width)

    dataset = FashionpediaReferringDataset(
        index_path=index_path,
        annotation_path=annotation_path,
        project_root=tmp_path,
        mask_decoder=counted_decoder,
    )
    (tmp_path / "images" / "a.png").unlink()

    assert dataset.sample_at(0).query == "这件衣服的口袋"
    assert dataset.target_union_area_fraction(0) == pytest.approx(66 / 192)
    assert dataset.target_union_area_fraction(0) == pytest.approx(66 / 192)
    assert decode_count == 2
    with pytest.raises(FileNotFoundError, match="Could not read"):
        _ = dataset[0]


def test_dataset_rejects_missing_source_annotation(tmp_path: Path) -> None:
    """A stale annotation reference must fail before a training epoch starts."""
    index_path, annotation_path = _write_source(tmp_path)
    sample = json.loads(index_path.read_text(encoding="utf-8"))
    sample["targets"][0]["source_annotation_id"] = 999
    index_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing Fashionpedia annotation IDs"):
        FashionpediaReferringDataset(
            index_path=index_path,
            annotation_path=annotation_path,
            project_root=tmp_path,
            mask_decoder=_polygon_decoder,
        )


def test_dataset_rejects_annotation_from_another_image(tmp_path: Path) -> None:
    """A Mask must belong to the same source image as the query record."""
    index_path, annotation_path = _write_source(tmp_path)
    source = json.loads(annotation_path.read_text(encoding="utf-8"))
    source["annotations"][0]["image_id"] = 20
    source["images"].append(
        {"id": 20, "file_name": "images/b.png", "width": 16, "height": 12}
    )
    annotation_path.write_text(json.dumps(source), encoding="utf-8")
    dataset = FashionpediaReferringDataset(
        index_path=index_path,
        annotation_path=annotation_path,
        project_root=tmp_path,
        mask_decoder=_polygon_decoder,
    )

    with pytest.raises(ValueError, match="belongs to image 20"):
        _ = dataset[0]


def test_dataset_rejects_empty_decoded_mask(tmp_path: Path) -> None:
    """Bad decoding cannot silently turn a positive target into background."""
    index_path, annotation_path = _write_source(tmp_path)
    dataset = FashionpediaReferringDataset(
        index_path=index_path,
        annotation_path=annotation_path,
        project_root=tmp_path,
        mask_decoder=lambda segmentation, height, width: np.zeros(
            (height, width), dtype=np.uint8
        ),
    )

    with pytest.raises(ValueError, match="empty Mask"):
        _ = dataset[0]


def test_dataset_rejects_source_category_mismatch(tmp_path: Path) -> None:
    """An existing Mask from the wrong category is still invalid supervision."""
    index_path, annotation_path = _write_source(tmp_path)
    source = json.loads(annotation_path.read_text(encoding="utf-8"))
    source["categories"].append({"id": 35, "name": "zipper"})
    source["annotations"][0]["category_id"] = 35
    annotation_path.write_text(json.dumps(source), encoding="utf-8")
    dataset = FashionpediaReferringDataset(
        index_path=index_path,
        annotation_path=annotation_path,
        project_root=tmp_path,
        mask_decoder=_polygon_decoder,
    )

    with pytest.raises(ValueError, match="has label zipper, not pocket"):
        _ = dataset[0]


def test_dataset_rejects_source_box_mismatch(tmp_path: Path) -> None:
    """A stale target box cannot accompany an otherwise valid source Mask."""
    index_path, annotation_path = _write_source(tmp_path)
    sample = json.loads(index_path.read_text(encoding="utf-8"))
    sample["targets"][0]["box"]["x_max"] = 6
    index_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    dataset = FashionpediaReferringDataset(
        index_path=index_path,
        annotation_path=annotation_path,
        project_root=tmp_path,
        mask_decoder=_polygon_decoder,
    )

    with pytest.raises(ValueError, match="bbox does not match"):
        _ = dataset[0]


def test_collate_preserves_variable_target_sets(tmp_path: Path) -> None:
    """Batch collation must not pad or merge independent query targets."""
    index_path, annotation_path = _write_source(tmp_path)
    dataset = FashionpediaReferringDataset(
        index_path=index_path,
        annotation_path=annotation_path,
        project_root=tmp_path,
        mask_decoder=_polygon_decoder,
    )

    batch = collate_referring_training_items([dataset[0], dataset[-1]])

    assert batch["queries"] == ["这件衣服的口袋", "这件衣服的口袋"]
    assert [masks.shape[0] for masks in batch["target_masks"]] == [2, 2]
    assert batch["source_annotation_ids"] == [(101, 102), (101, 102)]


def test_dataset_rejects_blank_jsonl_record(tmp_path: Path) -> None:
    """Blank records must not change sample numbering silently."""
    index_path, annotation_path = _write_source(tmp_path)
    index_path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Blank JSONL record"):
        FashionpediaReferringDataset(
            index_path=index_path,
            annotation_path=annotation_path,
            project_root=tmp_path,
            mask_decoder=_polygon_decoder,
        )


def test_dataset_image_limit_keeps_complete_query_groups(tmp_path: Path) -> None:
    """An image bound cannot truncate expressions belonging to its last image."""
    index_path, annotation_path = _write_source(tmp_path)
    first_sample = json.loads(index_path.read_text(encoding="utf-8"))
    second_first_image_sample = dict(first_sample)
    second_first_image_sample["id"] = "fashionpedia-train-10-pocket-basic-en-101-102"
    second_first_image_sample["query"] = "the pockets on the garment"
    second_first_image_sample["language"] = "en"
    second_first_image_sample["template_id"] = "basic-en"
    second_image_sample = _add_source_image(
        tmp_path=tmp_path,
        annotation_path=annotation_path,
        sample=first_sample,
        image_id=20,
        annotation_ids=(201, 202),
        file_name="images/b.png",
    )
    third_image_sample = _add_source_image(
        tmp_path=tmp_path,
        annotation_path=annotation_path,
        sample=first_sample,
        image_id=30,
        annotation_ids=(301, 302),
        file_name="images/c.png",
    )
    index_path.write_text(
        "".join(
            json.dumps(sample) + "\n"
            for sample in (
                first_sample,
                second_first_image_sample,
                second_image_sample,
                third_image_sample,
            )
        ),
        encoding="utf-8",
    )

    dataset = FashionpediaReferringDataset(
        index_path=index_path,
        annotation_path=annotation_path,
        project_root=tmp_path,
        max_images=2,
        mask_decoder=_polygon_decoder,
    )

    assert len(dataset) == 3
    assert [dataset[index].sample.source_image_id for index in range(3)] == [10, 10, 20]

    offset_dataset = FashionpediaReferringDataset(
        index_path=index_path,
        annotation_path=annotation_path,
        project_root=tmp_path,
        max_images=1,
        image_offset=1,
        mask_decoder=_polygon_decoder,
    )

    assert len(offset_dataset) == 1
    assert offset_dataset[0].sample.source_image_id == 20


def test_dataset_rejects_simultaneous_sample_and_image_limits(tmp_path: Path) -> None:
    """Callers must choose either a query prefix or an image-complete prefix."""
    index_path, annotation_path = _write_source(tmp_path)

    with pytest.raises(ValueError, match="mutually exclusive"):
        FashionpediaReferringDataset(
            index_path=index_path,
            annotation_path=annotation_path,
            project_root=tmp_path,
            max_samples=1,
            max_images=1,
            mask_decoder=_polygon_decoder,
        )


def test_dataset_image_offset_requires_image_limit(tmp_path: Path) -> None:
    """An offset cannot be combined with an ambiguous query-count boundary."""
    index_path, annotation_path = _write_source(tmp_path)

    with pytest.raises(ValueError, match="requires max_images"):
        FashionpediaReferringDataset(
            index_path=index_path,
            annotation_path=annotation_path,
            project_root=tmp_path,
            image_offset=1,
            mask_decoder=_polygon_decoder,
        )


def _write_source(tmp_path: Path) -> tuple[Path, Path]:
    """Write one two-target source image, annotation file, and query record."""
    image_path = tmp_path / "images" / "a.png"
    image_path.parent.mkdir(parents=True)
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :] = [10, 20, 30]
    assert cv2.imwrite(str(image_path), image)

    annotation_path = tmp_path / "annotations.json"
    source = {
        "categories": [{"id": 32, "name": "pocket"}],
        "images": [{"id": 10, "file_name": "images/a.png", "width": 16, "height": 12}],
        "annotations": [
            {
                "id": 101,
                "image_id": 10,
                "category_id": 32,
                "bbox": [1, 2, 4, 5],
                "segmentation": [[1, 2, 5, 2, 5, 7, 1, 7]],
            },
            {
                "id": 102,
                "image_id": 10,
                "category_id": 32,
                "bbox": [9, 2, 5, 5],
                "segmentation": [[9, 2, 14, 2, 14, 7, 9, 7]],
            },
        ],
    }
    annotation_path.write_text(json.dumps(source), encoding="utf-8")

    sample = {
        "id": "fashionpedia-train-10-pocket-basic-zh-101-102",
        "split": "train",
        "image_path": "images/a.png",
        "source_image_id": 10,
        "query": "这件衣服的口袋",
        "language": "zh",
        "dimensions": ["basic"],
        "target_label": "pocket",
        "targets": [
            {
                "source_annotation_id": 101,
                "label": "pocket",
                "box": {"x_min": 1, "y_min": 2, "x_max": 5, "y_max": 7},
            },
            {
                "source_annotation_id": 102,
                "label": "pocket",
                "box": {"x_min": 9, "y_min": 2, "x_max": 14, "y_max": 7},
            },
        ],
        "template_id": "basic-zh",
    }
    index_path = tmp_path / "referring.jsonl"
    index_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    return index_path, annotation_path


# pylint: disable-next=too-many-arguments
def _add_source_image(
    *,
    tmp_path: Path,
    annotation_path: Path,
    sample: dict[str, Any],
    image_id: int,
    annotation_ids: tuple[int, int],
    file_name: str,
) -> dict[str, Any]:
    """Append one equivalent source image and return its referring sample."""
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / file_name), image)
    source = json.loads(annotation_path.read_text(encoding="utf-8"))
    source["images"].append(
        {"id": image_id, "file_name": file_name, "width": 16, "height": 12}
    )
    for source_annotation, annotation_id in zip(
        source["annotations"][:2], annotation_ids
    ):
        annotation = dict(source_annotation)
        annotation["id"] = annotation_id
        annotation["image_id"] = image_id
        source["annotations"].append(annotation)
    annotation_path.write_text(json.dumps(source), encoding="utf-8")

    result = cast(dict[str, Any], json.loads(json.dumps(sample)))
    result["id"] = f"fashionpedia-train-{image_id}-pocket-basic-zh"
    result["source_image_id"] = image_id
    result["image_path"] = file_name
    for target, annotation_id in zip(result["targets"], annotation_ids):
        target["source_annotation_id"] = annotation_id
    return result


def _polygon_decoder(segmentation: object, height: int, width: int) -> np.ndarray:
    """Small test decoder; production decoding remains pycocotools-backed."""
    mask = np.zeros((height, width), dtype=np.uint8)
    assert isinstance(segmentation, list)
    polygons = [
        np.asarray(polygon, dtype=np.int32).reshape(-1, 2) for polygon in segmentation
    ]
    cv2.fillPoly(mask, polygons, 1)
    return mask
