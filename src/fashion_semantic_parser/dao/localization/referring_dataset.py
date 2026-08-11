"""Load Fashionpedia referring samples and their official source Masks."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import cv2
import numpy as np

from fashion_semantic_parser.dao.fashionpedia import (
    category_records_by_id,
    dict_records,
    is_integer,
    is_positive_number,
    normalize_coco_bbox_xywh,
    normalize_coco_segmentation,
    read_fashionpedia_json,
    source_category_name,
)
from fashion_semantic_parser.dao.localization.referring_training import (
    ReferringTrainingSample,
)
from fashion_semantic_parser.dao.localization.taxonomy import (
    map_fashionpedia_part_category,
)

MaskDecoder = Callable[[object, int, int], np.ndarray]


@dataclass(frozen=True)
class ReferringTrainingItem:
    """One complete-query training item with independent target Masks."""

    sample: ReferringTrainingSample
    image_rgb: np.ndarray
    target_masks: np.ndarray
    target_boxes: np.ndarray
    source_annotation_ids: tuple[int, ...]


class FashionpediaReferringDataset:
    """Random-access JSONL dataset backed by official Fashionpedia Masks.

    The class intentionally does not inherit from PyTorch so dataset inspection
    remains lightweight. It implements the map-style ``__len__``/``__getitem__``
    contract consumed by ``torch.utils.data.DataLoader``.
    """

    def __init__(
        self,
        *,
        index_path: Path,
        annotation_path: Path,
        project_root: Path,
        max_samples: int | None = None,
        mask_decoder: MaskDecoder | None = None,
    ) -> None:
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be greater than or equal to zero")
        self.index_path = Path(index_path)
        self.annotation_path = Path(annotation_path)
        self.project_root = Path(project_root)
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Referring index does not exist: {index_path}")
        if not self.annotation_path.is_file():
            raise FileNotFoundError(
                f"Fashionpedia annotations do not exist: {annotation_path}"
            )
        if not self.project_root.is_dir():
            raise NotADirectoryError(f"Project root does not exist: {project_root}")

        self._mask_decoder = mask_decoder or decode_coco_mask
        self._offsets, referenced_annotation_ids, referenced_image_ids = (
            _scan_jsonl_index(self.index_path, max_samples=max_samples)
        )
        source = read_fashionpedia_json(self.annotation_path)
        category_by_id = category_records_by_id(dict_records(source.get("categories")))
        self._annotations = _index_referenced_annotations(
            dict_records(source.get("annotations")),
            referenced_annotation_ids,
            category_by_id,
        )
        self._image_sizes = _index_referenced_image_sizes(
            dict_records(source.get("images")),
            referenced_image_ids,
        )
        del source

        missing_annotations = sorted(
            referenced_annotation_ids.difference(self._annotations)
        )
        if missing_annotations:
            raise ValueError(
                "Referring index references missing Fashionpedia annotation IDs: "
                f"{missing_annotations[:10]}"
            )
        missing_images = sorted(referenced_image_ids.difference(self._image_sizes))
        if missing_images:
            raise ValueError(
                "Referring index references missing Fashionpedia image IDs: "
                f"{missing_images[:10]}"
            )

    def __len__(self) -> int:
        """Return the bounded number of indexed query samples."""
        return len(self._offsets)

    def __getitem__(self, index: int) -> ReferringTrainingItem:
        """Load one RGB image and decode every referenced target Mask."""
        normalized_index = _normalize_sequence_index(index, len(self))
        sample = self._read_sample(self._offsets[normalized_index])
        image_path = _resolve_safe_project_path(self.project_root, sample.image_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read referring image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        height, width = self._image_sizes[sample.source_image_id]
        if image_rgb.shape[:2] != (height, width):
            raise ValueError(
                "Image dimensions do not match Fashionpedia metadata for "
                f"image {sample.source_image_id}: decoded={image_rgb.shape[:2]} "
                f"metadata={(height, width)}"
            )

        masks: list[np.ndarray] = []
        annotation_ids: list[int] = []
        for target in sample.targets:
            annotation = self._annotations[target.source_annotation_id]
            if annotation["image_id"] != sample.source_image_id:
                raise ValueError(
                    f"Annotation {target.source_annotation_id} belongs to image "
                    f"{annotation['image_id']}, not {sample.source_image_id}."
                )
            if annotation["label"] != target.label:
                raise ValueError(
                    f"Annotation {target.source_annotation_id} has label "
                    f"{annotation['label']}, not {target.label}."
                )
            target_box = np.asarray(
                [
                    target.box.x_min,
                    target.box.y_min,
                    target.box.x_max,
                    target.box.y_max,
                ],
                dtype=np.float64,
            )
            if not np.allclose(annotation["box"], target_box, atol=1e-6):
                raise ValueError(
                    f"Annotation {target.source_annotation_id} bbox does not match "
                    "the referring index."
                )
            mask = np.asarray(
                self._mask_decoder(annotation["segmentation"], height, width)
            )
            if mask.shape != (height, width):
                raise ValueError(
                    f"Annotation {target.source_annotation_id} decoded to "
                    f"{mask.shape}, expected {(height, width)}."
                )
            binary_mask = np.asarray(mask != 0, dtype=np.uint8)
            if not np.any(binary_mask):
                raise ValueError(
                    f"Annotation {target.source_annotation_id} decoded to an "
                    "empty Mask."
                )
            masks.append(binary_mask)
            annotation_ids.append(target.source_annotation_id)

        boxes = np.asarray(
            [
                [
                    target.box.x_min,
                    target.box.y_min,
                    target.box.x_max,
                    target.box.y_max,
                ]
                for target in sample.targets
            ],
            dtype=np.float32,
        )
        return ReferringTrainingItem(
            sample=sample,
            image_rgb=image_rgb,
            target_masks=np.stack(masks, axis=0),
            target_boxes=boxes,
            source_annotation_ids=tuple(annotation_ids),
        )

    def _read_sample(self, offset: int) -> ReferringTrainingSample:
        """Read and validate one JSONL record at a byte offset."""
        with self.index_path.open("rb") as index_file:
            index_file.seek(offset)
            line = index_file.readline()
        try:
            raw_sample = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid JSONL record at byte offset {offset} in {self.index_path}."
            ) from error
        return ReferringTrainingSample.model_validate(raw_sample)


def collate_referring_training_items(
    items: list[ReferringTrainingItem],
) -> dict[str, Any]:
    """Preserve variable image sizes and target counts in one training batch."""
    if not items:
        raise ValueError("Cannot collate an empty referring training batch.")
    return {
        "samples": [item.sample for item in items],
        "queries": [item.sample.query for item in items],
        "languages": [item.sample.language for item in items],
        "dimensions": [tuple(item.sample.dimensions) for item in items],
        "images_rgb": [item.image_rgb for item in items],
        "target_masks": [item.target_masks for item in items],
        "target_boxes": [item.target_boxes for item in items],
        "source_annotation_ids": [item.source_annotation_ids for item in items],
    }


def decode_coco_mask(segmentation: object, height: int, width: int) -> np.ndarray:
    """Decode a normalized polygon/RLE with COCO-compatible edge semantics."""
    normalized = normalize_coco_segmentation(segmentation)
    if normalized is None:
        raise ValueError("Cannot decode an invalid COCO segmentation.")
    try:
        from pycocotools import mask as mask_utils  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError(
            "pycocotools is required to decode official Fashionpedia Masks."
        ) from error

    if isinstance(normalized, list):
        rles = mask_utils.frPyObjects(normalized, height, width)
        encoded = mask_utils.merge(rles)
    elif isinstance(normalized.get("counts"), list):
        encoded = mask_utils.frPyObjects(normalized, height, width)
    else:
        encoded = normalized
    decoded = np.asarray(mask_utils.decode(encoded))
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    binary_mask: np.ndarray = np.asarray(decoded != 0, dtype=np.uint8)
    return binary_mask


def _scan_jsonl_index(
    index_path: Path,
    *,
    max_samples: int | None,
) -> tuple[list[int], set[int], set[int]]:
    """Collect byte offsets plus referenced source IDs in one bounded scan."""
    offsets: list[int] = []
    annotation_ids: set[int] = set()
    image_ids: set[int] = set()
    with index_path.open("rb") as index_file:
        while max_samples is None or len(offsets) < max_samples:
            offset = index_file.tell()
            line = index_file.readline()
            if not line:
                break
            if not line.strip():
                raise ValueError(
                    f"Blank JSONL record at byte offset {offset} in {index_path}."
                )
            try:
                sample = ReferringTrainingSample.model_validate_json(line)
            except Exception as error:
                raise ValueError(
                    f"Invalid referring record at byte offset {offset} in {index_path}."
                ) from error
            offsets.append(offset)
            image_ids.add(sample.source_image_id)
            annotation_ids.update(
                target.source_annotation_id for target in sample.targets
            )
    return offsets, annotation_ids, image_ids


def _index_referenced_annotations(
    annotations: list[dict[str, Any]],
    referenced_ids: set[int],
    category_by_id: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Keep only source Masks referenced by the selected JSONL records."""
    indexed: dict[int, dict[str, Any]] = {}
    for annotation in annotations:
        annotation_id = annotation.get("id")
        if not is_integer(annotation_id) or annotation_id not in referenced_ids:
            continue
        image_id = annotation.get("image_id")
        bbox = normalize_coco_bbox_xywh(annotation.get("bbox"))
        segmentation = normalize_coco_segmentation(annotation.get("segmentation"))
        source_name = source_category_name(annotation, category_by_id)
        category = map_fashionpedia_part_category(source_name)
        if (
            not is_integer(image_id)
            or bbox is None
            or segmentation is None
            or category is None
            or annotation.get("iscrowd") == 1
        ):
            raise ValueError(
                f"Referenced Fashionpedia annotation {annotation_id} is invalid."
            )
        if annotation_id in indexed:
            raise ValueError(f"Duplicate Fashionpedia annotation ID: {annotation_id}")
        indexed[annotation_id] = {
            "image_id": image_id,
            "segmentation": segmentation,
            "label": category.english_name,
            "box": np.asarray(
                [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                dtype=np.float64,
            ),
        }
    return indexed


def _index_referenced_image_sizes(
    images: list[dict[str, Any]],
    referenced_ids: set[int],
) -> dict[int, tuple[int, int]]:
    """Index validated source image dimensions as ``(height, width)``."""
    indexed: dict[int, tuple[int, int]] = {}
    for image in images:
        image_id = image.get("id")
        if not is_integer(image_id) or image_id not in referenced_ids:
            continue
        width = image.get("width")
        height = image.get("height")
        if not is_positive_number(width) or not is_positive_number(height):
            raise ValueError(f"Fashionpedia image {image_id} has invalid dimensions.")
        if image_id in indexed:
            raise ValueError(f"Duplicate Fashionpedia image ID: {image_id}")
        indexed[image_id] = (
            int(cast(int | float, height)),
            int(cast(int | float, width)),
        )
    return indexed


def _resolve_safe_project_path(project_root: Path, value: str) -> Path:
    """Resolve one project-relative image path without traversal."""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe project-relative image path: {value}")
    return project_root / relative


def _normalize_sequence_index(index: int, length: int) -> int:
    """Support standard positive and negative sequence indices."""
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("Referring dataset indices must be integers.")
    normalized = index + length if index < 0 else index
    if normalized < 0 or normalized >= length:
        raise IndexError("Referring dataset index out of range.")
    return normalized
