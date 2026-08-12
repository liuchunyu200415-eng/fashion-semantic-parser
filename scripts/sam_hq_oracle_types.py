"""Typed records shared by SAM-HQ oracle diagnostic scripts."""

from typing import Any, Protocol, TypedDict

import numpy as np


class TargetRow(TypedDict):
    """One unique Fashionpedia target used by an oracle diagnostic."""

    annotation_id: int
    label: str
    mask: np.ndarray
    box: tuple[float, float, float, float]


class ImageGroup(TypedDict):
    """One decoded image and its unique annotation targets."""

    image_rgb: np.ndarray | None
    targets: dict[int, TargetRow]


class CaseRow(TypedDict):
    """One persisted target-level oracle refinement result."""

    source_image_id: int
    source_annotation_id: int
    target_label: str
    target_area_pixels: int
    target_area_ratio: float
    prompt_box: tuple[float, float, float, float]
    mask_box: tuple[float, float, float, float] | None
    mask_quality: float
    mask_iou: float
    box_expansion_ratio: float
    candidate_count: int
    score_selected_candidate_index: int
    oracle_best_candidate_index: int
    oracle_best_mask_iou: float
    roi_crop_scale: float
    crop_box: tuple[int, int, int, int] | None
    positive_point: tuple[float, float] | None


class ImageRow(TypedDict):
    """One persisted image-level runtime result."""

    source_image_id: int
    target_count: int
    prompt_count: int
    elapsed_seconds: float


class PromptRow(TypedDict):
    """One model prompt and the matching evaluation Mask coordinates."""

    target: TargetRow
    box_expansion_ratio: float
    prompt_box: tuple[float, float, float, float]
    evaluation_mask: np.ndarray
    crop_box: tuple[int, int, int, int] | None
    positive_point: tuple[float, float] | None


class DatasetProtocol(Protocol):
    """Minimal map-style dataset contract used by a diagnostic."""

    def __len__(self) -> int:
        """Return the selected query count."""
        ...

    def __getitem__(self, index: int) -> Any:
        """Return one loaded referring item."""
        ...
