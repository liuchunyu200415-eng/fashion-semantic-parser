"""Open-query coarse crop proposals and GT-dependent coverage diagnostics."""

from dataclasses import dataclass

import numpy as np

from fashion_semantic_parser.service.dinov2_region_encoder import (
    DinoV2LetterboxGeometry,
)


@dataclass(frozen=True)
class CoarseCropBox:
    """One source-image crop using exclusive maximum coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


def select_query_peak_crops(
    patch_scores: np.ndarray,
    geometry: DinoV2LetterboxGeometry,
    *,
    crop_fraction: float,
    max_crops: int,
) -> tuple[CoarseCropBox, ...]:
    """Select distinct query-score peaks as category-free source crops.

    Args:
        patch_scores: Finite complete-query score grid shaped ``HxW``.
        geometry: DINOv2 source-image letterbox transform.
        crop_fraction: Crop side relative to the source image maximum dimension.
        max_crops: Maximum number of distinct peak-centered crops.

    Returns:
        One or more deterministic source-image crop boxes.

    Raises:
        ValueError: If scores, crop fraction, count, or geometry are invalid.
    """
    scores = np.asarray(patch_scores, dtype=np.float32)
    if (
        scores.ndim != 2
        or not scores.size
        or not np.all(np.isfinite(scores))
        or not 0.0 < crop_fraction <= 1.0
        or max_crops < 1
    ):
        raise ValueError("Coarse crop scores or settings are invalid.")
    centers = _source_patch_centers(
        (int(scores.shape[0]), int(scores.shape[1])),
        geometry,
    )
    candidates = [
        (float(scores[y, x]), y, x, center)
        for y, row in enumerate(centers)
        for x, center in enumerate(row)
        if center is not None
    ]
    if not candidates:
        raise ValueError("No DINOv2 patch centers fall inside source content.")
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    crop_side = max(
        1,
        round(crop_fraction * max(geometry.original_height, geometry.original_width)),
    )
    boxes: list[CoarseCropBox] = []
    for _, _, _, center in candidates:
        box = _centered_crop(center, crop_side, geometry)
        if any(_contains(existing, center) for existing in boxes):
            continue
        boxes.append(box)
        if len(boxes) == max_crops:
            break
    return tuple(boxes)


def crop_target_coverage(
    target_mask: np.ndarray,
    crops: tuple[CoarseCropBox, ...],
) -> tuple[float, float]:
    """Measure GT target coverage and crop-union image area for diagnosis.

    Args:
        target_mask: Non-empty source-image binary target Mask.
        crops: One or more source-image crop boxes.

    Returns:
        Target-pixel coverage and crop-union image-area fraction.

    Raises:
        ValueError: If target Mask or crop geometry is invalid.
    """
    target = np.asarray(target_mask, dtype=bool)
    if target.ndim != 2 or not target.any() or not crops:
        raise ValueError("Crop coverage requires a non-empty target and crops.")
    union = np.zeros_like(target, dtype=bool)
    for crop in crops:
        if (
            crop.x_min < 0
            or crop.y_min < 0
            or crop.x_max > target.shape[1]
            or crop.y_max > target.shape[0]
            or crop.x_min >= crop.x_max
            or crop.y_min >= crop.y_max
        ):
            raise ValueError("Coarse crop lies outside target image geometry.")
        union[crop.y_min : crop.y_max, crop.x_min : crop.x_max] = True
    target_coverage = np.logical_and(target, union).sum() / target.sum()
    return float(target_coverage), float(union.mean())


def _source_patch_centers(
    grid_shape: tuple[int, int],
    geometry: DinoV2LetterboxGeometry,
) -> list[list[tuple[float, float] | None]]:
    """Map valid letterboxed patch centers into source coordinates."""
    grid_height, grid_width = grid_shape
    rows: list[list[tuple[float, float] | None]] = []
    for y_index in range(grid_height):
        row: list[tuple[float, float] | None] = []
        square_y = (y_index + 0.5) * geometry.output_size / grid_height
        for x_index in range(grid_width):
            square_x = (x_index + 0.5) * geometry.output_size / grid_width
            content_x = square_x - geometry.left
            content_y = square_y - geometry.top
            if not (
                0.0 <= content_x < geometry.resized_width
                and 0.0 <= content_y < geometry.resized_height
            ):
                row.append(None)
                continue
            row.append(
                (
                    content_x * geometry.original_width / geometry.resized_width,
                    content_y * geometry.original_height / geometry.resized_height,
                )
            )
        rows.append(row)
    return rows


def _centered_crop(
    center: tuple[float, float],
    crop_side: int,
    geometry: DinoV2LetterboxGeometry,
) -> CoarseCropBox:
    """Clip a square peak-centered crop to source-image bounds."""
    width = min(crop_side, geometry.original_width)
    height = min(crop_side, geometry.original_height)
    x_min = min(
        geometry.original_width - width,
        max(0, round(center[0] - width / 2)),
    )
    y_min = min(
        geometry.original_height - height,
        max(0, round(center[1] - height / 2)),
    )
    return CoarseCropBox(x_min, y_min, x_min + width, y_min + height)


def _contains(box: CoarseCropBox, point: tuple[float, float]) -> bool:
    """Return whether a source point is already covered by a selected crop."""
    return box.x_min <= point[0] < box.x_max and box.y_min <= point[1] < box.y_max
