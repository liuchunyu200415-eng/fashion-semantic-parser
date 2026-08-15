"""Tests for DINOv2 Box-guided Mask2Former refinement."""

import pytest

from fashion_semantic_parser.models.localization import (
    LocalizationBoundingBox,
    LocalizedRegion,
    RegionLocalizationPrediction,
)
from fashion_semantic_parser.models.segmentation import (
    SegmentationBoundingBox,
    SegmentationInstance,
    SegmentationPrediction,
)
from fashion_semantic_parser.service.dense_mask2former_refinement import (
    DenseMask2FormerRefinementRegionLocalizationService,
    DenseMask2FormerRefinementSettings,
    localization_polygons_to_mask,
    refine_dense_prediction_with_mask2former,
)
from fashion_semantic_parser.service.region_localization import (
    Mask2FormerPartLocalizationService,
)


class _DenseService:
    """Return one fixed full-query DINOv2 result."""

    def __init__(self, prediction: RegionLocalizationPrediction) -> None:
        self.prediction = prediction
        self.calls: list[tuple[str, str]] = []

    def accepts_query(self, query: str) -> bool:
        """Accept non-empty local-region queries."""
        return bool(query.strip())

    def localize(self, image_path: str, query: str, **_: object):
        """Return the staged DINOv2 prediction."""
        self.calls.append((image_path, query))
        return self.prediction


class _PartService:
    """Return query-compatible supervised candidates."""

    def __init__(
        self,
        prediction: RegionLocalizationPrediction,
        *,
        supported: bool = True,
    ) -> None:
        self.prediction = prediction
        self.supported = supported
        self.calls: list[tuple[str, str]] = []

    def supports_query(self, query: str) -> bool:
        """Expose the staged fixed-part coverage decision."""
        return self.supported

    def localize(self, image_path: str, query: str, **_: object):
        """Return staged Mask2Former candidates."""
        self.calls.append((image_path, query))
        return self.prediction


def test_refinement_selects_overlap_but_preserves_full_query_and_dense_box() -> None:
    """Mask2Former must refine, not replace, full-query DINOv2 selection."""
    query = "衣服右侧带拉链的口袋"
    dense = _prediction(query, [_region("open_query_region", (40, 20, 75, 70))])
    parts = _prediction(
        query,
        [
            _region("pocket", (5, 20, 30, 70), confidence=0.99),
            _region("pocket", (45, 25, 70, 65), confidence=0.70),
        ],
    )

    result = refine_dense_prediction_with_mask2former(
        dense,
        parts,
        minimum_box_iou=0.05,
    )

    assert result.query == query
    assert result.regions[0].box == dense.regions[0].box
    assert result.regions[0].mask == parts.regions[1].mask
    assert result.regions[0].mask_source == "mask2former_box_guided"
    assert result.regions[0].box_source == "dense_coarse_localization"


def test_refinement_falls_back_when_supervised_mask_misses_dense_box() -> None:
    """An unrelated known-part Mask must not overwrite open-query output."""
    dense = _prediction("银色拉链", [_region("open_query_region", (40, 20, 75, 70))])
    parts = _prediction("银色拉链", [_region("zipper", (5, 5, 15, 15))])

    result = refine_dense_prediction_with_mask2former(
        dense,
        parts,
        minimum_box_iou=0.05,
    )

    assert result == dense


def test_refinement_unions_multiple_overlapping_targets() -> None:
    """An unqualified plural target remains one Top-1 union Mask result."""
    dense = _prediction("衣服的袖子", [_region("open_query_region", (10, 10, 90, 80))])
    parts = _prediction(
        "衣服的袖子",
        [
            _region("sleeve", (10, 20, 35, 75)),
            _region("sleeve", (65, 20, 90, 75)),
        ],
    )

    result = refine_dense_prediction_with_mask2former(
        dense,
        parts,
        minimum_box_iou=0.05,
    )

    assert len(result.regions) == 1
    assert len(result.regions[0].mask) == 2


def test_service_skips_mask2former_for_uncovered_open_query() -> None:
    """Unknown parts retain category-free DINOv2 output without fixed mapping."""
    dense_prediction = _prediction(
        "胸前的白色标志",
        [_region("open_query_region", (40, 20, 75, 70))],
    )
    dense_service = _DenseService(dense_prediction)
    part_service = _PartService(dense_prediction, supported=False)
    service = DenseMask2FormerRefinementRegionLocalizationService(
        dense_service,  # type: ignore[arg-type]
        part_service,  # type: ignore[arg-type]
    )

    result = service.localize("data/example.jpg", "胸前的白色标志")

    assert result == dense_prediction
    assert part_service.calls == []


def test_reusable_part_prediction_applies_query_filter_without_reinference() -> None:
    """Batch evaluation can reuse one Mask2Former image prediction safely."""
    segmentation = SegmentationPrediction(
        image_path="data/example.jpg",
        instances=[
            _instance("pocket", (5, 20, 30, 70)),
            _instance("pocket", (45, 25, 70, 65)),
            _instance("zipper", (60, 10, 65, 50)),
        ],
    )
    service = Mask2FormerPartLocalizationService(
        segmentation_service=pytest.fail,  # type: ignore[arg-type]
    )

    result = service.localize_from_segmentation(
        segmentation,
        "衣服右侧的口袋",
    )

    assert len(result.regions) == 1
    assert result.regions[0].region_label == "pocket"
    assert result.regions[0].box.x_min == 45.0


def test_refinement_settings_reject_invalid_overlap_gate() -> None:
    """The geometric replacement threshold must stay a valid IoU."""
    with pytest.raises(ValueError):
        DenseMask2FormerRefinementSettings(minimum_box_iou=1.1)


def test_localization_polygons_rasterize_multiple_instances() -> None:
    """Mask evaluation must retain every selected instance polygon."""
    mask = localization_polygons_to_mask(
        [[1, 1, 3, 1, 3, 3, 1, 3], [6, 6, 8, 6, 8, 8, 6, 8]],
        (10, 10),
    )

    assert mask[2, 2]
    assert mask[7, 7]
    assert not mask[5, 5]


def _prediction(
    query: str,
    regions: list[LocalizedRegion],
) -> RegionLocalizationPrediction:
    """Build one compact localization prediction."""
    return RegionLocalizationPrediction(
        image_path="data/example.jpg",
        query=query,
        regions=regions,
    )


def _region(
    label: str,
    box: tuple[float, float, float, float],
    *,
    confidence: float = 0.8,
) -> LocalizedRegion:
    """Build one rectangular localized region."""
    x_min, y_min, x_max, y_max = box
    return LocalizedRegion(
        region_label=label,
        matched_text=label,
        confidence=confidence,
        box=LocalizationBoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        ),
        mask=[[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]],
        mask_source=(
            "dense_local_reencoding" if label == "open_query_region" else None
        ),
        box_source=(
            "dense_coarse_localization" if label == "open_query_region" else None
        ),
    )


def _instance(
    label: str,
    box: tuple[float, float, float, float],
) -> SegmentationInstance:
    """Build one rectangular supervised part instance."""
    x_min, y_min, x_max, y_max = box
    return SegmentationInstance(
        category_id=1,
        category_label=label,
        confidence=0.8,
        box=SegmentationBoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        ),
        mask=[[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]],
    )
