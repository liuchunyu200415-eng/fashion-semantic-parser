"""Tests for conservative segmentation category-conflict handling."""

import pytest

from fashion_semantic_parser.service.segmentation_postprocessing import (
    ScoredCategoryBox,
    find_composite_dress_conflicts,
)


def _box(
    index: int,
    category_id: int,
    score: float,
    coordinates: tuple[float, float, float, float],
) -> ScoredCategoryBox:
    return ScoredCategoryBox(
        index=index,
        category_id=category_id,
        score=score,
        x_min=coordinates[0],
        y_min=coordinates[1],
        x_max=coordinates[2],
        y_max=coordinates[3],
    )


def test_composite_dress_is_suppressed_by_stronger_top_and_skirt() -> None:
    """A dress duplicating the union of stronger separates should be diagnosed."""
    predictions = [
        _box(0, 5, 0.828, (0.0, 0.0, 100.0, 200.0)),
        _box(1, 1, 0.925, (0.0, 0.0, 100.0, 90.0)),
        _box(2, 3, 0.824, (0.0, 90.0, 100.0, 200.0)),
    ]

    conflicts = find_composite_dress_conflicts(predictions)

    assert len(conflicts) == 1
    assert conflicts[0].dress_index == 0
    assert conflicts[0].top_index == 1
    assert conflicts[0].lower_index == 2
    assert conflicts[0].separates_score > conflicts[0].dress_score
    assert conflicts[0].union_iou == 1.0


def test_dress_is_retained_without_complete_separates_pair() -> None:
    """One overlapping top alone is insufficient evidence for suppression."""
    predictions = [
        _box(0, 5, 0.85, (0.0, 0.0, 100.0, 200.0)),
        _box(1, 1, 0.95, (0.0, 0.0, 100.0, 90.0)),
    ]

    assert find_composite_dress_conflicts(predictions) == []


def test_stronger_dress_is_retained() -> None:
    """Lower-confidence separates must not remove a stronger dress prediction."""
    predictions = [
        _box(0, 5, 0.95, (0.0, 0.0, 100.0, 200.0)),
        _box(1, 1, 0.86, (0.0, 0.0, 100.0, 90.0)),
        _box(2, 2, 0.84, (0.0, 90.0, 100.0, 200.0)),
    ]

    assert find_composite_dress_conflicts(predictions) == []


def test_spatially_unrelated_separates_do_not_suppress_dress() -> None:
    """High scores cannot compensate for poor component coverage and union IoU."""
    predictions = [
        _box(0, 5, 0.81, (0.0, 0.0, 100.0, 200.0)),
        _box(1, 1, 0.95, (200.0, 0.0, 300.0, 90.0)),
        _box(2, 3, 0.94, (200.0, 90.0, 300.0, 200.0)),
    ]

    assert find_composite_dress_conflicts(predictions) == []


def test_conflict_thresholds_are_validated() -> None:
    """Invalid experiment settings should fail before processing predictions."""
    with pytest.raises(ValueError, match="min_union_iou"):
        find_composite_dress_conflicts([], min_union_iou=1.1)
    with pytest.raises(ValueError, match="score_margin"):
        find_composite_dress_conflicts([], score_margin=-0.1)
