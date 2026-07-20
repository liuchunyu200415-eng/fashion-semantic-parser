"""Conservative category-conflict post-processing for garment predictions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredCategoryBox:
    """One scored category prediction with an xyxy box."""

    index: int
    category_id: int
    score: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class CompositeDressConflict:
    """Evidence that one dress duplicates a stronger top and lower-garment pair."""

    dress_index: int
    top_index: int
    lower_index: int
    dress_score: float
    separates_score: float
    union_iou: float
    top_coverage: float
    lower_coverage: float


def find_composite_dress_conflicts(
    predictions: list[ScoredCategoryBox],
    *,
    dress_category_id: int = 5,
    top_category_id: int = 1,
    lower_category_ids: frozenset[int] = frozenset({2, 3}),
    min_union_iou: float = 0.8,
    min_component_coverage: float = 0.8,
    score_margin: float = 0.0,
) -> list[CompositeDressConflict]:
    """Find dress boxes that duplicate a stronger top plus lower-garment pair.

    This only proposes suppression. Callers must validate the policy on a full
    held-out set before enabling it in an inference path.
    """
    _validate_threshold("min_union_iou", min_union_iou)
    _validate_threshold("min_component_coverage", min_component_coverage)
    if score_margin < 0.0:
        raise ValueError("score_margin must be greater than or equal to zero")

    dresses = [
        prediction
        for prediction in predictions
        if prediction.category_id == dress_category_id
    ]
    tops = [
        prediction
        for prediction in predictions
        if prediction.category_id == top_category_id
    ]
    lowers = [
        prediction
        for prediction in predictions
        if prediction.category_id in lower_category_ids
    ]
    conflicts: list[CompositeDressConflict] = []

    for dress in dresses:
        candidates: list[CompositeDressConflict] = []
        for top in tops:
            top_coverage = _intersection_over_component(top, dress)
            if top_coverage < min_component_coverage:
                continue
            for lower in lowers:
                if _center_y(top) >= _center_y(lower):
                    continue
                lower_coverage = _intersection_over_component(lower, dress)
                if lower_coverage < min_component_coverage:
                    continue

                union_box = _union_box(top, lower)
                union_iou = _box_iou(dress, union_box)
                if union_iou < min_union_iou:
                    continue
                separates_score = (top.score + lower.score) / 2.0
                if separates_score < dress.score + score_margin:
                    continue
                candidates.append(
                    CompositeDressConflict(
                        dress_index=dress.index,
                        top_index=top.index,
                        lower_index=lower.index,
                        dress_score=dress.score,
                        separates_score=separates_score,
                        union_iou=union_iou,
                        top_coverage=top_coverage,
                        lower_coverage=lower_coverage,
                    )
                )

        if candidates:
            conflicts.append(
                max(
                    candidates,
                    key=lambda candidate: (
                        candidate.separates_score - candidate.dress_score,
                        candidate.union_iou,
                    ),
                )
            )
    return conflicts


def _validate_threshold(name: str, value: float) -> None:
    """Validate an overlap threshold."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between zero and one")


def _center_y(box: ScoredCategoryBox) -> float:
    """Return the vertical center of one box."""
    return (box.y_min + box.y_max) / 2.0


def _area(box: ScoredCategoryBox) -> float:
    """Return non-negative box area."""
    return max(0.0, box.x_max - box.x_min) * max(0.0, box.y_max - box.y_min)


def _intersection_area(
    first: ScoredCategoryBox,
    second: ScoredCategoryBox,
) -> float:
    """Return intersection area for two xyxy boxes."""
    width = max(0.0, min(first.x_max, second.x_max) - max(first.x_min, second.x_min))
    height = max(0.0, min(first.y_max, second.y_max) - max(first.y_min, second.y_min))
    return width * height


def _intersection_over_component(
    component: ScoredCategoryBox,
    container: ScoredCategoryBox,
) -> float:
    """Return the fraction of a component box covered by a container box."""
    component_area = _area(component)
    if component_area <= 0.0:
        return 0.0
    return _intersection_area(component, container) / component_area


def _box_iou(first: ScoredCategoryBox, second: ScoredCategoryBox) -> float:
    """Return box IoU for two xyxy boxes."""
    intersection = _intersection_area(first, second)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def _union_box(
    first: ScoredCategoryBox,
    second: ScoredCategoryBox,
) -> ScoredCategoryBox:
    """Return the smallest box enclosing two component boxes."""
    return ScoredCategoryBox(
        index=-1,
        category_id=-1,
        score=0.0,
        x_min=min(first.x_min, second.x_min),
        y_min=min(first.y_min, second.y_min),
        x_max=max(first.x_max, second.x_max),
        y_max=max(first.y_max, second.y_max),
    )
