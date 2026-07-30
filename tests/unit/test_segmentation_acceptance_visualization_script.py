"""Tests for segmentation visual acceptance helpers."""

import numpy as np

from scripts.visualize_segmentation_acceptance import (
    _build_contact_sheet,
    _filter_predictions,
    _readable_selection_reasons,
    _select_acceptance_images,
)


def test_acceptance_selection_includes_samples_and_category_misses() -> None:
    """Acceptance images should expose normal examples and known misses."""
    ground_truth = {
        1: [{"category_id": 1}],
        2: [{"category_id": 1}],
        3: [{"category_id": 2}],
        4: [{"category_id": 2}],
    }
    predictions = {
        1: [{"category_id": 1}],
        2: [],
        3: [{"category_id": 2}],
        4: [],
    }

    image_ids, reasons = _select_acceptance_images(
        ground_truth_by_image=ground_truth,
        predictions_by_image=predictions,
        category_ids=[1, 2],
        samples_per_category=1,
        misses_per_category=1,
        seed=311,
    )

    assert 2 in image_ids
    assert 4 in image_ids
    assert "miss:category_1" in reasons[2]
    assert "miss:category_2" in reasons[4]
    assert any(
        reason == "sample:category_1"
        for values in reasons.values()
        for reason in values
    )
    assert any(
        reason == "sample:category_2"
        for values in reasons.values()
        for reason in values
    )
    assert all(
        not (
            "sample:category_1" in image_reasons and "miss:category_1" in image_reasons
        )
        for image_reasons in reasons.values()
    )
    assert all(
        not (
            "sample:category_2" in image_reasons and "miss:category_2" in image_reasons
        )
        for image_reasons in reasons.values()
    )


def test_acceptance_samples_require_the_category_prediction() -> None:
    """Normal samples should be hits while misses remain explicit failures."""
    ground_truth = {
        1: [{"category_id": 1}],
        2: [{"category_id": 1}],
    }
    predictions = {
        1: [],
        2: [{"category_id": 1}],
    }

    image_ids, reasons = _select_acceptance_images(
        ground_truth_by_image=ground_truth,
        predictions_by_image=predictions,
        category_ids=[1],
        samples_per_category=1,
        misses_per_category=1,
        seed=311,
    )

    assert image_ids == [2, 1]
    assert reasons[2] == ["sample:category_1"]
    assert reasons[1] == ["miss:category_1"]


def test_acceptance_selection_is_deterministic() -> None:
    """A fixed seed should preserve the review sample across runs."""
    ground_truth = {image_id: [{"category_id": 1}] for image_id in range(1, 8)}
    predictions = {image_id: [{"category_id": 1}] for image_id in ground_truth}

    first = _select_acceptance_images(
        ground_truth,
        predictions,
        category_ids=[1],
        samples_per_category=3,
        misses_per_category=0,
        seed=311,
    )
    second = _select_acceptance_images(
        ground_truth,
        predictions,
        category_ids=[1],
        samples_per_category=3,
        misses_per_category=0,
        seed=311,
    )

    assert first == second


def test_acceptance_prediction_filter_uses_deployment_threshold() -> None:
    """Only predictions meeting the selected score should be rendered."""
    predictions = [{"score": 0.79}, {"score": 0.8}, {"score": 0.91}]

    assert _filter_predictions(predictions, 0.8) == predictions[1:]


def test_acceptance_reasons_use_readable_category_names() -> None:
    """Contact sheets should explain sampling without opaque category ids."""
    reasons = _readable_selection_reasons(
        ["sample:category_1", "miss:category_4"],
        {1: "top", 4: "outerwear"},
    )

    assert reasons == ["sample:top", "miss:outerwear"]


def test_contact_sheet_pads_different_image_heights() -> None:
    """Mixed aspect ratios should produce a stable non-overlapping sheet."""
    first = np.full((20, 30, 3), 50, dtype=np.uint8)
    second = np.full((10, 30, 3), 100, dtype=np.uint8)
    third = np.full((15, 30, 3), 150, dtype=np.uint8)

    sheet = _build_contact_sheet([first, second, third], columns=2, gap=4)

    assert sheet.shape == (44, 64, 3)
    assert np.all(sheet[:20, :30] == 50)
    assert np.all(sheet[:10, 34:64] == 100)
    assert np.all(sheet[24:39, :30] == 150)
