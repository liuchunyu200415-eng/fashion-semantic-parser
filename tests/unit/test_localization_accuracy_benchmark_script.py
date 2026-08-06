"""Tests for bounded PRD 3.1.2 exact-ground-truth benchmarking."""

from types import SimpleNamespace

from fashion_semantic_parser.dao.localization.taxonomy import (
    PRD_LOCALIZATION_REGION_COVERAGE,
)
from scripts.benchmark_localization_accuracy import (
    _build_acceptance_result,
    _exact_prd_source_categories,
    _segmentation_prediction_to_coco,
    _segmentation_settings_overrides,
    _select_exact_gt_images,
)


def test_exact_source_categories_exclude_partial_and_missing_regions() -> None:
    """The 92% calculation must not use proxies without equivalent masks."""
    coverage = [
        SimpleNamespace(status="exact", source_categories=("collar", "lapel")),
        SimpleNamespace(status="partial", source_categories=("epaulette",)),
        SimpleNamespace(status="missing", source_categories=()),
        SimpleNamespace(status="exact", source_categories=("collar", "pocket")),
    ]

    assert _exact_prd_source_categories(coverage) == [
        "collar",
        "lapel",
        "pocket",
    ]


def test_checkpoint_and_candidate_threshold_can_be_overridden_together() -> None:
    """Checkpoint comparisons must not silently evaluate deployment weights."""
    assert _segmentation_settings_overrides(
        weights="model_0004999.pth",
        inference_score_threshold=0.0,
    ) == {
        "weights": "model_0004999.pth",
        "score_threshold": 0.0,
    }
    assert (
        _segmentation_settings_overrides(
            weights=None,
            inference_score_threshold=None,
        )
        == {}
    )


def test_project_exact_gt_scope_has_sixteen_sources_and_no_epaulette() -> None:
    """The formal scope is collar, pocket, and decoration, not shoulder proxy."""
    categories = _exact_prd_source_categories(PRD_LOCALIZATION_REGION_COVERAGE)

    assert len(categories) == 16
    assert "epaulette" not in categories
    assert {"collar", "lapel", "neckline", "pocket", "ruffle"} <= set(categories)


def test_smoke_image_limit_is_applied_per_category_then_unioned() -> None:
    """A smoke subset should retain representation from every exact label."""
    source = {
        "images": [
            {"id": 1, "file_name": "1.jpg"},
            {"id": 2, "file_name": "2.jpg"},
            {"id": 3, "file_name": "3.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 7, "iscrowd": 0},
            {"image_id": 2, "category_id": 7, "iscrowd": 0},
            {"image_id": 3, "category_id": 8, "iscrowd": 0},
        ],
        "categories": [],
    }

    images = _select_exact_gt_images(
        source,
        category_ids={7, 8},
        image_limit_per_category=1,
    )

    assert [image["id"] for image in images] == [1, 3]


def test_part_instances_keep_their_exact_coco_category() -> None:
    """Grouped decoration evaluation must not collapse predicted class labels."""
    instance = SimpleNamespace(
        category_label="ruffle",
        confidence=0.81,
        mask=[[1.0, 2.0, 5.0, 2.0, 5.0, 8.0]],
        box=SimpleNamespace(x_min=1.0, y_min=2.0, x_max=5.0, y_max=8.0),
    )

    results = _segmentation_prediction_to_coco(
        SimpleNamespace(instances=[instance]),
        image_id=12,
        category_ids={"ruffle": 17},
    )

    assert results[0]["category_id"] == 17
    assert results[0]["bbox"] == [1.0, 2.0, 4.0, 6.0]


def test_acceptance_result_is_bounded_to_exact_gt_scope() -> None:
    """Even a high exact-GT recall must not claim overall eight-region success."""
    coverage = [
        SimpleNamespace(
            english_name="collar",
            status="exact",
            source_categories=("collar",),
        ),
        SimpleNamespace(
            english_name="cuff",
            status="missing",
            source_categories=(),
        ),
    ]
    evaluation = {
        "categories": ["collar"],
        "segm_direct_iou": {"Recall50-collar": 95.0},
    }

    result = _build_acceptance_result(
        evaluation=evaluation,
        summary={},
        coverage=coverage,
    )

    assert result["accuracy_contract"]["exact_gt_scope_passed"] is True
    assert result["accuracy_contract"]["overall_prd_accuracy_passed"] is None
    assert result["coverage"]["unscored_prd_regions"] == ["cuff"]
