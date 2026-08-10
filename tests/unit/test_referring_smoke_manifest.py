"""Tests for the open-language localization smoke manifest contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from fashion_semantic_parser.dao.localization.referring_smoke import (
    ReferringSmokeManifest,
    load_referring_smoke_manifest,
)


def _box_target() -> dict[str, object]:
    return {
        "label": "left_cuff",
        "box": {"x_min": 1, "y_min": 2, "x_max": 5, "y_max": 8},
    }


def _case(**updates: object) -> dict[str, object]:
    case: dict[str, object] = {
        "id": "left_cuff_001",
        "image_path": "data/raw/referring_smoke/image.jpg",
        "query": "衣服左边的袖口",
        "grounding_prompt": "the cuff on the left side of the garment",
        "dimensions": ["basic", "spatial"],
        "novelty": "novel_composition",
        "reference_frame": "image",
        "annotation_status": "box",
        "targets": [_box_target()],
        "contrast_set_id": "image_001_cuffs",
    }
    case.update(updates)
    return case


def _manifest(*cases: dict[str, object]) -> ReferringSmokeManifest:
    return ReferringSmokeManifest.model_validate(
        {
            "schema_version": 1,
            "name": "open language smoke",
            "cases": list(cases or (_case(),)),
        }
    )


def test_manifest_normalizes_text_and_infers_target_count() -> None:
    """A labelled case should retain the full expression and exact target count."""
    manifest = _manifest(
        _case(
            query="  衣服左边的袖口  ",
            grounding_prompt=" the cuff   on the left side ",
        )
    )

    case = manifest.cases[0]
    assert case.query == "衣服左边的袖口"
    assert case.grounding_prompt == "the cuff on the left side"
    assert case.expected_count == 1
    assert case.dimensions == ["basic", "spatial"]


@pytest.mark.parametrize("field", ["image_path", "query", "grounding_prompt"])
def test_manifest_rejects_whitespace_only_required_text(field: str) -> None:
    """Whitespace cannot silently trigger a fixed-label or missing-image path."""
    with pytest.raises(ValidationError):
        _manifest(_case(**{field: "   "}))


def test_manifest_requires_unique_case_ids() -> None:
    """Per-query responses are keyed by case ID, not shared image ID."""
    with pytest.raises(ValidationError, match="Duplicate"):
        _manifest(_case(), _case(query="右边的袖口"))


def test_manifest_rejects_mismatched_expected_count() -> None:
    """Exact-set success requires an unambiguous number of targets."""
    with pytest.raises(ValidationError, match="expected_count"):
        _manifest(_case(expected_count=2))


def test_negative_and_unlabelled_cases_have_distinct_contracts() -> None:
    """Reviewed negatives are scored while unlabelled cases remain unscored."""
    negative = _case(
        id="negative_zipper",
        annotation_status="negative",
        targets=[],
        expected_count=None,
        dimensions=["attribute"],
        reference_frame=None,
    )
    unlabelled = _case(
        id="unlabelled_relation",
        annotation_status="unlabelled",
        targets=[],
        expected_count=None,
        dimensions=["relation"],
        reference_frame=None,
    )
    manifest = _manifest(negative, unlabelled)

    assert manifest.cases[0].expected_count == 0
    assert manifest.cases[1].expected_count is None

    with pytest.raises(ValidationError, match="cannot define expected_count"):
        _manifest({**unlabelled, "expected_count": 1})


def test_mask_case_requires_valid_coco_polygons() -> None:
    """Malformed Mask ground truth must fail before expensive inference."""
    valid = _case(
        annotation_status="mask",
        targets=[
            {
                "label": "left_cuff",
                "segmentation": [[1, 2, 5, 2, 5, 8, 1, 8]],
            }
        ],
    )
    assert _manifest(valid).cases[0].annotation_status == "mask"

    with pytest.raises(ValidationError, match="three numeric xy points"):
        _manifest(
            _case(
                annotation_status="mask",
                targets=[{"segmentation": [[1, 2, 3, 4]]}],
            )
        )


def test_spatial_queries_require_reference_frame_and_unique_dimensions() -> None:
    """Left/right meaning must be explicit and modifiers cannot be duplicated."""
    with pytest.raises(ValidationError, match="reference_frame"):
        _manifest(_case(reference_frame=None))
    with pytest.raises(ValidationError, match="duplicates"):
        _manifest(_case(dimensions=["spatial", "spatial"]))


def test_committed_template_covers_all_language_dimensions() -> None:
    """The starter set should exercise modifiers while remaining unscored."""
    project_root = Path(__file__).resolve().parents[2]
    manifest = load_referring_smoke_manifest(
        project_root / "data/benchmarks/localization/referring_smoke_v1.template.json"
    )

    assert len(manifest.cases) == 20
    assert {dimension for case in manifest.cases for dimension in case.dimensions} == {
        "basic",
        "spatial",
        "attribute",
        "relation",
    }
    assert all(case.annotation_status == "unlabelled" for case in manifest.cases)
