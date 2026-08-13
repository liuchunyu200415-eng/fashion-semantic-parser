"""Tests for paired image/Mask preparation in the DINOv2 region path."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from fashion_semantic_parser.service.dense_region_localization import (
    patch_scores_to_mask,
)
from fashion_semantic_parser.service.dinov2_region_encoder import (
    DinoV2RegionEncoder,
    DinoV2RegionEncoderSettings,
    letterbox_geometry,
    letterbox_image,
    letterbox_image_and_masks,
    load_dinov2_region_settings,
    masks_to_patch_occupancy,
    patch_scores_to_image,
)


def test_letterbox_preserves_aspect_ratio_and_independent_masks() -> None:
    """Image and target Masks must receive the same non-distorting transform."""
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :, 0] = 200
    masks = np.zeros((2, 10, 20), dtype=np.uint8)
    masks[0, 2:5, 2:6] = 1
    masks[1, 4:8, 14:19] = 1

    resized_image, resized_masks = letterbox_image_and_masks(
        image,
        masks,
        output_size=28,
    )

    assert resized_image.shape == (28, 28, 3)
    assert resized_masks.shape == (2, 28, 28)
    assert np.all(resized_masks.sum(axis=(1, 2)) > 0)
    assert np.all(resized_image[:7] == [124, 116, 104])
    assert np.all(resized_image[21:] == [124, 116, 104])
    assert resized_masks[0].sum() != resized_masks[1].sum()


def test_letterbox_rejects_misaligned_masks() -> None:
    """Mask coordinates cannot be pooled against a differently sized image."""
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    masks = np.zeros((1, 9, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="NxHxW"):
        letterbox_image_and_masks(image, masks, output_size=28)


def test_dense_letterbox_geometry_restores_patch_scores() -> None:
    """Dense patch scores must return to the source image without padding."""
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    letterboxed, geometry = letterbox_image(image, output_size=28)
    restored = patch_scores_to_image(
        np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        geometry,
    )

    assert letterboxed.shape == (28, 28, 3)
    assert geometry == letterbox_geometry((10, 20), output_size=28)
    assert geometry.top == 7
    assert geometry.left == 0
    assert restored.shape == (10, 20)
    assert np.all(np.isfinite(restored))
    assert restored[:, -1].mean() > restored[:, 0].mean()


def test_patch_mask_threshold_is_applied_after_continuous_restoration() -> None:
    """Schema-one/two inference must preserve probability interpolation order."""
    geometry = letterbox_geometry((5, 13), output_size=28)
    scores = np.asarray([[0.4, 0.6], [0.4, 0.6]], dtype=np.float32)

    restored = patch_scores_to_mask(scores, geometry, threshold=0.5)

    expected = patch_scores_to_image(scores, geometry) >= 0.5
    assert np.array_equal(restored, expected)
    with pytest.raises(ValueError, match="finite"):
        patch_scores_to_mask(scores, geometry, threshold=np.nan)


def test_dense_geometry_rejects_invalid_dimensions_and_scores() -> None:
    """Invalid source geometry and non-finite score grids must fail early."""
    with pytest.raises(ValueError, match="positive"):
        letterbox_geometry((0, 20), output_size=28)

    geometry = letterbox_geometry((10, 20), output_size=28)
    with pytest.raises(ValueError, match="finite 2D"):
        patch_scores_to_image(np.asarray([[np.nan]]), geometry)


def test_letterbox_preserves_target_smaller_than_one_output_pixel() -> None:
    """A valid tiny Fashionpedia part must remain an evaluable candidate."""
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    masks = np.zeros((1, 1000, 1000), dtype=np.uint8)
    masks[0, 999, 999] = 1

    _, resized_masks = letterbox_image_and_masks(
        image,
        masks,
        output_size=28,
    )

    assert resized_masks.sum() == 1
    assert resized_masks[0, 27, 27]


def test_letterbox_rejects_empty_source_target() -> None:
    """Target preservation cannot turn an invalid empty annotation into a part."""
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    masks = np.zeros((1, 10, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="source target Mask"):
        letterbox_image_and_masks(image, masks, output_size=28)


def test_region_settings_require_exact_patch_grid() -> None:
    """The input must map to a deterministic DINOv2 patch-token grid."""
    with pytest.raises(ValidationError, match="divisible"):
        DinoV2RegionEncoderSettings(input_size=512)


def test_high_resolution_config_preserves_prd_dinov2_contract() -> None:
    """Resolution smoke may change token density but not the PRD backbone."""
    settings = load_dinov2_region_settings(
        "configs/localization_dinov2_region_728.yaml"
    )

    assert settings.input_size == 728
    assert settings.input_size // settings.patch_size == 52
    assert settings.model_name == "dinov2_vits14"
    assert settings.feature_dimension == 384


def test_patch_occupancy_preserves_one_pixel_target() -> None:
    """Small accessories cannot disappear when Masks become patch selectors."""
    masks = np.zeros((1, 28, 28), dtype=np.uint8)
    masks[0, 27, 27] = 1

    occupancy = masks_to_patch_occupancy(masks, patch_size=14)

    assert occupancy.shape == (1, 2, 2)
    assert occupancy.sum() == 1
    assert occupancy[0, 1, 1]


def test_fp16_is_not_allowed_on_cpu() -> None:
    """CPU smoke cannot silently claim the CUDA fp16 implementation path."""
    with pytest.raises(ValidationError, match="CUDA"):
        DinoV2RegionEncoderSettings(device="cpu", precision="fp16")


def test_project_smoke_config_matches_official_small_backbone() -> None:
    """The committed smoke config must keep model geometry explicit."""
    settings = load_dinov2_region_settings()

    assert settings.model_name == "dinov2_vits14"
    assert settings.repo_commit == "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    assert settings.weights_size_bytes == 88283115
    assert settings.input_size == 518
    assert settings.patch_size == 14
    assert settings.feature_dimension == 384


def test_local_assets_require_pinned_detached_commit_and_weight_size(
    tmp_path: Path,
) -> None:
    """Runtime cannot silently use drifting source or partial official weights."""
    commit = "a" * 40
    repo_path = tmp_path / "dinov2"
    head_path = repo_path / ".git" / "HEAD"
    head_path.parent.mkdir(parents=True)
    head_path.write_text(commit + "\n", encoding="utf-8")
    weights_path = tmp_path / "weights.pth"
    weights_path.write_bytes(b"1234")
    encoder = DinoV2RegionEncoder(
        DinoV2RegionEncoderSettings(
            repo_commit=commit,
            weights_size_bytes=4,
        )
    )

    encoder._validate_local_assets(repo_path, weights_path)

    weights_path.write_bytes(b"123")
    with pytest.raises(RuntimeError, match="unexpected size"):
        encoder._validate_local_assets(repo_path, weights_path)
