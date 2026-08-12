"""Tests for supervised DINOv2 query-to-patch alignment."""

from types import SimpleNamespace

import numpy as np
import pytest

from fashion_semantic_parser.service.dense_patch_alignment import (
    DensePatchAlignmentSettings,
    balanced_patch_mask_loss,
    build_dense_patch_training_cache,
    dense_patch_logits,
    load_dense_patch_alignment_settings,
    mask_to_patch_fractions,
)
from fashion_semantic_parser.service.dinov2_region_encoder import (
    DinoV2DenseFeatureMap,
    DinoV2LetterboxGeometry,
)


def test_mask_to_patch_fractions_preserves_soft_coverage() -> None:
    """Patch targets should retain fractional area instead of category labels."""
    geometry = DinoV2LetterboxGeometry(
        original_height=4,
        original_width=4,
        resized_height=4,
        resized_width=4,
        top=0,
        left=0,
        output_size=4,
    )
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1

    fractions = mask_to_patch_fractions(mask, geometry, patch_size=2)

    assert fractions.shape == (2, 2)
    assert fractions[0, 0] == pytest.approx(0.25)
    assert fractions.sum() == pytest.approx(0.25)


def test_dense_training_cache_encodes_each_source_image_once() -> None:
    """Repeated language queries must reuse one frozen dense image feature map."""

    class FakeEncoder:
        """Return a deterministic two-by-two patch feature grid."""

        settings = SimpleNamespace(patch_size=2)

        def __init__(self) -> None:
            self.calls = 0

        def encode_dense(self, image_rgb: np.ndarray) -> DinoV2DenseFeatureMap:
            """Return fixed features with the input image geometry."""
            self.calls += 1
            geometry = DinoV2LetterboxGeometry(
                original_height=4,
                original_width=4,
                resized_height=4,
                resized_width=4,
                top=0,
                left=0,
                output_size=4,
            )
            return DinoV2DenseFeatureMap(
                features=np.ones((2, 2, 3), dtype=np.float32),
                geometry=geometry,
            )

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    first_mask = np.zeros((1, 4, 4), dtype=np.uint8)
    first_mask[0, :2, :2] = 1
    second_mask = np.zeros((1, 4, 4), dtype=np.uint8)
    second_mask[0, 2:, 2:] = 1
    items = [
        SimpleNamespace(
            sample=SimpleNamespace(source_image_id=7),
            image_rgb=image,
            target_masks=first_mask,
        ),
        SimpleNamespace(
            sample=SimpleNamespace(source_image_id=7),
            image_rgb=image.copy(),
            target_masks=second_mask,
        ),
    ]
    encoder = FakeEncoder()

    cache = build_dense_patch_training_cache(items, encoder)

    assert encoder.calls == 1
    assert cache.image_ids == (7,)
    assert cache.image_features.shape == (1, 4, 3)
    assert cache.query_image_indices.tolist() == [0, 0]
    assert cache.target_patch_fractions.shape == (2, 4)


def test_dense_patch_settings_match_committed_training_contract() -> None:
    """The project config must retain fixed probability calibration defaults."""
    settings = load_dense_patch_alignment_settings()

    assert settings.learning_rate == DensePatchAlignmentSettings().learning_rate
    assert settings.initial_logit_scale == pytest.approx(1.0 / 0.07)
    assert settings.training_steps == 300
    assert settings.batch_size == 32
    assert settings.probability_threshold == 0.5


def test_dense_patch_loss_trains_calibrated_similarity() -> None:
    """Aligned positive patches should produce a lower supervised loss."""
    torch = pytest.importorskip("torch")
    patches = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    query = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0]])
    scale = torch.tensor(2.0).log()

    aligned_logits = dense_patch_logits(
        patches,
        query,
        scale,
        torch.tensor(0.0),
        max_logit_scale=100.0,
    )
    reversed_logits = dense_patch_logits(
        patches,
        torch.tensor([[0.0, 1.0]]),
        scale,
        torch.tensor(0.0),
        max_logit_scale=100.0,
    )

    assert balanced_patch_mask_loss(
        aligned_logits,
        targets,
    ) < balanced_patch_mask_loss(reversed_logits, targets)
