"""Tests for small-part DINOv2 fine-tuning controls."""

from types import SimpleNamespace

import numpy as np
import pytest

from fashion_semantic_parser.service.dense_patch_finetuning import (
    DensePatchFineTuningSettings,
    configure_dinov2_last_blocks,
    copy_paste_same_label_instance,
    deterministic_epoch_batch_indices,
    load_dense_patch_finetuning_settings,
    load_trainable_dinov2_state_dict,
    query_loss_weight,
    query_loss_weight_from_area_fraction,
    trainable_dinov2_state_dict,
)


def test_committed_finetuning_config_locks_mentor_directed_scope() -> None:
    """The runnable config must expose every requested fine-tuning control."""
    settings = load_dense_patch_finetuning_settings()

    assert settings.unfreeze_last_blocks == 2
    assert settings.small_target_loss_weight == 2.0
    assert settings.copy_paste_probability == 0.5
    assert set(settings.weak_part_labels) == {
        "zipper",
        "rivet",
        "neckline",
        "pocket",
    }


def test_conservative_config_reduces_update_scope_and_learning_rates() -> None:
    """A failed aggressive smoke must lead to a materially safer A/B recipe."""
    baseline = load_dense_patch_finetuning_settings()
    conservative = load_dense_patch_finetuning_settings(
        "configs/localization_dense_patch_finetuning_conservative.yaml"
    )

    assert conservative.unfreeze_last_blocks == 1
    assert conservative.head_learning_rate < baseline.head_learning_rate
    assert conservative.backbone_learning_rate < baseline.backbone_learning_rate
    assert conservative.copy_paste_probability == baseline.copy_paste_probability


def test_scale_safe_config_reduces_optimizer_dose_after_1k_regression() -> None:
    """The 10k gate must not reuse the rejected 1k learning rates."""
    conservative = load_dense_patch_finetuning_settings(
        "configs/localization_dense_patch_finetuning_conservative.yaml"
    )
    scale_safe = load_dense_patch_finetuning_settings(
        "configs/localization_dense_patch_finetuning_scale_safe.yaml"
    )

    assert scale_safe.unfreeze_last_blocks == 1
    assert scale_safe.head_learning_rate < conservative.head_learning_rate
    assert scale_safe.backbone_learning_rate < conservative.backbone_learning_rate
    assert scale_safe.training_steps == 2500


def test_query_loss_weight_combines_small_and_weak_factors_with_cap() -> None:
    """Tiny weak-part supervision receives the configured bounded weight."""
    item = _item("rivet", image_id=1, mask_slice=(slice(0, 1), slice(0, 1)))
    settings = DensePatchFineTuningSettings(
        small_target_area_threshold=0.1,
        small_target_loss_weight=2.0,
        weak_part_loss_weight=2.0,
        maximum_query_loss_weight=3.0,
    )

    assert query_loss_weight(item, settings) == 3.0
    assert (
        query_loss_weight_from_area_fraction(item.sample, 1 / 36, settings) == 3.0
    )


def test_copy_paste_replaces_target_at_original_receiver_location() -> None:
    """Same-label appearance transfer must not move the referring target."""
    receiver = _item(
        "pocket",
        image_id=1,
        mask_slice=(slice(2, 5), slice(1, 4)),
        color=(10, 20, 30),
    )
    donor = _item(
        "pocket",
        image_id=2,
        mask_slice=(slice(0, 2), slice(4, 6)),
        color=(200, 100, 50),
    )

    image, union_mask = copy_paste_same_label_instance(
        receiver,
        donor,
        np.random.default_rng(7),
    )

    assert union_mask[:, :1].sum() == 0
    assert union_mask[2:5, 1:4].all()
    assert np.all(image[union_mask] == [200, 100, 50])


def test_copy_paste_rejects_different_labels() -> None:
    """A visually plausible but semantically different donor is invalid."""
    receiver = _item("pocket", image_id=1)
    donor = _item("zipper", image_id=2)

    with pytest.raises(ValueError, match="labels must match"):
        copy_paste_same_label_instance(
            receiver,
            donor,
            np.random.default_rng(7),
        )


def test_deterministic_batches_cover_each_epoch_without_replacement() -> None:
    """A nominal epoch must expose every selected query exactly once."""
    first = list(
        deterministic_epoch_batch_indices(
            sample_count=6,
            batch_size=2,
            steps=6,
            seed=312,
        )
    )
    second = list(
        deterministic_epoch_batch_indices(
            sample_count=6,
            batch_size=2,
            steps=6,
            seed=312,
        )
    )

    assert [batch.tolist() for batch in first] == [
        batch.tolist() for batch in second
    ]
    assert sorted(np.concatenate(first[:3]).tolist()) == list(range(6))
    assert sorted(np.concatenate(first[3:]).tolist()) == list(range(6))


def test_dinov2_finetuning_selects_and_restores_only_last_blocks() -> None:
    """Checkpoint state cannot silently include frozen backbone parameters."""
    torch = pytest.importorskip("torch")

    # pylint: disable-next=too-few-public-methods
    class TinyBackbone(torch.nn.Module):
        """Minimal block/norm structure matching the DINOv2 contract."""

        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = torch.nn.Linear(2, 2)
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
            self.norm = torch.nn.LayerNorm(2)

    source = TinyBackbone()
    trainable = configure_dinov2_last_blocks(source, 2)
    names = {name for name, _ in trainable}

    assert all(name.startswith(("blocks.2.", "blocks.3.", "norm.")) for name in names)
    assert not source.patch_embed.weight.requires_grad

    state = trainable_dinov2_state_dict(source)
    target = TinyBackbone()
    load_trainable_dinov2_state_dict(
        target,
        state,
        unfreeze_last_blocks=2,
    )
    assert set(state) == names


def _item(
    label: str,
    *,
    image_id: int,
    mask_slice: tuple[slice, slice] = (slice(1, 4), slice(1, 4)),
    color: tuple[int, int, int] = (10, 20, 30),
) -> SimpleNamespace:
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    image[:, :] = color
    mask = np.zeros((1, 6, 6), dtype=np.uint8)
    mask[(0, *mask_slice)] = 1
    return SimpleNamespace(
        sample=SimpleNamespace(
            target_label=label,
            source_image_id=image_id,
            dimensions=["basic"],
            source_attribute_ids=[],
        ),
        image_rgb=image,
        target_masks=mask,
    )
