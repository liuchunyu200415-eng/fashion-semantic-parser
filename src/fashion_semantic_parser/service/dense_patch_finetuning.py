"""Small-part weighting and semantics-safe Copy-Paste for DINOv2 fine-tuning."""

# Optional PyTorch components load only when the fine-tuning path is executed.
# pylint: disable=import-outside-toplevel

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, cast

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator

from fashion_semantic_parser.common.paths import resolve_project_path
from fashion_semantic_parser.dao.localization.referring_dataset import (
    ReferringTrainingItem,
)
from fashion_semantic_parser.dao.localization.referring_training import (
    ReferringTrainingSample,
)


class DensePatchFineTuningSettings(BaseModel):
    """Validated optimization and augmentation contract for backbone adaptation."""

    head_learning_rate: float = Field(default=1e-4, gt=0.0)
    backbone_learning_rate: float = Field(default=1e-5, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    training_steps: int = Field(default=100, ge=1)
    batch_size: int = Field(default=4, ge=1)
    seed: int = Field(default=312, ge=0)
    unfreeze_last_blocks: int = Field(default=2, ge=1, le=2)
    small_target_area_threshold: float = Field(default=0.01, gt=0.0, lt=1.0)
    small_target_loss_weight: float = Field(default=2.0, ge=1.0)
    weak_part_loss_weight: float = Field(default=1.5, ge=1.0)
    maximum_query_loss_weight: float = Field(default=3.0, ge=1.0)
    weak_part_labels: tuple[str, ...] = (
        "zipper",
        "rivet",
        "neckline",
        "pocket",
    )
    copy_paste_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    copy_paste_labels: tuple[str, ...] = (
        "zipper",
        "rivet",
        "neckline",
        "pocket",
    )

    @model_validator(mode="after")
    def validate_labels_and_weight_cap(self) -> "DensePatchFineTuningSettings":
        """Keep label lists stable and prevent the cap from disabling weighting."""
        if not _valid_label_list(self.weak_part_labels) or not _valid_label_list(
            self.copy_paste_labels
        ):
            raise ValueError("Fine-tuning label lists must be non-empty and unique.")
        if self.maximum_query_loss_weight < max(
            self.small_target_loss_weight,
            self.weak_part_loss_weight,
        ):
            raise ValueError("maximum_query_loss_weight cannot suppress one factor.")
        return self


class ReferringTrainingItemSource(Protocol):
    """Minimal lazy random-access contract used by the clean audit."""

    def __len__(self) -> int:
        """Return the number of query items."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> ReferringTrainingItem:
        """Load one complete item on demand."""
        raise NotImplementedError


@dataclass
class DenseFineTuningAuditRuntime:
    """Dependencies for a fixed clean-query loss audit."""

    encoder: Any
    dense_runtime: Any
    items: ReferringTrainingItemSource
    query_weights: np.ndarray
    device: str
    batch_size: int


def load_dense_patch_finetuning_settings(
    config_path: str | Path = "configs/localization_dense_patch_finetuning.yaml",
) -> DensePatchFineTuningSettings:
    """Load the validated small-part backbone adaptation contract."""
    path = resolve_project_path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cast(
        DensePatchFineTuningSettings,
        DensePatchFineTuningSettings.model_validate(raw),
    )


def query_loss_weight(
    item: ReferringTrainingItem,
    settings: DensePatchFineTuningSettings,
) -> float:
    """Return bounded multiplicative weight from GT size and weak-part status."""
    union_mask = np.asarray(item.target_masks, dtype=bool).any(axis=0)
    if not union_mask.any():
        raise ValueError("Fine-tuning requires a non-empty target Mask.")
    return query_loss_weight_from_area_fraction(
        item.sample,
        float(union_mask.mean()),
        settings,
    )


def query_loss_weight_from_area_fraction(
    sample: ReferringTrainingSample,
    target_area_fraction: float,
    settings: DensePatchFineTuningSettings,
) -> float:
    """Return the bounded query weight without retaining decoded image data."""
    if not 0.0 < target_area_fraction <= 1.0:
        raise ValueError("Target area fraction must be in (0, 1].")
    weight = 1.0
    if target_area_fraction < settings.small_target_area_threshold:
        weight *= settings.small_target_loss_weight
    if sample.target_label in settings.weak_part_labels:
        weight *= settings.weak_part_loss_weight
    return min(settings.maximum_query_loss_weight, weight)


# Explicit tensors keep this fixed audit independently usable.
# pylint: disable-next=too-many-locals
def clean_finetuning_audit_loss(
    audit: DenseFineTuningAuditRuntime,
    query_indices: np.ndarray,
) -> float:
    """Evaluate identical unaugmented queries before and after fine-tuning."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for fine-tuning audit.") from error
    from fashion_semantic_parser.service.dense_patch_alignment import (
        mask_to_patch_fractions,
    )
    from fashion_semantic_parser.service.dense_patch_training import runtime_loss

    indices = np.asarray(query_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Fine-tuning audit requires non-empty 1D query indices.")
    if int(indices.min()) < 0 or int(indices.max()) >= len(audit.items):
        raise ValueError("Fine-tuning audit query index is out of range.")
    audit.encoder.set_finetuning_mode(False)
    audit.dense_runtime.projection.eval()
    if audit.dense_runtime.decoder is None:
        raise RuntimeError("Fine-tuning audit requires the multiscale decoder.")
    audit.dense_runtime.decoder.eval()
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    try:
        with torch.inference_mode():
            for start in range(0, len(indices), audit.batch_size):
                batch_indices = indices[start : start + audit.batch_size]
                batch_items = [audit.items[int(index)] for index in batch_indices]
                images = [item.image_rgb for item in batch_items]
                masks = [item.target_masks.any(axis=0) for item in batch_items]
                patch_tensor, geometries = audit.encoder.encode_dense_trainable_batch(
                    images
                )
                targets = np.stack(
                    [
                        mask_to_patch_fractions(
                            mask,
                            geometry,
                            patch_size=audit.encoder.settings.patch_size,
                        ).reshape(-1)
                        for mask, geometry in zip(masks, geometries, strict=True)
                    ]
                )
                device_indices = torch.from_numpy(batch_indices).to(
                    device=audit.device,
                    dtype=torch.long,
                )
                weights = torch.from_numpy(audit.query_weights[batch_indices]).to(
                    device=audit.device
                )
                projected = audit.dense_runtime.projection(
                    audit.dense_runtime.text_tensor[device_indices]
                )
                loss = runtime_loss(
                    audit.dense_runtime,
                    patch_tensor,
                    projected,
                    torch.from_numpy(targets).to(device=audit.device),
                    weights,
                )
                batch_weight = float(weights.sum().item())
                weighted_loss_sum += float(loss.item()) * batch_weight
                weight_sum += batch_weight
    finally:
        audit.encoder.set_finetuning_mode(True)
        audit.dense_runtime.projection.train()
        audit.dense_runtime.decoder.train()
    if weight_sum <= 0.0 or not np.isfinite(weighted_loss_sum):
        raise RuntimeError("Fine-tuning clean audit produced invalid loss totals.")
    return weighted_loss_sum / weight_sum


def build_copy_paste_donor_groups(
    samples: Sequence[ReferringTrainingSample],
) -> dict[str, list[int]]:
    """Index potential same-label donors without changing query semantics."""
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample.target_label, []).append(index)
    return groups


def select_copy_paste_donor(
    receiver_index: int,
    samples: Sequence[ReferringTrainingSample],
    donor_groups: dict[str, list[int]],
    settings: DensePatchFineTuningSettings,
    rng: np.random.Generator,
) -> int | None:
    """Choose a different-image donor while preserving attribute supervision."""
    receiver = samples[receiver_index]
    if (
        receiver.target_label not in settings.copy_paste_labels
        or float(rng.random()) >= settings.copy_paste_probability
    ):
        return None
    candidates = []
    for index in donor_groups.get(receiver.target_label, []):
        donor = samples[index]
        if donor.source_image_id == receiver.source_image_id:
            continue
        if "attribute" in receiver.dimensions and (
            donor.source_attribute_ids != receiver.source_attribute_ids
        ):
            continue
        candidates.append(index)
    if not candidates:
        return None
    return int(candidates[int(rng.integers(0, len(candidates)))])


def copy_paste_same_label_instance(
    receiver: ReferringTrainingItem,
    donor: ReferringTrainingItem,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace one target appearance at the same receiver location.

    The destination instance geometry is replaced by a resized same-label donor
    Mask inside the original target box. Keeping that box fixed preserves
    spatial and garment-relation modifiers; attribute queries require matching
    source attributes at donor selection time.
    """
    if receiver.sample.target_label != donor.sample.target_label:
        raise ValueError("Copy-Paste donor and receiver labels must match.")
    receiver_masks = np.asarray(receiver.target_masks, dtype=bool)
    donor_masks = np.asarray(donor.target_masks, dtype=bool)
    if receiver_masks.shape[0] == 0 or donor_masks.shape[0] == 0:
        raise ValueError("Copy-Paste requires non-empty instance arrays.")
    receiver_index = int(rng.integers(0, receiver_masks.shape[0]))
    donor_index = int(rng.integers(0, donor_masks.shape[0]))
    receiver_box = _mask_box(receiver_masks[receiver_index])
    resized_image, resized_mask = _resize_donor_instance(
        donor.image_rgb,
        donor_masks[donor_index],
        output_width=receiver_box[2] - receiver_box[0],
        output_height=receiver_box[3] - receiver_box[1],
    )

    image = receiver.image_rgb.copy()
    destination = image[
        receiver_box[1] : receiver_box[3],
        receiver_box[0] : receiver_box[2],
    ]
    destination[resized_mask] = resized_image[resized_mask]
    augmented_masks = receiver_masks.copy()
    augmented_masks[receiver_index] = False
    augmented_masks[
        receiver_index,
        receiver_box[1] : receiver_box[3],
        receiver_box[0] : receiver_box[2],
    ] = resized_mask
    union_mask = augmented_masks.any(axis=0)
    if not union_mask.any():
        raise RuntimeError("Copy-Paste produced an empty training target.")
    return image, union_mask


def configure_dinov2_last_blocks(
    model: Any,
    unfreeze_last_blocks: int,
) -> list[tuple[str, Any]]:
    """Freeze DINOv2 except its final blocks and terminal normalization layer."""
    blocks = getattr(model, "blocks", None)
    terminal_norm = getattr(model, "norm", None)
    if (
        blocks is None
        or terminal_norm is None
        or not hasattr(model, "named_parameters")
    ):
        raise ValueError("DINOv2 model lacks blocks, norm, or named parameters.")
    block_count = len(blocks)
    if block_count == 0 or not 1 <= unfreeze_last_blocks <= min(2, block_count):
        raise ValueError("unfreeze_last_blocks must select one or two model blocks.")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in blocks[-unfreeze_last_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in terminal_norm.parameters():
        parameter.requires_grad = True
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("DINOv2 fine-tuning selected no trainable parameters.")
    return trainable


def trainable_dinov2_state_dict(model: Any) -> dict[str, Any]:
    """Return CPU tensors only for explicitly trainable DINOv2 parameters."""
    state = model.state_dict()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not trainable_names:
        raise ValueError("DINOv2 has no trainable parameters to save.")
    return {
        name: state[name].detach().cpu()
        for name in sorted(trainable_names)
        if name in state
    }


def load_trainable_dinov2_state_dict(
    model: Any,
    state_dict: dict[str, Any],
    *,
    unfreeze_last_blocks: int,
) -> None:
    """Strictly restore the recorded trainable DINOv2 parameter subset."""
    trainable = configure_dinov2_last_blocks(model, unfreeze_last_blocks)
    expected = {name for name, _ in trainable}
    actual = set(state_dict)
    if actual != expected:
        missing = sorted(expected.difference(actual))[:5]
        unexpected = sorted(actual.difference(expected))[:5]
        raise ValueError(
            "Fine-tuned DINOv2 state keys differ: "
            f"missing={missing} unexpected={unexpected}"
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError("Fine-tuned DINOv2 state contains unexpected parameters.")
    model.eval()


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return an exclusive integer box around one non-empty binary Mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Copy-Paste instance Mask cannot be empty.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _resize_donor_instance(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop and resize one donor object with nearest-neighbor Mask geometry."""
    box = _mask_box(mask)
    image_crop = image_rgb[box[1] : box[3], box[0] : box[2]]
    mask_crop = mask[box[1] : box[3], box[0] : box[2]]
    resized_image = cv2.resize(
        image_crop,
        (output_width, output_height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_mask = cv2.resize(
        np.asarray(mask_crop, dtype=np.uint8),
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    if not resized_mask.any():
        raise ValueError("Copy-Paste resize removed the donor instance.")
    return resized_image, resized_mask


def _valid_label_list(labels: tuple[str, ...]) -> bool:
    """Return whether a configured label tuple is non-empty, unique, and clean."""
    return (
        bool(labels)
        and len(set(labels)) == len(labels)
        and all(label.strip() for label in labels)
    )
