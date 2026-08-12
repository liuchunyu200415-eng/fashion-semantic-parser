"""Generate class-agnostic high-recall region proposals with official SAM-HQ."""

import hashlib
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Mapping, cast

import numpy as np
import yaml
from pydantic import BaseModel, Field

from fashion_semantic_parser.common.exceptions import (
    ConfigurationError,
    ModelNotReadyError,
)
from fashion_semantic_parser.common.paths import resolve_project_path
from fashion_semantic_parser.service.grounded_sam_hq import (
    _load_sam_hq_module,
    _precision_context,
    _required_asset,
)


class SAMHQProposalSettings(BaseModel):
    """Validated settings for class-agnostic SAM-HQ candidate generation."""

    sam_hq_repo: str | None = "external/sam-hq"
    sam_hq_repo_commit: str = Field(
        default="e696978d60352dc9a26b12631cd91781502c6546",
        pattern=r"^[0-9a-f]{40}$",
    )
    sam_hq_weights: str = "models/checkpoints/localization/sam_hq_vit_b.pth"
    sam_hq_weights_sha256: str = Field(
        default="14a9d662cd6f5a9c2dba6d40ab0058d88d287e4a18fd6fdc6ad5fb1a3fdeaa57",
        pattern=r"^[0-9a-f]{64}$",
    )
    sam_hq_model_type: Literal["vit_b", "vit_l", "vit_h", "vit_tiny"] = "vit_b"
    sam_hq_module: Literal["auto", "segment_anything_hq", "segment_anything"] = "auto"
    device: str = "cuda"
    precision: Literal["fp32", "fp16"] = "fp16"
    points_per_side: int = Field(default=32, ge=1, le=128)
    points_per_batch: int = Field(default=64, ge=1, le=256)
    pred_iou_thresh: float = Field(default=0.75, ge=0.0, le=1.0)
    stability_score_thresh: float = Field(default=0.80, ge=0.0, le=1.0)
    box_nms_thresh: float = Field(default=0.70, ge=0.0, le=1.0)
    crop_n_layers: int = Field(default=1, ge=0, le=3)
    crop_nms_thresh: float = Field(default=0.70, ge=0.0, le=1.0)
    crop_n_points_downscale_factor: int = Field(default=2, ge=1, le=8)
    min_mask_region_area: int = Field(default=16, ge=0)
    max_regions: int = Field(default=200, ge=1, le=1000)


@dataclass(frozen=True)
class SAMHQMaskProposal:
    """One validated class-agnostic SAM-HQ proposal in image coordinates."""

    box: tuple[float, float, float, float]
    mask: np.ndarray
    area: int
    predicted_iou: float
    stability_score: float


def best_proposal_mask_iou(
    target_mask: np.ndarray,
    proposals: list[SAMHQMaskProposal],
) -> tuple[float, int | None]:
    """Return the best independent proposal IoU for one non-empty target Mask."""
    target = np.asarray(target_mask, dtype=bool)
    if target.ndim != 2 or not target.any():
        raise ValueError("Proposal recall requires one non-empty target Mask.")
    best_iou = 0.0
    best_index: int | None = None
    for index, proposal in enumerate(proposals):
        if proposal.mask.shape != target.shape:
            raise ValueError("Proposal and target Mask dimensions must match.")
        intersection = int(np.logical_and(target, proposal.mask).sum())
        union = int(np.logical_or(target, proposal.mask).sum())
        iou = intersection / union if union else 0.0
        if iou > best_iou:
            best_iou = iou
            best_index = index
    return best_iou, best_index


class SAMHQAutomaticProposalGenerator:
    """Lazy reusable official SAM-HQ automatic Mask generator."""

    def __init__(
        self,
        settings: SAMHQProposalSettings,
        *,
        generator: Any | None = None,
    ) -> None:
        self.settings = settings
        self._generator = generator
        self._torch: Any | None = None
        self._model_init_lock = Lock()
        self._inference_lock = Lock()
        self._prepare_external_path()

    def generate(self, image_rgb: np.ndarray) -> list[SAMHQMaskProposal]:
        """Return quality-ranked valid proposals for one uint8 RGB image."""
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("SAM-HQ proposal input must be an HxWx3 uint8 RGB array.")
        generator = self._get_generator()
        torch = self._torch
        with self._inference_lock:
            context = (
                _precision_context(
                    torch,
                    device=self.settings.device,
                    precision=self.settings.precision,
                )
                if torch is not None
                else nullcontext()
            )
            with context:
                raw_proposals = generator.generate(image)
        if not isinstance(raw_proposals, list):
            raise ModelNotReadyError("SAM-HQ automatic generator returned no list.")
        image_shape = (int(image.shape[0]), int(image.shape[1]))
        proposals: list[SAMHQMaskProposal] = []
        for record in raw_proposals:
            proposal = self._validate_proposal(record, image_shape=image_shape)
            if proposal is not None:
                proposals.append(proposal)
        proposals.sort(
            key=lambda proposal: (
                proposal.predicted_iou,
                proposal.stability_score,
                proposal.area,
            ),
            reverse=True,
        )
        return proposals[: self.settings.max_regions]

    def synchronize(self) -> None:
        """Synchronize CUDA so proposal timing includes queued GPU work."""
        if (
            self._torch is not None
            and self.settings.device == "cuda"
            and self._torch.cuda.is_available()
        ):
            self._torch.cuda.synchronize()

    def _prepare_external_path(self) -> None:
        """Expose the official SAM-HQ checkout without global PYTHONPATH edits."""
        if not self.settings.sam_hq_repo:
            return
        try:
            path = resolve_project_path(self.settings.sam_hq_repo)
        except ValueError:
            return
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    def _get_generator(self) -> Any:
        """Load the pinned official model and automatic generator once."""
        if self._generator is not None:
            return self._generator
        with self._model_init_lock:
            if self._generator is not None:
                return self._generator
            if not self.settings.sam_hq_repo:
                raise ModelNotReadyError(
                    "SAM-HQ proposal generation requires the pinned source checkout."
                )
            repo_path = resolve_project_path(self.settings.sam_hq_repo)
            weights_path = _required_asset(
                self.settings.sam_hq_weights,
                "SAM-HQ weights",
            )
            self._validate_local_assets(repo_path, weights_path)
            module = _load_sam_hq_module(self.settings.sam_hq_module)
            try:
                import torch  # type: ignore[import-not-found]

                registry = getattr(module, "sam_model_registry")
                generator_class = getattr(module, "SamAutomaticMaskGenerator")
                sam = registry[self.settings.sam_hq_model_type](
                    checkpoint=str(weights_path)
                )
                sam.to(device=self.settings.device)
                sam.eval()
                self._generator = generator_class(
                    sam,
                    points_per_side=self.settings.points_per_side,
                    points_per_batch=self.settings.points_per_batch,
                    pred_iou_thresh=self.settings.pred_iou_thresh,
                    stability_score_thresh=self.settings.stability_score_thresh,
                    box_nms_thresh=self.settings.box_nms_thresh,
                    crop_n_layers=self.settings.crop_n_layers,
                    crop_nms_thresh=self.settings.crop_nms_thresh,
                    crop_n_points_downscale_factor=(
                        self.settings.crop_n_points_downscale_factor
                    ),
                    min_mask_region_area=self.settings.min_mask_region_area,
                    output_mode="binary_mask",
                )
                self._torch = torch
            except (
                AttributeError,
                ImportError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
            ) as error:
                raise ModelNotReadyError(
                    "SAM-HQ automatic proposal generator could not be loaded."
                ) from error
        return self._generator

    def _validate_local_assets(self, repo_path: Path, weights_path: Path) -> None:
        """Reject drifting official source or an unverified SAM-HQ checkpoint."""
        head_path = repo_path / ".git" / "HEAD"
        if not head_path.is_file():
            raise ModelNotReadyError(
                "Pinned SAM-HQ checkout is missing; run "
                "scripts/setup_sam_hq_proposal_model.sh."
            )
        actual_commit = head_path.read_text(encoding="utf-8").strip()
        if actual_commit != self.settings.sam_hq_repo_commit:
            raise ModelNotReadyError(
                "SAM-HQ checkout is not at the pinned commit: "
                f"expected={self.settings.sam_hq_repo_commit} actual={actual_commit}"
            )
        digest = hashlib.sha256()
        with weights_path.open("rb") as weights_file:
            for chunk in iter(lambda: weights_file.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != self.settings.sam_hq_weights_sha256:
            raise ModelNotReadyError(
                "SAM-HQ weights checksum mismatch: "
                f"expected={self.settings.sam_hq_weights_sha256} "
                f"actual={actual_sha256}"
            )

    def _validate_proposal(
        self,
        record: Any,
        *,
        image_shape: tuple[int, int],
    ) -> SAMHQMaskProposal | None:
        """Convert one external record into a strict project proposal."""
        if not isinstance(record, Mapping):
            raise ModelNotReadyError("SAM-HQ proposal record must be a mapping.")
        mask = np.asarray(record.get("segmentation"), dtype=bool)
        if mask.shape != image_shape:
            raise ModelNotReadyError(
                "SAM-HQ proposal Mask dimensions do not match the input image."
            )
        predicted_iou = _quality_score(record.get("predicted_iou"), "predicted_iou")
        stability_score = _quality_score(
            record.get("stability_score"),
            "stability_score",
        )
        area = int(mask.sum())
        if area == 0:
            return None
        if area < self.settings.min_mask_region_area:
            return None
        y_values, x_values = np.nonzero(mask)
        box = (
            float(x_values.min()),
            float(y_values.min()),
            float(x_values.max() + 1),
            float(y_values.max() + 1),
        )
        return SAMHQMaskProposal(
            box=box,
            mask=mask,
            area=area,
            predicted_iou=predicted_iou,
            stability_score=stability_score,
        )


def load_sam_hq_proposal_settings(
    config_path: str | Path = "configs/localization_sam_hq_proposals.yaml",
) -> SAMHQProposalSettings:
    """Load the project SAM-HQ proposal configuration."""
    try:
        path = resolve_project_path(config_path)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    if not path.is_file():
        raise ConfigurationError(f"SAM-HQ proposal config not found: {config_path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid SAM-HQ proposal YAML: {config_path}"
        ) from error
    return cast(SAMHQProposalSettings, SAMHQProposalSettings.model_validate(raw))


def _quality_score(value: Any, name: str) -> float:
    """Validate one finite normalized external quality score."""
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ModelNotReadyError(f"SAM-HQ proposal {name} is invalid.") from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ModelNotReadyError(f"SAM-HQ proposal {name} must be in [0, 1].")
    return score
