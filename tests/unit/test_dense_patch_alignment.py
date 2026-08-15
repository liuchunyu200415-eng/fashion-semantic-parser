"""Tests for supervised DINOv2 query-to-patch alignment."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fashion_semantic_parser.service.dense_patch_alignment import (
    DensePatchAlignmentSettings,
    apply_finetuned_dinov2_checkpoint,
    balanced_patch_mask_loss,
    build_dense_patch_training_cache,
    dense_patch_logits,
    load_dense_patch_alignment_checkpoint,
    load_dense_patch_alignment_settings,
    mask_to_patch_fractions,
)
from fashion_semantic_parser.service.dense_patch_area import (
    build_query_area_predictor,
)
from fashion_semantic_parser.service.dense_patch_decoder import (
    build_multiscale_patch_decoder,
)
from fashion_semantic_parser.service.dense_patch_metrics import (
    patch_probability_metrics,
    select_patch_probability_threshold,
)
from fashion_semantic_parser.service.dinov2_region_encoder import (
    DinoV2DenseFeatureMap,
    DinoV2LetterboxGeometry,
)
from fashion_semantic_parser.service.region_text_alignment import (
    RegionTextAlignmentSettings,
    build_text_projection,
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
    assert settings.calibration_thresholds[0] == 0.5
    assert settings.calibration_thresholds[-1] == 0.99


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


def test_dense_patch_loss_applies_normalized_query_weights() -> None:
    """Small-part weighting must affect the aggregate without changing scale."""
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[4.0, -4.0], [-4.0, 4.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    unweighted = balanced_patch_mask_loss(logits, targets)
    first_weighted = balanced_patch_mask_loss(
        logits,
        targets,
        torch.tensor([4.0, 1.0]),
    )
    second_weighted = balanced_patch_mask_loss(
        logits,
        targets,
        torch.tensor([1.0, 4.0]),
    )

    assert first_weighted < unweighted < second_weighted


def test_training_threshold_selection_prefers_tighter_equal_recall_mask() -> None:
    """Training calibration should reduce overprediction without validation GT."""
    probabilities = np.asarray(
        [
            [0.99, 0.90, 0.70, 0.60],
            [0.95, 0.85, 0.40, 0.30],
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    selected, audit = select_patch_probability_threshold(
        probabilities,
        targets,
        (0.5, 0.8, 0.9),
    )

    assert selected == 0.8
    assert audit["0.800"]["patch_recall50"] == 1.0
    assert audit["0.800"]["mean_patch_iou"] == 1.0
    assert (
        patch_probability_metrics(
            probabilities,
            targets,
            threshold=selected,
        )["patch_recall50_count"]
        == 2
    )


def test_patch_probability_metrics_retain_empty_prediction_misses() -> None:
    """An overly strict threshold must leave every missed query in the denominator."""
    metrics = patch_probability_metrics(
        np.asarray([[0.4, 0.3]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        threshold=0.9,
    )

    assert metrics["query_count"] == 1
    assert metrics["patch_recall50_count"] == 0
    assert metrics["patch_recall50"] == 0.0
    assert metrics["mean_patch_iou"] == 0.0


def test_schema_two_checkpoint_restores_multiscale_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation must strictly restore the trained decoder architecture.

    Args:
        tmp_path: Isolated checkpoint directory.
        monkeypatch: Resolver patch fixture.
    """
    torch = pytest.importorskip("torch")
    path = tmp_path / "dense_decoder.pt"
    alignment = RegionTextAlignmentSettings(
        text_dimension=4,
        region_dimension=4,
        hidden_dimension=4,
    )
    dense = DensePatchAlignmentSettings(
        decoder_hidden_dimension=8,
        decoder_branch_dimension=8,
        decoder_dilations=(1,),
    )
    projection = build_text_projection(alignment)
    decoder = build_multiscale_patch_decoder(4, dense)
    torch.save(
        {
            "schema_version": 2,
            "alignment_settings": alignment.model_dump(mode="json"),
            "dense_settings": dense.model_dump(mode="json"),
            "projection_state_dict": projection.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "logit_scale": 1.0,
            "logit_bias": 0.0,
            "base_encoders_frozen": True,
            "dinov2_model": "dinov2_vits14",
            "text_model": "BAAI/bge-m3",
            "model_type": "multiscale_decoder",
        },
        path,
    )
    monkeypatch.setattr(
        "fashion_semantic_parser.service.dense_patch_alignment.resolve_project_path",
        lambda _: path,
    )

    checkpoint = load_dense_patch_alignment_checkpoint(path, device="cpu")

    assert checkpoint.model_type == "multiscale_decoder"
    assert checkpoint.decoder is not None
    assert checkpoint.area_predictor is None
    assert checkpoint.training_input_size == 518


def test_schema_three_checkpoint_restores_query_area_predictor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema three must restore decoder and open-query area control.

    Args:
        tmp_path: Isolated checkpoint directory.
        monkeypatch: Resolver patch fixture.
    """
    torch = pytest.importorskip("torch")
    path = tmp_path / "dense_area_decoder.pt"
    alignment = RegionTextAlignmentSettings(
        text_dimension=4,
        region_dimension=4,
        hidden_dimension=4,
    )
    dense = DensePatchAlignmentSettings(
        decoder_hidden_dimension=8,
        decoder_branch_dimension=8,
        decoder_dilations=(1,),
        area_hidden_dimension=8,
    )
    projection = build_text_projection(alignment)
    decoder = build_multiscale_patch_decoder(4, dense)
    area_predictor = build_query_area_predictor(4, dense)
    torch.save(
        {
            "schema_version": 3,
            "alignment_settings": alignment.model_dump(mode="json"),
            "dense_settings": dense.model_dump(mode="json"),
            "projection_state_dict": projection.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "area_predictor_state_dict": area_predictor.state_dict(),
            "logit_scale": 1.0,
            "logit_bias": 0.0,
            "base_encoders_frozen": True,
            "dinov2_model": "dinov2_vits14",
            "text_model": "BAAI/bge-m3",
            "model_type": "multiscale_area_decoder",
            "dinov2_input_size": 728,
        },
        path,
    )
    monkeypatch.setattr(
        "fashion_semantic_parser.service.dense_patch_alignment.resolve_project_path",
        lambda _: path,
    )

    checkpoint = load_dense_patch_alignment_checkpoint(path, device="cpu")

    assert checkpoint.model_type == "multiscale_area_decoder"
    assert checkpoint.decoder is not None
    assert checkpoint.area_predictor is not None
    assert checkpoint.training_input_size == 728


def test_schema_four_checkpoint_records_finetuned_dinov2_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backbone adaptation must be explicit and separate from frozen schemas."""
    torch = pytest.importorskip("torch")
    path = tmp_path / "dense_finetuned.pt"
    alignment = RegionTextAlignmentSettings(
        text_dimension=4,
        region_dimension=4,
        hidden_dimension=4,
    )
    dense = DensePatchAlignmentSettings(
        decoder_hidden_dimension=8,
        decoder_branch_dimension=8,
        decoder_dilations=(1,),
    )
    projection = build_text_projection(alignment)
    decoder = build_multiscale_patch_decoder(4, dense)
    torch.save(
        {
            "schema_version": 4,
            "alignment_settings": alignment.model_dump(mode="json"),
            "dense_settings": dense.model_dump(mode="json"),
            "projection_state_dict": projection.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "logit_scale": 1.0,
            "logit_bias": 0.0,
            "base_encoders_frozen": False,
            "dinov2_model": "dinov2_vits14",
            "text_model": "BAAI/bge-m3",
            "model_type": "multiscale_decoder",
            "dinov2_input_size": 728,
            "dinov2_unfrozen_block_count": 2,
            "dinov2_trainable_state_dict": {"blocks.10.weight": torch.ones(1)},
        },
        path,
    )
    monkeypatch.setattr(
        "fashion_semantic_parser.service.dense_patch_alignment.resolve_project_path",
        lambda _: path,
    )

    checkpoint = load_dense_patch_alignment_checkpoint(path, device="cpu")

    assert checkpoint.dinov2_unfrozen_block_count == 2
    assert checkpoint.dinov2_trainable_state_dict is not None
    assert set(checkpoint.dinov2_trainable_state_dict) == {"blocks.10.weight"}


def test_apply_finetuned_checkpoint_is_noop_for_frozen_schema() -> None:
    """Legacy checkpoints must not trigger a partial backbone restore."""

    class FakeEncoder:
        """Record unexpected restore calls."""

        def load_finetuned_state_dict(self, *_: object, **__: object) -> None:
            """Fail if the frozen checkpoint tries to mutate DINOv2."""
            raise AssertionError("Frozen checkpoint attempted DINOv2 restore.")

    checkpoint = SimpleNamespace(
        dinov2_trainable_state_dict=None,
        dinov2_unfrozen_block_count=0,
    )

    apply_finetuned_dinov2_checkpoint(FakeEncoder(), checkpoint)
