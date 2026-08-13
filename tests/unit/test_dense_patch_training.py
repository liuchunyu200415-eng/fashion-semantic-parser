"""Tests for dense training runtime model dispatch."""

from types import SimpleNamespace
from typing import Any

import pytest

from fashion_semantic_parser.service.dense_patch_training import (
    DenseTrainingRuntime,
    runtime_predictions,
)


def _runtime(
    model_type: str,
    *,
    decoder: object | None = None,
    area_predictor: object | None = None,
) -> DenseTrainingRuntime:
    """Build a minimal runtime for model-dispatch tests."""
    return DenseTrainingRuntime(
        projection=object(),
        decoder=decoder,
        area_predictor=area_predictor,
        model_type=model_type,
        log_scale=object(),
        logit_bias=object(),
        text_tensor=object(),
        cache=object(),
        settings=SimpleNamespace(max_logit_scale=100.0),
        device="cpu",
    )


def test_area_runtime_dispatches_patch_and_area_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Area mode must return both spatial logits and target-area logits.

    Args:
        monkeypatch: Function replacement fixture.
    """
    patch_logits = object()
    area_logits = object()
    monkeypatch.setattr(
        "fashion_semantic_parser.service.dense_patch_training."
        + "multiscale_patch_decoder_logits",
        lambda *_: patch_logits,
    )
    monkeypatch.setattr(
        "fashion_semantic_parser.service.dense_patch_training.query_area_logits",
        lambda *_: area_logits,
    )

    outputs = runtime_predictions(
        _runtime(
            "multiscale_area_decoder",
            decoder=object(),
            area_predictor=object(),
        ),
        object(),
        object(),
    )

    assert outputs == (patch_logits, area_logits)


def test_multiscale_runtime_returns_no_area_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema-two multiscale checkpoints must retain their original behavior.

    Args:
        monkeypatch: Function replacement fixture.
    """
    patch_logits = object()
    monkeypatch.setattr(
        "fashion_semantic_parser.service.dense_patch_training."
        + "multiscale_patch_decoder_logits",
        lambda *_: patch_logits,
    )

    outputs = runtime_predictions(
        _runtime("multiscale_decoder", decoder=object()),
        object(),
        object(),
    )

    assert outputs == (patch_logits, None)


def test_cosine_runtime_retains_legacy_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema-one cosine checkpoints must remain trainable.

    Args:
        monkeypatch: Function replacement fixture.
    """
    patch_logits = object()
    monkeypatch.setattr(
        "fashion_semantic_parser.service.dense_patch_training.dense_patch_logits",
        lambda *args, **kwargs: patch_logits,
    )

    outputs = runtime_predictions(
        _runtime("cosine_calibration"),
        object(),
        object(),
    )

    assert outputs == (patch_logits, None)


@pytest.mark.parametrize(
    ("model_type", "decoder", "area_predictor", "message"),
    [
        ("multiscale_decoder", None, None, "decoder"),
        ("multiscale_area_decoder", object(), None, "area predictor"),
        ("cosine_calibration", None, object(), "inconsistent"),
    ],
)
def test_runtime_rejects_inconsistent_model_components(
    model_type: str,
    decoder: Any,
    area_predictor: Any,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed checkpoint runtime must fail instead of changing semantics.

    Args:
        model_type: Runtime model identifier.
        decoder: Optional decoder sentinel.
        area_predictor: Optional area predictor sentinel.
        message: Expected error fragment.
        monkeypatch: Function replacement fixture.
    """
    monkeypatch.setattr(
        "fashion_semantic_parser.service.dense_patch_training."
        + "multiscale_patch_decoder_logits",
        lambda *_: object(),
    )

    with pytest.raises(RuntimeError, match=message):
        runtime_predictions(
            _runtime(
                model_type,
                decoder=decoder,
                area_predictor=area_predictor,
            ),
            object(),
            object(),
        )
