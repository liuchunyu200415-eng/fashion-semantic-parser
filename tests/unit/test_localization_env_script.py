"""Tests for localization environment diagnostics."""

from pathlib import Path
from types import SimpleNamespace

from scripts.check_localization_env import (
    _path_status,
    _recommendations,
    _transformers_status,
)


def test_path_status_reports_model_size(tmp_path: Path) -> None:
    """Weight diagnostics should distinguish present files from directories."""
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"x" * 1024**2)

    status = _path_status(checkpoint)

    assert status["exists"] is True
    assert status["size_mb"] > 0


def test_recommendations_report_only_missing_runtime_blockers() -> None:
    """The checker should explain data, GPU, model, and import blockers."""
    report = {
        "torch": {"installed": True, "cuda_available": False},
        "paths": {
            "train_coco": {"exists": True, "valid": True},
            "validation_coco": {"exists": False},
            "grounding_dino_repo": {"exists": False},
            "grounding_dino_config": {"exists": False},
            "grounding_dino_weights": {"exists": False},
            "sam_hq_weights": {"exists": False},
        },
        "grounding_dino": {"installed": False},
        "transformers": {
            "installed": True,
            "version": "5.14.1",
            "torch_available": False,
            "bert_model_available": False,
        },
        "detectron2": {"installed": False},
        "sam_hq": {"installed": False},
    }

    recommendations = _recommendations(report)

    assert any("GPU instance" in row for row in recommendations)
    assert any("validation_coco" in row for row in recommendations)
    assert any("GroundingDINO" in row for row in recommendations)
    assert any("transformers==4.35.2" in row for row in recommendations)
    assert any("Detectron2" in row for row in recommendations)
    assert any("segment-anything-hq" in row for row in recommendations)


def test_transformers_status_rejects_backend_without_pytorch(
    monkeypatch,
) -> None:
    """Importable Transformers is not ready when it disables PyTorch models."""
    transformers = SimpleNamespace(
        __version__="5.14.1",
        __file__="/tmp/transformers/__init__.py",
    )
    transformers_utils = SimpleNamespace(is_torch_available=lambda: False)

    def fake_import(module_name: str):
        if module_name == "transformers":
            return transformers
        if module_name == "transformers.utils":
            return transformers_utils
        raise ImportError(module_name)

    monkeypatch.setattr(
        "scripts.check_localization_env.importlib.import_module",
        fake_import,
    )

    status = _transformers_status()

    assert status["installed"] is True
    assert status["torch_available"] is False
    assert status["bert_model_available"] is False
