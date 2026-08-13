"""Tests for application configuration loading."""

from fashion_semantic_parser.config import load_settings


def test_load_settings_reads_default_config() -> None:
    """Default configuration should load from the project configs directory."""
    settings = load_settings()

    assert settings.app.name == "fashion-semantic-parser"
    assert settings.service.port == 8000
    assert settings.segmentation.config_path.endswith(
        "segmentation_mask2former_deployment.yaml"
    )
    assert settings.segmentation.query_auto_subject_roi is True
    assert settings.localization.backend == "dense_local_reencoding"
    assert settings.localization.config_path.endswith(
        "localization_mask2former_parts_deployment.yaml"
    )
    assert settings.localization.fallback_config_path.endswith(
        "localization_grounded_sam_hq.yaml"
    )
    assert settings.localization.dense_config_path.endswith(
        "localization_dense_local_reencoding.yaml"
    )
    assert settings.datasets.fashionpedia_root == "data/raw/fashionpedia"
