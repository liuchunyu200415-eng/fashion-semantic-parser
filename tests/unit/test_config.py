"""Tests for application configuration loading."""

from fashion_semantic_parser.config import Settings, load_settings


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


def test_refinement_backend_has_a_validated_geometric_gate() -> None:
    """Deployment config can select the staged DINOv2-to-Mask2Former path."""
    settings = Settings.model_validate(
        {
            "localization": {
                "backend": "dense_mask2former_refinement",
                "refinement_minimum_box_iou": 0.10,
            }
        }
    )

    assert settings.localization.backend == "dense_mask2former_refinement"
    assert settings.localization.refinement_minimum_box_iou == 0.10
