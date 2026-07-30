"""Application configuration loading."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from fashion_semantic_parser.common.paths import resolve_project_path


class PathSettings(BaseModel):
    """Project-relative paths for data, models, and outputs."""

    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    knowledge_dir: str = "data/knowledge"
    model_dir: str = "models/checkpoints"
    output_dir: str = "outputs"


class ServiceSettings(BaseModel):
    """HTTP service settings."""

    host: str = "0.0.0.0"
    port: int = 8000


class ModelSettings(BaseModel):
    """Model runtime settings."""

    device: str = "cuda"
    precision: str = "fp16"


class SegmentationServiceSettings(BaseModel):
    """Garment segmentation inference service settings."""

    config_path: str = "configs/segmentation_mask2former_deployment.yaml"
    query_auto_subject_roi: bool = True


class LocalizationServiceSettings(BaseModel):
    """Language-guided local-region inference service settings."""

    config_path: str = "configs/localization_grounded_sam_hq.yaml"


class DatasetSettings(BaseModel):
    """Project-relative dataset root paths."""

    fashionai_root: str = "data/raw/fashionai/round1_fashionAI_attributes_test_a"
    deepfashion2_root: str = "data/raw/deepfashion2"
    fashionpedia_root: str = "data/raw/fashionpedia"


class AppSettings(BaseModel):
    """Top-level application settings."""

    name: str = "fashion-semantic-parser"
    env: str = "development"


class Settings(BaseModel):
    """Typed project configuration."""

    app: AppSettings = Field(default_factory=AppSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    segmentation: SegmentationServiceSettings = Field(
        default_factory=SegmentationServiceSettings
    )
    localization: LocalizationServiceSettings = Field(
        default_factory=LocalizationServiceSettings
    )
    datasets: DatasetSettings = Field(default_factory=DatasetSettings)


def load_settings(config_path: str | Path = "configs/app.yaml") -> Settings:
    """Load project settings from a YAML file.

    Args:
        config_path: Project-relative path to the YAML configuration file.

    Returns:
        Parsed settings object.
    """
    resolved_path = resolve_project_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}
    return Settings.model_validate(raw_config)
