"""Tests for application configuration loading."""

from fashion_semantic_parser.config import load_settings


def test_load_settings_reads_default_config() -> None:
    """Default configuration should load from the project configs directory."""
    settings = load_settings()

    assert settings.app.name == "fashion-semantic-parser"
    assert settings.service.port == 8000
