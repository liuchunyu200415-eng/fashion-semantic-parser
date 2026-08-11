"""Tests for the reproducible PRD 3.1.2 Conda environment file."""

from pathlib import Path

import yaml


def test_environment_installs_project_from_repository_root() -> None:
    """Editable install paths are relative to the environment file directory."""
    environment_path = (
        Path(__file__).resolve().parents[2] / "environment" / "prd_3_1_2_training.yaml"
    )
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    pip_dependencies = next(
        dependency["pip"]
        for dependency in environment["dependencies"]
        if isinstance(dependency, dict) and "pip" in dependency
    )

    assert "-e .." in pip_dependencies
    assert "nodefaults" in environment["channels"]
