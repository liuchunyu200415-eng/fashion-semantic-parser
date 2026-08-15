"""Contract tests for the isolated PRD Detectron2 installer."""

from pathlib import Path

import yaml

from scripts.check_prd_312_detectron2_env import detectron2_runtime_ready

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = PROJECT_ROOT / "scripts/setup_prd_312_detectron2.sh"
ENVIRONMENT_FILE = PROJECT_ROOT / "environment/prd_3_1_2_training.yaml"


def test_detectron2_setup_is_pinned_and_targets_prd_environment() -> None:
    """The installer must not borrow an opaque base-environment package."""
    script = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "fashion-prd-312" in script
    assert "d1e04565d3bec8719335b88be9e9b961bf3ec464" in script
    assert "TORCH_CUDA_ARCH_LIST" in script
    assert "--no-build-isolation" in script
    assert "python -m pip wheel" in script
    assert "--editable" not in script
    assert "--force-reinstall" in script
    assert "setuptools==80.9.0" in script
    assert "import pkg_resources" in script
    assert "torch.utils.cpp_extension" in script
    assert "/root/miniconda3/lib/python3.10/site-packages" not in script
    assert "check_prd_312_detectron2_env.py" in script


def test_detectron2_runtime_dependencies_are_environment_locked() -> None:
    """Repairing the PRD environment must retain Detectron2 dependencies."""
    payload = yaml.safe_load(ENVIRONMENT_FILE.read_text(encoding="utf-8"))
    pip_dependencies = next(
        dependency["pip"]
        for dependency in payload["dependencies"]
        if isinstance(dependency, dict) and "pip" in dependency
    )

    assert "fvcore==0.1.5.post20221221" in pip_dependencies
    assert "iopath==0.1.9" in pip_dependencies
    assert "hydra-core==1.3.2" in pip_dependencies
    assert "setuptools==80.9.0" in pip_dependencies


def test_joint_runtime_readiness_requires_exact_cuda_stack() -> None:
    """A partial CPU or version-only setup cannot pass the final gate."""
    report = {
        "python": "3.10.12",
        "torch": "2.1.2+cu121",
        "cuda_available": True,
        "detectron2": "0.6",
        "detectron2_cuda_arch": ["8.6"],
        "sentence_transformers": "3.0.1",
        "mask2former_importable": True,
        "errors": [],
    }

    assert detectron2_runtime_ready(report)
    report["cuda_available"] = False
    assert not detectron2_runtime_ready(report)
