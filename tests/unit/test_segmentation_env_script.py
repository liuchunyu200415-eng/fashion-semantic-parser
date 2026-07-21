"""Tests for segmentation environment diagnostics."""

from pathlib import Path

from scripts.check_segmentation_env import (
    _detectron2_arch_is_compatible,
    _directory_file_status,
    _recommendations,
)


def test_directory_file_status_reports_empty_symlink_target(tmp_path: Path) -> None:
    """A migrated link with an empty target must not look training-ready."""
    target = tmp_path / "data-volume"
    target.mkdir()
    link = tmp_path / "train"
    link.symlink_to(target, target_is_directory=True)

    assert _directory_file_status(link) == {
        "exists": True,
        "valid": True,
        "is_symlink": True,
        "file_count": 0,
    }


def test_detectron2_architecture_check_rejects_previous_gpu_build() -> None:
    """An sm_89 extension cannot run CUDA kernels on an sm_86 GPU."""
    assert not _detectron2_arch_is_compatible(
        {"compute_capability": "8.6"},
        {"cuda_arch_flags": "8.9"},
    )
    assert _detectron2_arch_is_compatible(
        {"compute_capability": "8.6"},
        {"cuda_arch_flags": "8.0, 8.6"},
    )


def test_recommendations_report_missing_fashionpedia_and_arch_mismatch() -> None:
    """Environment output should explain both cloned-instance blockers."""
    datasets = {
        "train": {"exists": True, "valid": True, "image_count": 100},
        "validation": {"exists": True, "valid": True, "image_count": 100},
        "fashionpedia_train": {"exists": False, "is_symlink": True},
        "fashionpedia_validation": {"exists": True, "valid": True},
        "fashionpedia_train_images": {
            "exists": True,
            "valid": True,
            "file_count": 0,
        },
        "fashionpedia_validation_images": {
            "exists": True,
            "valid": True,
            "file_count": 3200,
        },
    }

    recommendations = _recommendations(
        datasets,
        {
            "installed": True,
            "cuda_available": True,
            "compute_capability": "8.6",
        },
        {"installed": True, "cuda_arch_flags": "8.9"},
        {"installed": True},
    )

    assert any("current GPU (8.6)" in row for row in recommendations)
    assert any("fashionpedia_train COCO" in row for row in recommendations)
    assert any("contains no files" in row for row in recommendations)
