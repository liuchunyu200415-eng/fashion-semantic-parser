"""Tests for the PRD 3.1.2 acceptance holdout candidate inventory."""

# OpenCV exposes these extension-generated attributes only at runtime.
# pylint: disable=no-member

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from fashion_semantic_parser.dao.localization.prd_312_acceptance_holdout import (
    Prd312AcceptanceHoldoutImage,
    prepare_prd_312_acceptance_holdout_inventory,
    read_path_list,
)


def _write_image(path: Path, value: int) -> None:
    """Write one deterministic 8x8 RGB fixture image."""
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_inventory_hashes_decodes_and_excludes_unsafe_candidates(
    tmp_path: Path,
) -> None:
    """Explicit use, duplicate bytes, and decode failures remain auditable.

    Args:
        tmp_path: Isolated project directory supplied by pytest.
    """
    root = tmp_path / "project"
    root.mkdir()
    _write_image(root / "used.png", 10)
    _write_image(root / "used-copy.png", 10)
    _write_image(root / "candidate.png", 20)
    _write_image(root / "candidate-copy.png", 20)
    (root / "broken.jpg").write_bytes(b"not-an-image")
    paths = [
        "used.png",
        "used-copy.png",
        "candidate.png",
        "candidate-copy.png",
        "broken.jpg",
        "candidate.png",
    ]

    inventory, summary = prepare_prd_312_acceptance_holdout_inventory(
        project_root=root,
        image_paths=paths,
        excluded_paths=["used.png"],
        generated_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
    )

    by_name = {image.image_path: image for image in inventory.images}
    assert summary.source_path_count == 6
    assert summary.unique_source_path_count == 5
    assert summary.duplicate_source_path_count == 1
    assert summary.candidate_image_count == 1
    assert summary.excluded_image_count == 4
    assert summary.explicit_excluded_image_count == 2
    assert summary.duplicate_content_image_count == 1
    assert summary.unreadable_image_count == 1
    assert by_name["candidate-copy.png"].status == "candidate"
    assert by_name["candidate-copy.png"].width == 8
    assert by_name["candidate.png"].exclusion_reasons == ["duplicate_image_content"]
    assert by_name["used-copy.png"].exclusion_reasons == ["development_use_exclusion"]
    assert inventory.independence_attested is False
    assert inventory.formal_holdout_ready is False


def test_unknown_exclusion_fails_closed(tmp_path: Path) -> None:
    """A typo in the development-use exclusion list cannot be ignored.

    Args:
        tmp_path: Isolated project directory supplied by pytest.
    """
    root = tmp_path / "project"
    root.mkdir()
    _write_image(root / "candidate.png", 20)

    with pytest.raises(ValueError, match="missing from the source list"):
        prepare_prd_312_acceptance_holdout_inventory(
            project_root=root,
            image_paths=["candidate.png"],
            excluded_paths=["unknown.png"],
            generated_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
        )


def test_image_outside_project_root_is_rejected(tmp_path: Path) -> None:
    """Holdout records cannot point to untracked paths outside the project.

    Args:
        tmp_path: Isolated project directory supplied by pytest.
    """
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.png"
    _write_image(outside, 30)

    with pytest.raises(ValueError, match="escapes the project root"):
        prepare_prd_312_acceptance_holdout_inventory(
            project_root=root,
            image_paths=[str(outside)],
            excluded_paths=[],
            generated_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
        )


def test_project_mounted_dataset_symlink_is_supported(tmp_path: Path) -> None:
    """AutoDL data volumes may be mounted through a project-local symlink.

    Args:
        tmp_path: Isolated project and mounted-data directories from pytest.
    """
    root = tmp_path / "project"
    root.mkdir()
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    _write_image(mounted / "candidate.png", 40)
    (root / "data").symlink_to(mounted, target_is_directory=True)

    inventory, summary = prepare_prd_312_acceptance_holdout_inventory(
        project_root=root,
        image_paths=["data/candidate.png"],
        excluded_paths=[],
        generated_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
    )

    assert summary.candidate_image_count == 1
    assert inventory.images[0].image_path == "data/candidate.png"


def test_candidate_record_requires_dimensions_and_no_exclusion_reason() -> None:
    """Schema validation keeps candidate and excluded states unambiguous."""
    with pytest.raises(ValidationError, match="decoded dimensions"):
        Prd312AcceptanceHoldoutImage(
            image_path="data/image.jpg",
            image_sha256="0" * 64,
            file_size_bytes=10,
            status="candidate",
        )


def test_path_list_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    """Operational path lists may contain stable explanatory comments.

    Args:
        tmp_path: Isolated directory supplied by pytest.
    """
    path = tmp_path / "paths.txt"
    path.write_text("# comment\n\na.jpg\n b.jpg \n", encoding="utf-8")

    assert read_path_list(path) == ["a.jpg", "b.jpg"]
