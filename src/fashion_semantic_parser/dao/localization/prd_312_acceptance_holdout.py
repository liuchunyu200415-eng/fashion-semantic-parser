"""Auditable image inventory for the independent PRD 3.1.2 holdout."""

# OpenCV exposes these extension-generated attributes only at runtime.
# pylint: disable=no-member

import hashlib
import os
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
from pydantic import BaseModel, Field, model_validator

HoldoutCandidateStatus = Literal["candidate", "excluded"]


class Prd312AcceptanceHoldoutImage(BaseModel):
    """One decoded image and its immutable content identity."""

    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_size_bytes: int = Field(ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    status: HoldoutCandidateStatus
    exclusion_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status(self) -> "Prd312AcceptanceHoldoutImage":
        """Keep candidate status and exclusion evidence consistent.

        Returns:
            The validated image inventory record.

        Raises:
            ValueError: If status, dimensions, or exclusion reasons conflict.
        """
        if self.status == "candidate":
            if self.exclusion_reasons:
                raise ValueError("Candidate images cannot have exclusion reasons.")
            if self.width is None or self.height is None:
                raise ValueError("Candidate images require decoded dimensions.")
        elif not self.exclusion_reasons:
            raise ValueError("Excluded images require at least one reason.")
        return self


class Prd312AcceptanceHoldoutInventory(BaseModel):
    """Versioned candidate pool that remains unapproved for formal scoring."""

    schema_version: Literal[1] = 1
    name: Literal["prd_312_acceptance_holdout_candidates_v1"] = (
        "prd_312_acceptance_holdout_candidates_v1"
    )
    source_dataset: Literal["fashionpedia"] = "fashionpedia"
    source_partition: Literal["test"] = "test"
    generated_at: datetime
    source_list_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exclusion_list_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    independence_attested: Literal[False] = False
    formal_holdout_ready: Literal[False] = False
    images: list[Prd312AcceptanceHoldoutImage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> "Prd312AcceptanceHoldoutInventory":
        """Reject duplicate paths and timestamps without timezone evidence.

        Returns:
            The validated candidate inventory.

        Raises:
            ValueError: If paths repeat or the generation time lacks a timezone.
        """
        if self.generated_at.utcoffset() is None:
            raise ValueError("Holdout inventory timestamp must include a timezone.")
        paths = [image.image_path for image in self.images]
        if len(paths) != len(set(paths)):
            raise ValueError("Holdout inventory image paths must be unique.")
        return self


class Prd312AcceptanceHoldoutSummary(BaseModel):
    """Bounded audit counters for one candidate-inventory build."""

    source_path_count: int = Field(ge=0)
    unique_source_path_count: int = Field(ge=0)
    duplicate_source_path_count: int = Field(ge=0)
    candidate_image_count: int = Field(ge=0)
    excluded_image_count: int = Field(ge=0)
    explicit_excluded_image_count: int = Field(ge=0)
    duplicate_content_image_count: int = Field(ge=0)
    unreadable_image_count: int = Field(ge=0)
    exclusion_reason_counts: dict[str, int]
    source_list_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exclusion_list_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    independence_attested: Literal[False] = False
    formal_holdout_ready: Literal[False] = False


def prepare_prd_312_acceptance_holdout_inventory(
    *,
    project_root: Path,
    image_paths: Sequence[str],
    excluded_paths: Sequence[str],
    generated_at: datetime,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[Prd312AcceptanceHoldoutInventory, Prd312AcceptanceHoldoutSummary]:
    """Validate, hash, deduplicate, and classify holdout candidate images.

    Args:
        project_root: Repository root containing every listed image.
        image_paths: Candidate image paths, absolute or project-relative.
        excluded_paths: Development-used image paths that must not be selected.
        generated_at: Time at which the mechanical inventory was generated.
        progress_callback: Optional processed/total progress hook.

    Returns:
        Validated candidate inventory and compact audit summary.

    Raises:
        ValueError: If input paths are empty, unsafe, or exclusions are unknown.
        OSError: If an image cannot be read for hashing.
    """
    if not image_paths:
        raise ValueError("Acceptance holdout source list cannot be empty.")
    root = project_root.resolve(strict=True)
    normalized_paths = [
        _project_relative(root, value, must_exist=True) for value in image_paths
    ]
    unique_paths = sorted(set(normalized_paths))
    normalized_exclusions = {
        _project_relative(root, value, must_exist=False) for value in excluded_paths
    }
    unknown_exclusions = sorted(normalized_exclusions.difference(unique_paths))
    if unknown_exclusions:
        raise ValueError(
            "Excluded images are missing from the source list: "
            + ", ".join(unknown_exclusions)
        )
    decoded: list[dict[str, object]] = []
    for index, relative_path in enumerate(unique_paths, start=1):
        absolute_path = root / relative_path
        image_sha256 = _file_sha256(absolute_path)
        image = cv2.imread(str(absolute_path), cv2.IMREAD_COLOR)
        decoded.append(
            {
                "image_path": relative_path,
                "image_sha256": image_sha256,
                "file_size_bytes": absolute_path.stat().st_size,
                "width": None if image is None else int(image.shape[1]),
                "height": None if image is None else int(image.shape[0]),
            }
        )
        if progress_callback is not None:
            progress_callback(index, len(unique_paths))
    records = _classify_images(decoded, normalized_exclusions)
    reason_counts = Counter(
        reason for record in records for reason in record.exclusion_reasons
    )
    source_list_hash = _line_list_sha256(unique_paths)
    exclusion_list_hash = _line_list_sha256(sorted(normalized_exclusions))
    inventory = Prd312AcceptanceHoldoutInventory(
        generated_at=generated_at,
        source_list_sha256=source_list_hash,
        exclusion_list_sha256=exclusion_list_hash,
        images=records,
    )
    summary = Prd312AcceptanceHoldoutSummary(
        source_path_count=len(normalized_paths),
        unique_source_path_count=len(unique_paths),
        duplicate_source_path_count=len(normalized_paths) - len(unique_paths),
        candidate_image_count=sum(record.status == "candidate" for record in records),
        excluded_image_count=sum(record.status == "excluded" for record in records),
        explicit_excluded_image_count=reason_counts["development_use_exclusion"],
        duplicate_content_image_count=reason_counts["duplicate_image_content"],
        unreadable_image_count=reason_counts["unreadable_image"],
        exclusion_reason_counts=dict(sorted(reason_counts.items())),
        source_list_sha256=source_list_hash,
        exclusion_list_sha256=exclusion_list_hash,
    )
    return inventory, summary


def read_path_list(path: Path) -> list[str]:
    """Read non-empty, non-comment paths from one UTF-8 text file.

    Args:
        path: Path-list text file.

    Returns:
        Paths in source order.
    """
    return [
        stripped
        for line in path.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]


def _classify_images(
    decoded: Sequence[dict[str, object]],
    excluded_paths: set[str],
) -> list[Prd312AcceptanceHoldoutImage]:
    """Apply explicit, decode, and duplicate-content exclusions."""
    explicitly_excluded_hashes = {
        str(row["image_sha256"])
        for row in decoded
        if str(row["image_path"]) in excluded_paths
    }
    first_candidate_by_hash: dict[str, str] = {}
    records: list[Prd312AcceptanceHoldoutImage] = []
    for row in decoded:
        image_path = str(row["image_path"])
        image_sha256 = str(row["image_sha256"])
        reasons: list[str] = []
        if image_path in excluded_paths or image_sha256 in explicitly_excluded_hashes:
            reasons.append("development_use_exclusion")
        if row["width"] is None or row["height"] is None:
            reasons.append("unreadable_image")
        if not reasons:
            if image_sha256 in first_candidate_by_hash:
                reasons.append("duplicate_image_content")
            else:
                first_candidate_by_hash[image_sha256] = image_path
        records.append(
            Prd312AcceptanceHoldoutImage.model_validate(
                {
                    **row,
                    "status": "excluded" if reasons else "candidate",
                    "exclusion_reasons": reasons,
                }
            )
        )
    return records


def _project_relative(root: Path, value: str, *, must_exist: bool) -> str:
    """Resolve one lexical project path while allowing mounted data symlinks."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = Path(os.path.abspath(candidate))
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Holdout image escapes the project root: {value}") from error
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"Holdout image does not exist: {value}")
    return relative


def _file_sha256(path: Path) -> str:
    """Hash one image without loading its complete bytes into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_list_sha256(values: Sequence[str]) -> str:
    """Hash a normalized ordered line list, including its final newline."""
    payload = "".join(f"{value}\n" for value in values)
    return hashlib.sha256(payload.encode()).hexdigest()
