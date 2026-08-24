"""CSV exchange format for human PRD 3.1.2 image screening."""

import csv
from pathlib import Path

from fashion_semantic_parser.dao.localization.prd_312_acceptance_screening import (
    TARGET_REGIONS,
    WEAK_TARGET_LABELS,
    Prd312AcceptanceScreeningPlan,
    Prd312AcceptanceScreeningRecord,
    WeakTargetLabel,
)

_IMMUTABLE_COLUMNS = ("image_path", "image_sha256", "width", "height")
_STATUS_COLUMNS = (
    "screening_status",
    "rejection_reason",
    "screened_by",
    "screened_at",
    "notes",
)
SCREENING_CSV_COLUMNS = (
    *_IMMUTABLE_COLUMNS,
    "screening_status",
    *(f"{region}_count" for region in TARGET_REGIONS),
    "zipper_count",
    "rivet_count",
    "neckline_count",
    "pocket_label_count",
    *_STATUS_COLUMNS[1:],
)


def write_prd_312_acceptance_screening_csv(
    path: Path,
    plan: Prd312AcceptanceScreeningPlan,
) -> None:
    """Atomically write one human-editable screening CSV.

    Args:
        path: Destination CSV path.
        plan: Pending or partially reviewed screening plan.

    Raises:
        OSError: If the CSV cannot be written or published.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SCREENING_CSV_COLUMNS)
            writer.writeheader()
            for record in plan.records:
                writer.writerow(_screening_csv_row(record))
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def import_prd_312_acceptance_screening_csv(
    plan: Prd312AcceptanceScreeningPlan,
    csv_path: Path,
) -> Prd312AcceptanceScreeningPlan:
    """Import human decisions while protecting candidate identity columns.

    Args:
        plan: Original screening worklist.
        csv_path: Human-edited screening CSV.

    Returns:
        Fully validated plan containing imported decisions.

    Raises:
        ValueError: If rows are missing, duplicated, malformed, or tampered.
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != SCREENING_CSV_COLUMNS:
            raise ValueError("Screening CSV columns differ from the frozen template.")
        rows = list(reader)
    expected = {record.image_path: record for record in plan.records}
    if len(rows) != len(expected):
        raise ValueError("Screening CSV row count differs from the frozen worklist.")
    imported: list[Prd312AcceptanceScreeningRecord] = []
    seen: set[str] = set()
    for row in rows:
        image_path = row["image_path"]
        if image_path in seen:
            raise ValueError(f"Screening CSV repeats image path: {image_path}")
        seen.add(image_path)
        original = expected.get(image_path)
        if original is None:
            raise ValueError(f"Screening CSV contains an unknown image: {image_path}")
        _validate_immutable_columns(row, original)
        imported.append(_screening_record_from_csv(row, original))
    missing = sorted(set(expected).difference(seen))
    if missing:
        raise ValueError("Screening CSV is missing images: " + ", ".join(missing[:5]))
    payload = plan.model_dump(mode="json")
    payload["records"] = [record.model_dump(mode="json") for record in imported]
    return Prd312AcceptanceScreeningPlan.model_validate(payload)


def _screening_csv_row(record: Prd312AcceptanceScreeningRecord) -> dict[str, object]:
    """Flatten one record into the stable CSV schema."""
    row: dict[str, object] = {
        "image_path": record.image_path,
        "image_sha256": record.image_sha256,
        "width": record.width,
        "height": record.height,
        "screening_status": record.screening_status,
        "rejection_reason": record.rejection_reason or "",
        "screened_by": record.screened_by or "",
        "screened_at": record.screened_at.isoformat() if record.screened_at else "",
        "notes": record.notes or "",
    }
    row.update(
        {
            f"{region}_count": record.target_region_instance_counts.get(region, 0)
            for region in TARGET_REGIONS
        }
    )
    row.update(
        {
            _weak_label_column(label): record.weak_target_label_instance_counts.get(
                label, 0
            )
            for label in WEAK_TARGET_LABELS
        }
    )
    return row


def _screening_record_from_csv(
    row: dict[str, str],
    original: Prd312AcceptanceScreeningRecord,
) -> Prd312AcceptanceScreeningRecord:
    """Parse one human-edited CSV row into the strict record schema."""
    region_counts = {
        region: count
        for region in TARGET_REGIONS
        if (count := _parse_count(row[f"{region}_count"], f"{region}_count")) > 0
    }
    weak_counts = {
        label: count
        for label in WEAK_TARGET_LABELS
        if (
            count := _parse_count(
                row[_weak_label_column(label)], _weak_label_column(label)
            )
        )
        > 0
    }
    return Prd312AcceptanceScreeningRecord.model_validate(
        {
            "image_path": original.image_path,
            "image_sha256": original.image_sha256,
            "width": original.width,
            "height": original.height,
            "screening_status": row["screening_status"].strip(),
            "target_region_instance_counts": region_counts,
            "weak_target_label_instance_counts": weak_counts,
            "rejection_reason": row["rejection_reason"].strip() or None,
            "screened_by": row["screened_by"].strip() or None,
            "screened_at": row["screened_at"].strip() or None,
            "notes": row["notes"].strip() or None,
        }
    )


def _validate_immutable_columns(
    row: dict[str, str],
    original: Prd312AcceptanceScreeningRecord,
) -> None:
    """Reject edits to identity and decoded-dimension columns."""
    expected = {
        "image_path": original.image_path,
        "image_sha256": original.image_sha256,
        "width": str(original.width),
        "height": str(original.height),
    }
    for column in _IMMUTABLE_COLUMNS:
        if row[column].strip() != expected[column]:
            message = "Screening CSV changed immutable column {}: {}".format(
                column,
                original.image_path,
            )
            raise ValueError(message)


def _parse_count(value: str, column: str) -> int:
    """Parse one non-negative integer CSV count."""
    stripped = value.strip()
    if not stripped:
        return 0
    try:
        parsed = int(stripped)
    except ValueError as error:
        raise ValueError(f"Screening count must be an integer: {column}") from error
    if parsed < 0:
        raise ValueError(f"Screening count cannot be negative: {column}")
    return parsed


def _weak_label_column(label: WeakTargetLabel) -> str:
    """Return the unambiguous CSV column for one weak target label."""
    return "pocket_label_count" if label == "pocket" else f"{label}_count"
