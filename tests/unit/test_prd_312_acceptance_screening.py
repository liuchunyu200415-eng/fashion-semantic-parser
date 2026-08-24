"""Tests for the PRD 3.1.2 holdout image-screening workflow."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fashion_semantic_parser.dao.localization.prd_312_acceptance import (
    Prd312AcceptanceContract,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_holdout import (
    Prd312AcceptanceHoldoutImage,
    Prd312AcceptanceHoldoutInventory,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_planning import (
    build_prd_312_acceptance_review_plan,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_screening import (
    Prd312AcceptanceScreeningRecord,
    Prd312AcceptanceScreeningSources,
    build_prd_312_acceptance_screening_plan,
    summarize_prd_312_acceptance_screening,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_screening_csv import (
    import_prd_312_acceptance_screening_csv,
    write_prd_312_acceptance_screening_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _review_plan():
    """Return the deterministic review plan from the tracked contract."""
    contract = Prd312AcceptanceContract.model_validate_json(
        (PROJECT_ROOT / "configs/prd_312_acceptance_contract.json").read_text(
            encoding="utf-8"
        )
    )
    return build_prd_312_acceptance_review_plan(contract)


def _inventory() -> Prd312AcceptanceHoldoutInventory:
    """Return three candidate images and one mechanically excluded image."""
    images = [
        Prd312AcceptanceHoldoutImage(
            image_path=f"data/raw/fashionpedia/test/{index}.jpg",
            image_sha256=f"{index:064x}",
            file_size_bytes=100 + index,
            width=640,
            height=480,
            status="candidate",
        )
        for index in range(1, 4)
    ]
    images.append(
        Prd312AcceptanceHoldoutImage(
            image_path="data/raw/fashionpedia/test/excluded.jpg",
            image_sha256=f"{4:064x}",
            file_size_bytes=104,
            width=640,
            height=480,
            status="excluded",
            exclusion_reasons=["development_use_exclusion"],
        )
    )
    return Prd312AcceptanceHoldoutInventory(
        generated_at="2026-08-24T10:00:00+08:00",
        source_list_sha256="a" * 64,
        exclusion_list_sha256="b" * 64,
        images=images,
    )


def _screening_plan():
    """Return a pending screening plan bound to deterministic fake hashes."""
    return build_prd_312_acceptance_screening_plan(
        inventory=_inventory(),
        review_plan=_review_plan(),
        sources=Prd312AcceptanceScreeningSources(
            candidate_inventory_path=(
                "data/benchmarks/localization/"
                + "prd_312_acceptance_holdout_candidates_v1.json"
            ),
            candidate_inventory_sha256="c" * 64,
            review_plan_path=(
                "data/benchmarks/localization/" + "prd_312_acceptance_review_v1.json"
            ),
            review_plan_sha256="d" * 64,
        ),
        generated_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
    )


def test_screening_plan_contains_only_candidates_without_model_selection() -> None:
    """Excluded images stay out and every accepted candidate starts pending."""
    plan = _screening_plan()

    assert len(plan.records) == 3
    assert all(record.screening_status == "pending" for record in plan.records)
    assert all("excluded" not in record.image_path for record in plan.records)
    assert plan.model_assisted_selection_prohibited is True
    assert plan.formal_holdout_ready is False
    assert plan.required_region_case_counts == {
        "collar": 125,
        "cuff": 125,
        "hem": 125,
        "pocket": 125,
        "shoulder": 125,
        "waist": 125,
        "pattern": 125,
        "decoration": 125,
    }


def test_screening_record_rejects_incomplete_or_inconsistent_evidence() -> None:
    """Status transitions require reviewer evidence and compatible counts."""
    base = {
        "image_path": "data/image.jpg",
        "image_sha256": "0" * 64,
        "width": 640,
        "height": 480,
    }

    with pytest.raises(ValidationError, match="requires reviewer evidence"):
        Prd312AcceptanceScreeningRecord(
            **base,
            screening_status="eligible",
            target_region_instance_counts={"collar": 1},
        )
    with pytest.raises(ValidationError, match="exceeds collar count"):
        Prd312AcceptanceScreeningRecord(
            **base,
            screening_status="eligible",
            target_region_instance_counts={"collar": 1},
            weak_target_label_instance_counts={"neckline": 2},
            screened_by="reviewer",
            screened_at="2026-08-24T10:00:00+08:00",
        )
    with pytest.raises(ValidationError, match="require a rejection reason"):
        Prd312AcceptanceScreeningRecord(
            **base,
            screening_status="rejected",
            screened_by="reviewer",
            screened_at="2026-08-24T10:00:00+08:00",
        )


def test_screening_csv_round_trip_and_summary(tmp_path: Path) -> None:
    """Human CSV decisions import into strict records and bounded deficits.

    Args:
        tmp_path: Isolated path for the editable CSV fixture.
    """
    plan = _screening_plan()
    csv_path = tmp_path / "screening.csv"
    write_prd_312_acceptance_screening_csv(csv_path, plan)
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0].update(
        {
            "screening_status": "eligible",
            "collar_count": "2",
            "neckline_count": "1",
            "screened_by": "reviewer-a",
            "screened_at": "2026-08-24T10:00:00+08:00",
        }
    )
    rows[1].update(
        {
            "screening_status": "rejected",
            "rejection_reason": "none of the locked regions is visible",
            "screened_by": "reviewer-a",
            "screened_at": "2026-08-24T10:01:00+08:00",
        }
    )
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    imported = import_prd_312_acceptance_screening_csv(plan, csv_path)
    summary = summarize_prd_312_acceptance_screening(imported)

    assert imported.records[0].target_region_instance_counts == {"collar": 2}
    assert imported.records[0].weak_target_label_instance_counts == {"neckline": 1}
    assert summary.screening_status_counts == {
        "eligible": 1,
        "pending": 1,
        "rejected": 1,
    }
    assert summary.multi_target_region_case_capacity == 1
    assert summary.region_supply_deficits["collar"] == 124
    assert summary.minimum_supply_ready is False
    assert summary.formal_holdout_ready is False


def test_screening_csv_rejects_identity_tampering(tmp_path: Path) -> None:
    """Human screening cannot silently replace the frozen image hash.

    Args:
        tmp_path: Isolated path for the tampered CSV fixture.
    """
    plan = _screening_plan()
    csv_path = tmp_path / "screening.csv"
    write_prd_312_acceptance_screening_csv(csv_path, plan)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    first = lines[1].split(",")
    first[header.index("image_sha256")] = "f" * 64
    lines[1] = ",".join(first)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable column image_sha256"):
        import_prd_312_acceptance_screening_csv(plan, csv_path)


def test_screening_csv_rejects_missing_rows(tmp_path: Path) -> None:
    """Every frozen candidate must remain represented in the edited CSV.

    Args:
        tmp_path: Isolated path for the incomplete CSV fixture.
    """
    plan = _screening_plan()
    csv_path = tmp_path / "screening.csv"
    write_prd_312_acceptance_screening_csv(csv_path, plan)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    csv_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row count differs"):
        import_prd_312_acceptance_screening_csv(plan, csv_path)


def test_screening_plan_json_is_serializable() -> None:
    """The worklist stays portable as a plain UTF-8 JSON artifact."""
    payload = json.loads(_screening_plan().model_dump_json())

    assert payload["schema_version"] == 1
    assert payload["records"][0]["image_path"].endswith("1.jpg")
