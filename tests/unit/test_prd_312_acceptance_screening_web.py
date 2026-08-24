"""Tests for the local PRD 3.1.2 human-screening web workspace."""

from pathlib import Path
from typing import cast

import pytest
from fastapi import HTTPException, Request

from fashion_semantic_parser.api.prd_312_acceptance_screening_web import (
    Prd312AcceptanceScreeningUpdate,
    Prd312AcceptanceScreeningWorkspace,
    create_prd_312_acceptance_screening_app,
)
from fashion_semantic_parser.common.paths import PROJECT_ROOT
from fashion_semantic_parser.dao.localization.prd_312_acceptance_review import (
    write_model_json_atomic,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_screening import (
    Prd312AcceptanceScreeningPlan,
    Prd312AcceptanceScreeningRecord,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_screening_csv import (
    write_prd_312_acceptance_screening_csv,
)


def _screening_plan() -> Prd312AcceptanceScreeningPlan:
    """Return a two-image frozen screening plan."""
    return Prd312AcceptanceScreeningPlan(
        generated_at="2026-08-24T10:00:00+08:00",
        candidate_inventory_path="data/holdout.json",
        candidate_inventory_sha256="a" * 64,
        review_plan_path="data/review.json",
        review_plan_sha256="b" * 64,
        source_list_sha256="c" * 64,
        exclusion_list_sha256="d" * 64,
        required_region_case_counts={
            "collar": 1,
            "cuff": 1,
            "hem": 1,
            "pocket": 1,
            "shoulder": 1,
            "waist": 1,
            "pattern": 1,
            "decoration": 1,
        },
        minimum_weak_label_case_counts={
            "zipper": 1,
            "rivet": 1,
            "neckline": 1,
            "pocket": 1,
        },
        required_multi_target_case_count=1,
        records=[
            Prd312AcceptanceScreeningRecord(
                image_path=f"data/images/{index}.jpg",
                image_sha256=f"{index:064x}",
                width=640,
                height=480,
            )
            for index in (1, 2)
        ],
    )


def _workspace(tmp_path: Path) -> Prd312AcceptanceScreeningWorkspace:
    """Create one CSV-backed workspace with two existing image fixtures.

    Args:
        tmp_path: Isolated repository-like directory supplied by pytest.

    Returns:
        Initialized screening workspace.
    """
    project_root = tmp_path / "project"
    image_root = project_root / "data" / "images"
    image_root.mkdir(parents=True)
    (image_root / "1.jpg").write_bytes(b"first-image")
    (image_root / "2.jpg").write_bytes(b"second-image")
    plan = _screening_plan()
    plan_path = project_root / "data" / "screening.json"
    csv_path = project_root / "outputs" / "screening.csv"
    write_model_json_atomic(plan_path, plan)
    write_prd_312_acceptance_screening_csv(csv_path, plan)
    return Prd312AcceptanceScreeningWorkspace(
        project_root=project_root,
        plan_path=plan_path,
        csv_path=csv_path,
        reviewed_output_path=project_root / "data" / "reviewed.json",
        summary_output_path=project_root / "outputs" / "summary.json",
    )


def test_workspace_atomically_saves_and_recovers_review_progress(
    tmp_path: Path,
) -> None:
    """One valid decision persists to CSV and both derived JSON artifacts.

    Args:
        tmp_path: Isolated repository-like directory supplied by pytest.
    """
    workspace = _workspace(tmp_path)
    update = Prd312AcceptanceScreeningUpdate(
        image_sha256=f"{1:064x}",
        screening_status="eligible",
        target_region_instance_counts={"collar": 2},
        weak_target_label_instance_counts={"neckline": 1},
        screened_by="reviewer-a",
        notes="clear neckline",
    )

    record = workspace.update_record(0, update)
    progress = workspace.progress_payload()

    assert record["screening_status"] == "eligible"
    assert cast(str, record["screened_at"]).endswith("Z")
    assert progress["screened_image_count"] == 1
    assert progress["pending_image_count"] == 1
    assert (tmp_path / "project/data/reviewed.json").is_file()
    assert (tmp_path / "project/outputs/summary.json").is_file()
    recovered = _workspace_from_existing(tmp_path)
    assert recovered.record_payload(0)["screening_status"] == "eligible"


def test_workspace_rejects_identity_change_and_invalid_index(tmp_path: Path) -> None:
    """Web updates cannot replace the frozen image or address unknown rows.

    Args:
        tmp_path: Isolated repository-like directory supplied by pytest.
    """
    workspace = _workspace(tmp_path)
    update = Prd312AcceptanceScreeningUpdate(
        image_sha256="f" * 64,
        screening_status="pending",
    )

    with pytest.raises(ValueError, match="identity has changed"):
        workspace.update_record(0, update)
    with pytest.raises(IndexError, match="out of range"):
        workspace.record_payload(10)


def test_screening_app_requires_token_and_serves_frozen_image(tmp_path: Path) -> None:
    """Remote-style access is authenticated and images come from the plan.

    Args:
        tmp_path: Isolated repository-like directory supplied by pytest.
    """
    workspace = _workspace(tmp_path)
    html_path = tmp_path / "screening.html"
    html_path.write_text("<html>screening-ui</html>", encoding="utf-8")
    app = create_prd_312_acceptance_screening_app(
        workspace=workspace,
        html_path=html_path,
        access_token="secure-screening-token",
    )
    index_endpoint = _endpoint(app, "/", "GET")
    record_endpoint = _endpoint(app, "/api/records/{index_value}", "GET")
    image_endpoint = _endpoint(app, "/api/images/{index_value}", "GET")

    with pytest.raises(HTTPException) as unauthorized:
        index_endpoint(_request())
    with pytest.raises(HTTPException) as wrong_token:
        index_endpoint(_request("wrong"))
    response = index_endpoint(_request("secure-screening-token"))
    record = record_endpoint(0, _request("secure-screening-token"))
    image = image_endpoint(0, _request("secure-screening-token"))

    assert unauthorized.value.status_code == 401
    assert wrong_token.value.status_code == 401
    assert response.status_code == 200
    assert record["image_path"].endswith("1.jpg")
    assert Path(image.path).read_bytes() == b"first-image"


def test_screening_app_validates_decision_before_saving(tmp_path: Path) -> None:
    """An eligible decision without counts returns a bounded validation error.

    Args:
        tmp_path: Isolated repository-like directory supplied by pytest.
    """
    workspace = _workspace(tmp_path)
    html_path = tmp_path / "screening.html"
    html_path.write_text("<html>screening-ui</html>", encoding="utf-8")
    app = create_prd_312_acceptance_screening_app(
        workspace=workspace,
        html_path=html_path,
    )
    save_endpoint = _endpoint(app, "/api/records/{index_value}", "POST")

    update = Prd312AcceptanceScreeningUpdate.model_validate(
        {
            "image_sha256": f"{1:064x}",
            "screening_status": "eligible",
            "screened_by": "reviewer-a",
        }
    )

    with pytest.raises(HTTPException) as invalid:
        save_endpoint(0, update, _request())

    assert invalid.value.status_code == 422
    assert "at least one target region" in invalid.value.detail
    assert workspace.record_payload(0)["screening_status"] == "pending"


def test_screening_html_is_offline_and_contains_required_controls() -> None:
    """The shipped page remains self-contained and exposes review controls."""
    html_path = PROJECT_ROOT / "web" / "prd_312_acceptance_screening.html"
    html = html_path.read_text(encoding="utf-8")

    assert 'id="candidate-image"' in html
    assert 'id="save-next"' in html
    assert "/api/records/" in html
    assert "页面不调用任何定位模型" in html
    assert '<script src="' not in html
    assert "https://" not in html


def _workspace_from_existing(
    tmp_path: Path,
) -> Prd312AcceptanceScreeningWorkspace:
    """Reload the current CSV without replacing its saved human decisions."""
    project_root = tmp_path / "project"
    return Prd312AcceptanceScreeningWorkspace(
        project_root=project_root,
        plan_path=project_root / "data" / "screening.json",
        csv_path=project_root / "outputs" / "screening.csv",
        reviewed_output_path=project_root / "data" / "reviewed-reloaded.json",
        summary_output_path=project_root / "outputs" / "summary-reloaded.json",
    )


def _request(token: str | None = None) -> Request:
    """Create a minimal same-origin HTTP request for direct route tests."""
    query = b"" if token is None else f"token={token}".encode()
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": query,
        }
    )


def _endpoint(app: object, path: str, method: str):
    """Return one FastAPI route endpoint selected by path and method."""
    return next(
        route.endpoint
        for route in app.routes  # type: ignore[attr-defined]
        if getattr(route, "path", None) == path
        and method in getattr(route, "methods", set())
    )
