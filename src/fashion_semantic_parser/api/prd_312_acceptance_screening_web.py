"""Dedicated web workspace for human PRD 3.1.2 acceptance image screening."""

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from fashion_semantic_parser.dao.localization.prd_312_acceptance import (
    AcceptanceTargetRegion,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_review import (
    write_model_json_atomic,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_screening import (
    Prd312AcceptanceScreeningPlan,
    Prd312AcceptanceScreeningRecord,
    ScreeningStatus,
    WeakTargetLabel,
    load_prd_312_acceptance_screening_plan,
    summarize_prd_312_acceptance_screening,
)
from fashion_semantic_parser.dao.localization.prd_312_acceptance_screening_csv import (
    import_prd_312_acceptance_screening_csv,
    write_prd_312_acceptance_screening_csv,
)


class Prd312AcceptanceScreeningUpdate(BaseModel):
    """Mutable human fields accepted by the screening web service."""

    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    screening_status: ScreeningStatus
    target_region_instance_counts: dict[AcceptanceTargetRegion, int] = Field(
        default_factory=dict
    )
    weak_target_label_instance_counts: dict[WeakTargetLabel, int] = Field(
        default_factory=dict
    )
    rejection_reason: str | None = None
    screened_by: str | None = None
    notes: str | None = None


class Prd312AcceptanceScreeningWorkspace:
    """Thread-safe CSV-backed human screening state."""

    def __init__(
        self,
        project_root: Path,
        plan_path: Path,
        csv_path: Path,
        reviewed_output_path: Path,
        summary_output_path: Path,
    ) -> None:
        """Load and validate the frozen worklist and current CSV checkpoint.

        Args:
            project_root: Repository root containing every candidate image.
            plan_path: Frozen pending screening-plan JSON.
            csv_path: Current human-editable CSV checkpoint.
            reviewed_output_path: Derived validated screening-plan output.
            summary_output_path: Derived progress-summary output.
        """
        self._project_root = project_root.resolve(strict=True)
        self._plan_path = plan_path
        self._csv_path = csv_path
        self._reviewed_output_path = reviewed_output_path
        self._summary_output_path = summary_output_path
        self._lock = threading.Lock()
        frozen = load_prd_312_acceptance_screening_plan(plan_path)
        self._plan = import_prd_312_acceptance_screening_csv(frozen, csv_path)

    def record_payload(self, index: int) -> dict[str, object]:
        """Return one record with stable navigation metadata.

        Args:
            index: Zero-based worklist index.

        Returns:
            JSON-compatible record payload.

        Raises:
            IndexError: If the index is outside the worklist.
        """
        with self._lock:
            record = self._record(index)
            payload = cast(dict[str, object], record.model_dump(mode="json"))
            payload.update({"index": index, "record_count": len(self._plan.records)})
            return payload

    def progress_payload(self) -> dict[str, object]:
        """Return the current validated screening summary.

        Returns:
            JSON-compatible progress summary.
        """
        with self._lock:
            summary = summarize_prd_312_acceptance_screening(self._plan)
            return cast(dict[str, object], summary.model_dump(mode="json"))

    def next_index(self, after: int, status: ScreeningStatus) -> int | None:
        """Find the next matching record with one wrap-around pass.

        Args:
            after: Zero-based index after which the search starts.
            status: Screening status to locate.

        Returns:
            Matching index, or ``None`` when no record has that status.
        """
        with self._lock:
            count = len(self._plan.records)
            for offset in range(1, count + 1):
                index = (after + offset) % count
                if self._plan.records[index].screening_status == status:
                    return index
            return None

    def image_path(self, index: int) -> Path:
        """Resolve one candidate image while preserving project containment.

        Args:
            index: Zero-based worklist index.

        Returns:
            Existing candidate image path, including mounted-data symlinks.

        Raises:
            ValueError: If the lexical path escapes the project root.
            FileNotFoundError: If the candidate image is no longer available.
        """
        with self._lock:
            relative_path = self._record(index).image_path
        candidate = Path(os.path.abspath(self._project_root / relative_path))
        try:
            candidate.relative_to(self._project_root)
        except ValueError as error:
            raise ValueError("Screening image escapes the project root.") from error
        if not candidate.is_file():
            raise FileNotFoundError(f"Screening image is missing: {relative_path}")
        return candidate

    def update_record(
        self,
        index: int,
        update: Prd312AcceptanceScreeningUpdate,
    ) -> dict[str, object]:
        """Validate, atomically checkpoint, and return one human decision.

        Args:
            index: Zero-based worklist index.
            update: Mutable human screening fields.

        Returns:
            Updated record payload.

        Raises:
            ValueError: If identity or screening evidence is invalid.
            IndexError: If the index is outside the worklist.
        """
        with self._lock:
            original = self._record(index)
            if update.image_sha256 != original.image_sha256:
                raise ValueError("Screening update image identity has changed.")
            record = self._updated_record(original, update)
            payload = self._plan.model_dump(mode="json")
            payload["records"][index] = record.model_dump(mode="json")
            updated_plan = Prd312AcceptanceScreeningPlan.model_validate(payload)
            write_prd_312_acceptance_screening_csv(self._csv_path, updated_plan)
            write_model_json_atomic(self._reviewed_output_path, updated_plan)
            summary = summarize_prd_312_acceptance_screening(updated_plan)
            write_model_json_atomic(self._summary_output_path, summary)
            self._plan = updated_plan
            result = cast(dict[str, object], record.model_dump(mode="json"))
            result.update({"index": index, "record_count": len(self._plan.records)})
            return result

    def _record(self, index: int) -> Prd312AcceptanceScreeningRecord:
        """Return one in-range record without acquiring the workspace lock."""
        if index < 0 or index >= len(self._plan.records):
            raise IndexError(f"Screening index is out of range: {index}")
        return self._plan.records[index]

    @staticmethod
    def _updated_record(
        original: Prd312AcceptanceScreeningRecord,
        update: Prd312AcceptanceScreeningUpdate,
    ) -> Prd312AcceptanceScreeningRecord:
        """Build strict record evidence and server-side review timestamp."""
        if update.screening_status == "pending":
            return Prd312AcceptanceScreeningRecord(
                image_path=original.image_path,
                image_sha256=original.image_sha256,
                width=original.width,
                height=original.height,
            )
        payload = {
            "image_path": original.image_path,
            "image_sha256": original.image_sha256,
            "width": original.width,
            "height": original.height,
            **update.model_dump(mode="json"),
            "screened_at": _utc_now(),
        }
        return Prd312AcceptanceScreeningRecord.model_validate(payload)


def create_prd_312_acceptance_screening_app(
    *,
    workspace: Prd312AcceptanceScreeningWorkspace,
    html_path: Path,
    access_token: str | None = None,
) -> FastAPI:
    """Create the dedicated human-screening FastAPI application.

    Args:
        workspace: Validated CSV-backed screening state.
        html_path: Offline HTML user-interface asset.
        access_token: Optional token required for every HTTP request.

    Returns:
        Configured screening-only FastAPI application.

    Raises:
        FileNotFoundError: If the offline HTML asset is unavailable.
    """
    if not html_path.is_file():
        raise FileNotFoundError(f"Screening HTML asset is missing: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    app = FastAPI(title="PRD 3.1.2 Acceptance Screening", docs_url=None)

    def authorize(request: Request) -> None:
        """Reject requests without the configured screening token."""
        if access_token is None:
            return
        provided = request.headers.get("x-screening-token")
        if provided is None:
            provided = request.query_params.get("token")
        if provided != access_token:
            raise HTTPException(status_code=401, detail="Invalid screening token.")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        """Serve the offline human-screening page."""
        authorize(request)
        return HTMLResponse(html)

    @app.get("/api/progress")
    def progress(request: Request) -> dict[str, object]:
        """Return current screening progress and quota deficits."""
        authorize(request)
        return workspace.progress_payload()

    @app.get("/api/records/{index_value}")
    def record(index_value: int, request: Request) -> dict[str, object]:
        """Return one screening record by zero-based index."""
        authorize(request)
        try:
            return workspace.record_payload(index_value)
        except IndexError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/records/{index_value}/next")
    def next_record(
        index_value: int,
        request: Request,
        status: ScreeningStatus = "pending",
    ) -> dict[str, int | None]:
        """Return the next record index matching one screening status."""
        authorize(request)
        return {"index": workspace.next_index(index_value, status)}

    @app.get("/api/images/{index_value}", response_class=FileResponse)
    def image(index_value: int, request: Request) -> FileResponse:
        """Serve one candidate image resolved only from the frozen worklist."""
        authorize(request)
        try:
            return FileResponse(workspace.image_path(index_value))
        except (IndexError, FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/records/{index_value}")
    def save_record(
        index_value: int,
        update: Prd312AcceptanceScreeningUpdate,
        request: Request,
    ) -> dict[str, object]:
        """Validate and atomically save one human screening decision."""
        authorize(request)
        try:
            return workspace.update_record(index_value, update)
        except IndexError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


def _utc_now() -> str:
    """Return one timezone-aware ISO-8601 server timestamp."""
    return datetime.now(timezone.utc).isoformat()
