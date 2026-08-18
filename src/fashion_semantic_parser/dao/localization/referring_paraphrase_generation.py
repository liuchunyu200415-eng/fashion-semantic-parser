"""Resume-safe execution of audited referring-expression paraphrase jobs."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from fashion_semantic_parser.dao.localization.referring_paraphrase import (
    ReferringParaphraseJob,
    ReferringParaphraseResult,
)


class ReferringParaphraseGenerator(Protocol):
    """Model-independent boundary used by the offline job runner."""

    # The runner needs exactly one vendor-independent generation operation.
    # pylint: disable=too-few-public-methods

    def paraphrase(self, job: ReferringParaphraseJob) -> ReferringParaphraseResult:
        """Generate one auditable unreviewed result."""


class ReferringParaphraseFailure(BaseModel):
    """One retained model or validation failure for later retry."""

    source_sample_id: str
    error_type: str
    message: str


class ReferringParaphraseGenerationSummary(BaseModel):
    """Counts from one bounded, resume-safe generation invocation."""

    job_count: int
    selected_job_count: int
    preexisting_result_count: int
    generated_result_count: int
    failed_result_count: int
    total_result_count: int
    remaining_job_count: int
    output_path: str
    failure_path: str


# Audit paths and checkpoint controls stay explicit at this offline boundary.
# pylint: disable-next=too-many-arguments,too-many-locals
def run_referring_paraphrase_jobs(
    *,
    job_path: Path,
    output_path: Path,
    failure_path: Path,
    generator: ReferringParaphraseGenerator,
    limit: int | None = None,
    checkpoint_every: int = 20,
) -> ReferringParaphraseGenerationSummary:
    """Generate selected missing jobs and atomically checkpoint valid results."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least one when provided.")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least one.")
    jobs = _read_jobs(job_path)
    jobs_by_id = {job.source_sample_id: job for job in jobs}
    if len(jobs_by_id) != len(jobs):
        raise ValueError("Paraphrase jobs contain duplicate source_sample_id.")
    existing_results = _read_existing_results(output_path)
    _validate_existing_results(existing_results, jobs_by_id)
    existing_ids = {result.source_sample_id for result in existing_results}
    pending_jobs = [job for job in jobs if job.source_sample_id not in existing_ids]
    selected_jobs = pending_jobs[:limit] if limit is not None else pending_jobs

    results = list(existing_results)
    failures: list[ReferringParaphraseFailure] = []
    generated_count = 0
    for index, job in enumerate(selected_jobs, start=1):
        try:
            result = generator.paraphrase(job)
            _validate_generated_result(result, job)
            results.append(result)
            generated_count += 1
            print(
                f"[{index}/{len(selected_jobs)}] id={job.source_sample_id} "
                f"paraphrases={len(result.paraphrases)}"
            )
        # Remote-code model failures are not restricted to built-in exception
        # classes. Retain each failure rather than dropping a training row.
        # pylint: disable-next=broad-exception-caught
        except Exception as error:
            failures.append(
                ReferringParaphraseFailure(
                    source_sample_id=job.source_sample_id,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            print(
                f"[{index}/{len(selected_jobs)}] id={job.source_sample_id} "
                f"error={type(error).__name__}: {error}"
            )
        if index % checkpoint_every == 0:
            _write_models_atomic(output_path, results)
            _write_models_atomic(failure_path, failures)
    _write_models_atomic(output_path, results)
    _write_models_atomic(failure_path, failures)

    total_result_count = len(results)
    return ReferringParaphraseGenerationSummary(
        job_count=len(jobs),
        selected_job_count=len(selected_jobs),
        preexisting_result_count=len(existing_results),
        generated_result_count=generated_count,
        failed_result_count=len(failures),
        total_result_count=total_result_count,
        remaining_job_count=len(jobs) - total_result_count,
        output_path=str(output_path),
        failure_path=str(failure_path),
    )


def _read_jobs(path: Path) -> list[ReferringParaphraseJob]:
    """Read non-empty, blank-free Qwen-VL generation jobs."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Paraphrase job index is empty: {path}")
    if any(not line.strip() for line in lines):
        raise ValueError(f"Paraphrase job index contains a blank record: {path}")
    return [ReferringParaphraseJob.model_validate_json(line) for line in lines]


def _read_existing_results(path: Path) -> list[ReferringParaphraseResult]:
    """Read a prior checkpoint when resuming, rejecting partial JSONL rows."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f"Paraphrase results contain a blank record: {path}")
    return [ReferringParaphraseResult.model_validate_json(line) for line in lines]


def _validate_existing_results(
    results: list[ReferringParaphraseResult],
    jobs_by_id: dict[str, ReferringParaphraseJob],
) -> None:
    """Ensure a resume checkpoint still belongs to the immutable job file."""
    result_ids = [result.source_sample_id for result in results]
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("Paraphrase results contain duplicate source_sample_id.")
    for result in results:
        job = jobs_by_id.get(result.source_sample_id)
        if job is None:
            raise ValueError(
                f"Paraphrase result references unknown job: {result.source_sample_id}"
            )
        _validate_generated_result(result, job)


def _validate_generated_result(
    result: ReferringParaphraseResult,
    job: ReferringParaphraseJob,
) -> None:
    """Reject provenance, language, review, or result-count drift."""
    if result.source_sample_id != job.source_sample_id:
        raise ValueError("Generated result source_sample_id differs from its job.")
    if result.source_fingerprint != job.source_fingerprint:
        raise ValueError("Generated result source_fingerprint differs from its job.")
    if result.language != job.language:
        raise ValueError("Generated result language differs from its job.")
    if result.review_status != "unreviewed":
        raise ValueError("Model generation cannot mark its own output as reviewed.")
    if len(result.paraphrases) != job.requested_paraphrase_count:
        raise ValueError("Generated paraphrase count differs from its job.")


def _write_models_atomic(path: Path, rows: Sequence[BaseModel]) -> None:
    """Replace a JSONL checkpoint only after all rows serialize."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(row.model_dump_json() + "\n")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
