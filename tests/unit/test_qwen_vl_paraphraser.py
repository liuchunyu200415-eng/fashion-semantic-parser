"""Tests for PRD-aligned Qwen-VL referring-expression paraphrases."""

import json
from pathlib import Path
from typing import Any

import pytest

from fashion_semantic_parser.dao.localization.referring_paraphrase import (
    ReferringParaphraseJob,
)
from fashion_semantic_parser.dao.localization.referring_paraphrase_generation import (
    run_referring_paraphrase_jobs,
)
from fashion_semantic_parser.service.qwen_vl_paraphraser import (
    QwenVlParaphraseSettings,
    QwenVlParaphraser,
    build_qwen_vl_paraphrase_prompt,
    parse_qwen_vl_paraphrases,
)


class _FakeQwenVlModel:
    """Return deterministic official-chat-shaped responses without model assets."""

    # The fake intentionally implements only the official chat surface.
    # pylint: disable=too-few-public-methods

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def chat(
        self,
        tokenizer: Any,
        query: str,
        history: None,
        **kwargs: object,
    ) -> tuple[str, object]:
        """Return the next configured model response."""
        del tokenizer, history, kwargs
        self.prompts.append(query)
        return self.responses.pop(0), None


def test_qwen_prompt_preserves_full_referent_constraints() -> None:
    """The model sees full language, count, dimensions, and target identity."""
    job = _job()

    prompt = build_qwen_vl_paraphrase_prompt(job)

    assert "衣服右侧的口袋" in prompt
    assert "Target part: pocket" in prompt
    assert "Target instance count: 1" in prompt
    assert "basic, spatial" in prompt
    assert "exactly 2" in prompt
    assert "JSON array" in prompt


def test_qwen_paraphraser_outputs_unreviewed_pinned_provenance() -> None:
    """A model can generate text but cannot approve its own supervision."""
    model = _FakeQwenVlModel(['["请定位衣服右边的口袋", "找出衣服右侧口袋"]'])
    paraphraser = QwenVlParaphraser(
        QwenVlParaphraseSettings(retry_count=0),
        tokenizer=object(),
        model=model,
    )

    result = paraphraser.paraphrase(_job())

    assert result.review_status == "unreviewed"
    assert result.generator_model == (
        "Qwen/Qwen-VL-Chat-Int4@55acaf444e9f5adfd47105b875571a23d7f7fa30"
    )
    assert result.source_fingerprint == "a" * 64
    assert len(result.paraphrases) == 2
    assert model.prompts[0].count("衣服右侧的口袋") == 1


def test_qwen_paraphraser_retries_with_stricter_format_instruction() -> None:
    """A deterministic retry changes the prompt instead of repeating a failure."""
    model = _FakeQwenVlModel(
        [
            "Here are the rewrites: not-json",
            '["请定位衣服右边的口袋", "找出衣服右侧口袋"]',
        ]
    )
    paraphraser = QwenVlParaphraser(
        QwenVlParaphraseSettings(retry_count=1),
        tokenizer=object(),
        model=model,
    )

    result = paraphraser.paraphrase(_job())

    assert len(result.paraphrases) == 2
    assert len(model.prompts) == 2
    assert "prior format was invalid" not in model.prompts[0]
    assert "prior format was invalid" in model.prompts[1]


def test_qwen_parser_rejects_count_language_and_source_copy() -> None:
    """Structural drift is rejected before separate human semantic review."""
    with pytest.raises(ValueError, match="Expected 2"):
        parse_qwen_vl_paraphrases(
            '["请定位衣服右边的口袋"]',
            expected_count=2,
            source_query="衣服右侧的口袋",
            language="zh",
        )
    with pytest.raises(ValueError, match="Chinese characters"):
        parse_qwen_vl_paraphrases(
            '["find the right pocket", "locate the pocket on the right"]',
            expected_count=2,
            source_query="衣服右侧的口袋",
            language="zh",
        )
    with pytest.raises(ValueError, match="copied the source"):
        parse_qwen_vl_paraphrases(
            '["衣服右侧的口袋", "请定位衣服右边的口袋"]',
            expected_count=2,
            source_query="衣服右侧的口袋",
            language="zh",
        )


def test_generation_runner_resumes_and_retains_failed_job(tmp_path: Path) -> None:
    """A valid checkpoint is skipped while failures remain explicit and retryable."""
    jobs = [_job(), _job(source_sample_id="sample-2", fingerprint="b" * 64)]
    job_path = tmp_path / "jobs.jsonl"
    _write_models(job_path, jobs)
    output_path = tmp_path / "results.jsonl"
    failure_path = tmp_path / "failures.jsonl"
    first_model = _FakeQwenVlModel(
        [
            '["请定位衣服右边的口袋", "找出衣服右侧口袋"]',
            "not-json",
        ]
    )
    first_generator = QwenVlParaphraser(
        QwenVlParaphraseSettings(retry_count=0),
        tokenizer=object(),
        model=first_model,
    )

    first_summary = run_referring_paraphrase_jobs(
        job_path=job_path,
        output_path=output_path,
        failure_path=failure_path,
        generator=first_generator,
        checkpoint_every=1,
    )

    assert first_summary.generated_result_count == 1
    assert first_summary.failed_result_count == 1
    assert first_summary.remaining_job_count == 1
    retry_model = _FakeQwenVlModel(['["请找到服装右边的口袋", "定位服装右侧的口袋"]'])
    retry_generator = QwenVlParaphraser(
        QwenVlParaphraseSettings(retry_count=0),
        tokenizer=object(),
        model=retry_model,
    )

    retry_summary = run_referring_paraphrase_jobs(
        job_path=job_path,
        output_path=output_path,
        failure_path=failure_path,
        generator=retry_generator,
    )

    assert retry_summary.preexisting_result_count == 1
    assert retry_summary.generated_result_count == 1
    assert retry_summary.failed_result_count == 0
    assert retry_summary.remaining_job_count == 0
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2
    assert failure_path.read_text(encoding="utf-8") == ""


def test_qwen_setup_pins_official_prd_model() -> None:
    """The setup boundary cannot drift to an unlisted paraphrase provider."""
    script_path = Path(__file__).resolve().parents[2] / "scripts"
    setup_source = (script_path / "setup_qwen_vl_paraphrase_model.py").read_text(
        encoding="utf-8"
    )

    assert 'MODEL_NAME = "Qwen/Qwen-VL-Chat-Int4"' in setup_source
    assert "55acaf444e9f5adfd47105b875571a23d7f7fa30" in setup_source
    assert "OpenAI" not in setup_source
    assert "DeepSeek" not in setup_source


def _job(
    *,
    source_sample_id: str = "sample-1",
    fingerprint: str = "a" * 64,
) -> ReferringParaphraseJob:
    return ReferringParaphraseJob(
        source_sample_id=source_sample_id,
        source_fingerprint=fingerprint,
        source_query="衣服右侧的口袋",
        language="zh",
        dimensions=["basic", "spatial"],
        target_label="pocket",
        target_count=1,
        requested_paraphrase_count=2,
        instruction=(
            "Rewrite into exactly 2 Chinese expressions without changing "
            "the referent."
        ),
    )


def _write_models(path: Path, rows: list[ReferringParaphraseJob]) -> None:
    path.write_text(
        "".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
