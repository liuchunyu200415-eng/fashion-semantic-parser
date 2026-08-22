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
    QwenVlParaphraser,
    QwenVlParaphraseSettings,
    build_qwen_vl_paraphrase_prompt,
    collect_qwen_vl_paraphrase_candidates,
    parse_qwen_vl_paraphrases,
    resolve_qwen_vl_model_path,
)


class _FakeQwenVlModel:
    """Return deterministic official-chat-shaped responses without model assets."""

    # The fake intentionally implements only the official chat surface.
    # pylint: disable=too-few-public-methods

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.generation_kwargs: list[dict[str, object]] = []

    def chat(
        self,
        tokenizer: Any,
        query: str,
        history: None,
        **kwargs: object,
    ) -> tuple[str, object]:
        """Return the next configured model response."""
        del tokenizer, history
        self.prompts.append(query)
        self.generation_kwargs.append(kwargs)
        return self.responses.pop(0), None


def test_qwen_prompt_preserves_full_referent_constraints() -> None:
    """The model sees full language, count, dimensions, and target identity."""
    job = _job()

    prompt = build_qwen_vl_paraphrase_prompt(job)

    assert "衣服右侧的口袋" in prompt
    assert "目标部件：pocket" in prompt
    assert "目标实例数量：1" in prompt
    assert "basic, spatial" in prompt
    assert "恰好包含 2 条" in prompt
    assert "JSON 字符串数组" in prompt


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
        ":prd312-semantic-gate-v11"
    )
    assert result.source_fingerprint == "a" * 64
    assert len(result.paraphrases) == 2
    assert model.prompts[0].count("衣服右侧的口袋") == 1
    assert model.generation_kwargs == [
        {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_new_tokens": 256,
        }
    ]


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
    assert "上一次回答无效" not in model.prompts[0]
    assert "上一次回答无效" in model.prompts[1]


def test_qwen_paraphraser_retains_partial_semantically_valid_result() -> None:
    """One valid rewrite survives when retries cannot safely produce two."""
    model = _FakeQwenVlModel(
        [
            '["请定位衣服右边的口袋"]',
            '["衣服右侧的口袋"]',
        ]
    )
    paraphraser = QwenVlParaphraser(
        QwenVlParaphraseSettings(retry_count=1),
        tokenizer=object(),
        model=model,
    )

    result = paraphraser.paraphrase(_job())

    assert result.paraphrases == ["请定位衣服右边的口袋"]
    assert result.generator_model.endswith(":prd312-semantic-gate-v11")


def test_qwen_parser_accepts_mislabeled_fenced_json() -> None:
    """A wrong Markdown language tag cannot hide an otherwise valid JSON array."""
    response = """```css
["A round neckline", "A neckline in a round shape"]
```"""

    result = parse_qwen_vl_paraphrases(
        response,
        expected_count=2,
        source_query="the garment neckline with round shaping",
        language="en",
    )

    assert result == ["A round neckline", "A neckline in a round shape"]


def test_qwen_retry_targets_source_copy() -> None:
    """A copied source candidate receives a specific correction on retry."""
    model = _FakeQwenVlModel(
        [
            '["衣服右侧的口袋", "找出衣服右侧口袋"]',
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
    assert "已接受的候选表达" in model.prompts[1]


def test_qwen_candidate_collection_filters_source_and_duplicates() -> None:
    """Useful candidates survive even when one model response is imperfect."""
    response = """```json
["衣服右侧的口袋", "衣服右边的口袋", "衣服右边的口袋"]
```"""

    candidates = collect_qwen_vl_paraphrase_candidates(
        response,
        source_query="衣服右侧的口袋",
        language="zh",
    )

    assert candidates == ["衣服右边的口袋"]


def test_qwen_candidate_gate_strips_meta_labels_and_requires_direction() -> None:
    """Meta prose is removed while candidates that drop right are rejected."""
    response = """```json
["请求式表达：请指出衣服右侧的口袋", "请指出衣服上的口袋"]
```"""

    candidates = collect_qwen_vl_paraphrase_candidates(
        response,
        source_query="衣服右侧的口袋",
        language="zh",
        job=_job(),
    )

    assert candidates == ["请指出衣服右侧的口袋"]


def test_qwen_candidate_gate_rejects_changed_garment_relation() -> None:
    """A neckline on a dress cannot become an invented neckline style."""
    job = ReferringParaphraseJob(
        source_sample_id="relation-sample",
        source_fingerprint="c" * 64,
        source_query="the neckline on the dress",
        language="en",
        dimensions=["basic", "relation"],
        target_label="neckline",
        target_count=1,
        reference_category="dress",
        requested_paraphrase_count=3,
        instruction="Preserve the dress relation.",
    )

    candidates = collect_qwen_vl_paraphrase_candidates(
        '["A sleek V-neckline", "Locate the neckline on the dress"]',
        source_query=job.source_query,
        language="en",
        job=job,
    )

    assert candidates == ["Locate the neckline on the dress"]


def test_qwen_candidate_gate_disambiguates_top_garment_from_position() -> None:
    """Top garment relations cannot become a pocket at the garment's top."""
    job = ReferringParaphraseJob(
        source_sample_id="top-pocket",
        source_fingerprint="1" * 64,
        source_query="the pocket on the top garment",
        language="en",
        dimensions=["basic", "relation"],
        target_label="pocket",
        target_count=1,
        reference_category="top",
        requested_paraphrase_count=3,
        instruction="Top is the garment category.",
    )

    candidates = collect_qwen_vl_paraphrase_candidates(
        '["the pocket on top of the garment", "Locate the top garment\'s pocket"]',
        source_query=job.source_query,
        language="en",
        job=job,
    )

    assert candidates == ["Locate the top garment's pocket"]


def test_qwen_candidate_gate_rejects_prose_actions_and_repeated_words() -> None:
    """Passing lexical anchors cannot admit prose, actions, or malformed text."""
    job = ReferringParaphraseJob(
        source_sample_id="zipper-sample",
        source_fingerprint="d" * 64,
        source_query="衣服下方的拉链",
        language="zh",
        dimensions=["basic", "spatial"],
        target_label="zipper",
        target_count=1,
        spatial_modifier="lower",
        requested_paraphrase_count=3,
        instruction="Preserve the lower zipper target.",
    )

    candidates = collect_qwen_vl_paraphrase_candidates(
        '["拉开衣服下方的拉链", "请定位衣服下方的拉链", "衣服下方的拉链"]',
        source_query=job.source_query,
        language="zh",
        job=job,
    )

    assert candidates == ["请定位衣服下方的拉链"]


def test_qwen_candidate_gate_rejects_repeated_sentence_fragments() -> None:
    """Repeated n-grams cannot survive merely because anchors are present."""
    job = ReferringParaphraseJob(
        source_sample_id="coat-pocket-repeat",
        source_fingerprint="5" * 64,
        source_query="the pocket on the coat",
        language="en",
        dimensions=["basic", "relation"],
        target_label="pocket",
        target_count=1,
        reference_category="coat",
        requested_paraphrase_count=3,
        instruction="Preserve the coat pocket target.",
    )

    candidates = collect_qwen_vl_paraphrase_candidates(
        '["Pocket on the coat pocket on the coat Pocket on the coat", '
        '"Find the pocket on the coat"]',
        source_query=job.source_query,
        language="en",
        job=job,
    )

    assert candidates == ["Find the pocket on the coat"]


def test_qwen_candidate_gate_rejects_attribute_object_and_confirmation_drift() -> None:
    """Boat stays a neckline attribute and yes/no questions are not localization."""
    job = ReferringParaphraseJob(
        source_sample_id="boat-neckline",
        source_fingerprint="e" * 64,
        source_query="the boat neckline",
        language="en",
        dimensions=["basic", "attribute"],
        target_label="neckline",
        target_count=1,
        attribute_phrase="boat",
        requested_paraphrase_count=3,
        instruction="Preserve boat as the neckline attribute.",
    )

    candidates = collect_qwen_vl_paraphrase_candidates(
        '["neckline on boat", "Is this the boat neckline?", "Find the boat neckline"]',
        source_query=job.source_query,
        language="en",
        job=job,
    )

    assert candidates == ["Find the boat neckline"]


def test_qwen_candidate_gate_rejects_identity_questions() -> None:
    """What-type questions identify a part instead of locating its region."""
    job = ReferringParaphraseJob(
        source_sample_id="dress-neckline-identity",
        source_fingerprint="6" * 64,
        source_query="the neckline on the dress",
        language="en",
        dimensions=["basic", "relation"],
        target_label="neckline",
        target_count=1,
        reference_category="dress",
        requested_paraphrase_count=3,
        instruction="Preserve the dress neckline target.",
    )

    candidates = collect_qwen_vl_paraphrase_candidates(
        '["What is the neckline on the dress?", '
        '"What is the location of the neckline on the dress?", '
        '"Find the neckline on the dress"]',
        source_query=job.source_query,
        language="en",
        job=job,
    )

    assert candidates == ["Find the neckline on the dress"]


def test_qwen_candidate_gate_rejects_indefinite_garment_relation() -> None:
    """A part on this garment cannot drift to a part on any garment."""
    job = ReferringParaphraseJob(
        source_sample_id="pants-rivet-definiteness",
        source_fingerprint="7" * 64,
        source_query="这件裤子上的铆钉",
        language="zh",
        dimensions=["basic", "relation"],
        target_label="rivet",
        target_count=1,
        reference_category="pants",
        requested_paraphrase_count=3,
        instruction="Preserve the specific pants relation.",
    )

    candidates = collect_qwen_vl_paraphrase_candidates(
        '["我需要找到一条裤子上的铆钉", "请找出这条裤子上的铆钉"]',
        source_query=job.source_query,
        language="zh",
        job=job,
    )

    assert candidates == ["请找出这条裤子上的铆钉"]


def test_qwen_candidate_gate_rejects_english_labels_in_chinese_text() -> None:
    """Chinese localization text cannot retain an English model-side label."""
    candidates = collect_qwen_vl_paraphrase_candidates(
        '["定位衣服下方的 rivet 部件", "请定位衣服下方的铆钉"]',
        source_query="衣服下方的铆钉",
        language="zh",
        job=ReferringParaphraseJob(
            source_sample_id="lower-rivet",
            source_fingerprint="f" * 64,
            source_query="衣服下方的铆钉",
            language="zh",
            dimensions=["basic", "spatial"],
            target_label="rivet",
            target_count=1,
            spatial_modifier="lower",
            requested_paraphrase_count=3,
            instruction="Preserve the lower rivet target.",
        ),
    )

    assert candidates == ["请定位衣服下方的铆钉"]


def test_qwen_candidate_gate_rejects_new_ordinal_and_garment_relation() -> None:
    """A rewrite cannot select an indexed instance or invent a garment class."""
    relation_job = ReferringParaphraseJob(
        source_sample_id="pants-rivet",
        source_fingerprint="2" * 64,
        source_query="all rivets on the pants",
        language="en",
        dimensions=["basic", "relation"],
        target_label="rivet",
        target_count=2,
        reference_category="pants",
        requested_paraphrase_count=3,
        instruction="Preserve the pants rivet set.",
    )
    attribute_job = ReferringParaphraseJob(
        source_sample_id="boat-neckline-relation-drift",
        source_fingerprint="3" * 64,
        source_query="the boat neckline",
        language="en",
        dimensions=["basic", "attribute"],
        target_label="neckline",
        target_count=1,
        attribute_phrase="boat",
        requested_paraphrase_count=3,
        instruction="Preserve boat as the neckline attribute.",
    )

    assert collect_qwen_vl_paraphrase_candidates(
        '["Where is the second rivet on the pants?", '
        '"Find the rivet on the pants", "Find all rivets on the pants"]',
        source_query=relation_job.source_query,
        language="en",
        job=relation_job,
    ) == ["Find all rivets on the pants"]
    assert collect_qwen_vl_paraphrase_candidates(
        '["boat neckline top", "boat neckline shirt", "Find the boat neckline"]',
        source_query=attribute_job.source_query,
        language="en",
        job=attribute_job,
    ) == ["Find the boat neckline"]


def test_qwen_candidate_gate_preserves_english_target_count() -> None:
    """A multi-pocket union cannot shrink to one indefinite pocket."""
    job = ReferringParaphraseJob(
        source_sample_id="coat-pockets",
        source_fingerprint="8" * 64,
        source_query="all pockets on the coat",
        language="en",
        dimensions=["basic", "relation"],
        target_label="pocket",
        target_count=2,
        reference_category="coat",
        requested_paraphrase_count=3,
        instruction="Preserve all coat pockets.",
    )

    response = json.dumps(
        [
            "A pocket on the coat",
            "Find the pocket on the coat",
            "Find all pockets on the coat",
            "Where are the coat pockets?",
            "What are the two pockets on the coat?",
        ]
    )
    candidates = collect_qwen_vl_paraphrase_candidates(
        response,
        source_query=job.source_query,
        language="en",
        job=job,
    )

    assert candidates == [
        "Find all pockets on the coat",
        "Where are the coat pockets?",
    ]


def test_qwen_candidate_gate_preserves_chinese_target_count() -> None:
    """Chinese multi-target rewrites must keep an explicit all-instance marker."""
    job = ReferringParaphraseJob(
        source_sample_id="jacket-pockets",
        source_fingerprint="9" * 64,
        source_query="这件夹克上的所有口袋",
        language="zh",
        dimensions=["basic", "relation"],
        target_label="pocket",
        target_count=2,
        reference_category="jacket",
        requested_paraphrase_count=3,
        instruction="保留夹克上的全部口袋。",
    )

    candidates = collect_qwen_vl_paraphrase_candidates(
        '["这件夹克上的口袋", "请找出这件夹克上的所有口袋", '
        '"这件夹克上的全部口袋在哪里", "哪些是这件夹克上的所有口袋"]',
        source_query=job.source_query,
        language="zh",
        job=job,
    )

    assert candidates == [
        "请找出这件夹克上的所有口袋",
        "这件夹克上的全部口袋在哪里",
    ]


def test_qwen_candidate_gate_rejects_invented_multi_target() -> None:
    """A one-instance source cannot expand to all matching parts."""
    job = ReferringParaphraseJob(
        source_sample_id="single-pocket",
        source_fingerprint="a" * 64,
        source_query="the pocket on the coat",
        language="en",
        dimensions=["basic", "relation"],
        target_label="pocket",
        target_count=1,
        reference_category="coat",
        requested_paraphrase_count=3,
        instruction="Preserve the single coat pocket.",
    )

    candidates = collect_qwen_vl_paraphrase_candidates(
        '["Find all pockets on the coat", "Find the pocket on the coat"]',
        source_query=job.source_query,
        language="en",
        job=job,
    )

    assert candidates == ["Find the pocket on the coat"]


def test_qwen_candidate_gate_rejects_existence_and_quoted_meta_text() -> None:
    """Presence classification and quoted meta formatting are not localization."""
    job = ReferringParaphraseJob(
        source_sample_id="lower-rivet-existence",
        source_fingerprint="4" * 64,
        source_query="衣服下部的铆钉",
        language="zh",
        dimensions=["basic", "spatial"],
        target_label="rivet",
        target_count=1,
        spatial_modifier="lower",
        requested_paraphrase_count=3,
        instruction="Preserve the lower rivet target.",
    )

    candidates = collect_qwen_vl_paraphrase_candidates(
        '["衣服的下部有铆钉", "衣服的下部有没有铆钉？", '
        '"定位 \'衣服下部的铆钉\'", "请定位衣服下部的铆钉"]',
        source_query=job.source_query,
        language="zh",
        job=job,
    )

    assert candidates == ["请定位衣服下部的铆钉"]


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
    assert first_summary.generated_paraphrase_count == 2
    assert first_summary.partial_generated_result_count == 0
    assert first_summary.failed_result_count == 1
    assert first_summary.total_paraphrase_count == 2
    assert first_summary.remaining_job_count == 1
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert "JSONDecodeError" in failure["message"]
    assert 'response_preview="not-json"' in failure["message"]
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
    assert retry_summary.generated_paraphrase_count == 2
    assert retry_summary.partial_generated_result_count == 0
    assert retry_summary.failed_result_count == 0
    assert retry_summary.total_paraphrase_count == 4
    assert retry_summary.remaining_job_count == 0
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2
    assert failure_path.read_text(encoding="utf-8") == ""


def test_generation_runner_rejects_results_from_an_old_strategy(
    tmp_path: Path,
) -> None:
    """Resume cannot silently mix rows produced by two generation policies."""
    job_path = tmp_path / "jobs.jsonl"
    _write_models(job_path, [_job()])
    output_path = tmp_path / "results.jsonl"
    failure_path = tmp_path / "failures.jsonl"
    first_generator = QwenVlParaphraser(
        QwenVlParaphraseSettings(retry_count=0),
        tokenizer=object(),
        model=_FakeQwenVlModel(['["请定位衣服右边的口袋", "找出衣服右侧口袋"]']),
    )
    run_referring_paraphrase_jobs(
        job_path=job_path,
        output_path=output_path,
        failure_path=failure_path,
        generator=first_generator,
    )
    changed_generator = QwenVlParaphraser(
        QwenVlParaphraseSettings(model_revision="b" * 40, retry_count=0),
        tokenizer=object(),
        model=_FakeQwenVlModel([]),
    )

    with pytest.raises(ValueError, match="model identity"):
        run_referring_paraphrase_jobs(
            job_path=job_path,
            output_path=output_path,
            failure_path=failure_path,
            generator=changed_generator,
        )


def test_generation_runner_rejects_prior_pilot_policy(tmp_path: Path) -> None:
    """The v11 identity gate cannot silently trust rows emitted by v10."""
    job = _job()
    job_path = tmp_path / "jobs.jsonl"
    _write_models(job_path, [job])
    output_path = tmp_path / "results.jsonl"
    failure_path = tmp_path / "failures.jsonl"
    v11_generator = QwenVlParaphraser(
        QwenVlParaphraseSettings(retry_count=0),
        tokenizer=object(),
        model=_FakeQwenVlModel([]),
    )
    v10_result = {
        "source_sample_id": job.source_sample_id,
        "source_fingerprint": job.source_fingerprint,
        "language": job.language,
        "generator_model": v11_generator.generator_identity.replace(
            "semantic-gate-v11",
            "semantic-gate-v10",
        ),
        "review_status": "unreviewed",
        "paraphrases": ["请定位衣服右边的口袋", "找出衣服右侧口袋"],
    }
    output_path.write_text(
        json.dumps(v10_result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model identity"):
        run_referring_paraphrase_jobs(
            job_path=job_path,
            output_path=output_path,
            failure_path=failure_path,
            generator=v11_generator,
        )


def test_qwen_setup_pins_official_prd_model() -> None:
    """The setup boundary cannot drift to an unlisted paraphrase provider."""
    script_path = Path(__file__).resolve().parents[2] / "scripts"
    setup_source = (script_path / "setup_qwen_vl_paraphrase_model.py").read_text(
        encoding="utf-8"
    )
    setup_shell = (script_path / "setup_qwen_vl_paraphrase_model.sh").read_text(
        encoding="utf-8"
    )

    assert 'MODEL_NAME = "Qwen/Qwen-VL-Chat-Int4"' in setup_source
    assert "55acaf444e9f5adfd47105b875571a23d7f7fa30" in setup_source
    assert "OpenAI" not in setup_source
    assert "DeepSeek" not in setup_source
    assert "optimum==1.22.0" in setup_shell
    assert "optimum==1.21.4" not in setup_shell
    assert "transformers==4.44.2" in setup_shell
    assert 'version("optimum")' in setup_shell
    assert "optimum.__version__" not in setup_shell


def test_qwen_sampling_policy_has_bounded_hard_case_retries() -> None:
    """Hard cases receive ten deterministic attempts without unbounded loops."""
    settings = QwenVlParaphraseSettings()

    assert settings.retry_count + 1 == 10
    assert settings.generation_seed == 312
    assert settings.temperature == 0.7
    assert settings.top_p == 0.8


def test_qwen_setup_accepts_absolute_data_volume_path(tmp_path: Path) -> None:
    """Large model assets may be placed outside the capacity-limited repo disk."""
    model_path = tmp_path / "qwen-vl-chat-int4"

    assert resolve_qwen_vl_model_path(str(model_path)) == model_path.resolve()


def test_qwen_runtime_accepts_absolute_data_volume_path(tmp_path: Path) -> None:
    """Inference reaches asset validation instead of rejecting an absolute path."""
    model_path = tmp_path / "qwen-vl-chat-int4"
    model_path.mkdir()
    paraphraser = QwenVlParaphraser(
        QwenVlParaphraseSettings(model_path=str(model_path))
    )

    with pytest.raises(RuntimeError, match="Pinned Qwen-VL assets are missing"):
        paraphraser.load()


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
        spatial_modifier="right",
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
