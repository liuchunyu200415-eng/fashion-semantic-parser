"""Generate auditable referring-expression rewrites with PRD Qwen-VL."""

import json
import re
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import yaml
from pydantic import BaseModel, Field

from fashion_semantic_parser.common.paths import resolve_project_path
from fashion_semantic_parser.dao.localization.referring_paraphrase import (
    ReferringParaphraseJob,
    ReferringParaphraseResult,
)


class QwenVlParaphraseSettings(BaseModel):
    """Pinned, local-only Qwen-VL paraphrase-generation settings."""

    model_name: Literal["Qwen/Qwen-VL-Chat-Int4"] = "Qwen/Qwen-VL-Chat-Int4"
    model_revision: str = Field(
        default="55acaf444e9f5adfd47105b875571a23d7f7fa30",
        pattern=r"^[0-9a-f]{40}$",
    )
    model_path: str = "models/checkpoints/localization/qwen-vl-chat-int4"
    device_map: Literal["cuda", "auto"] = "cuda"
    max_new_tokens: int = Field(default=256, ge=32, le=1024)
    retry_count: int = Field(default=4, ge=0, le=10)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    generation_seed: int = Field(default=312, ge=0)


class QwenVlChatModel(Protocol):
    """Small protocol covering the official Qwen-VL chat entry point."""

    # The protocol deliberately mirrors Qwen-VL's single public inference API.
    # pylint: disable=too-few-public-methods

    def chat(
        self,
        tokenizer: Any,
        query: str,
        history: None,
        **kwargs: object,
    ) -> tuple[str, object]:
        """Return the assistant response and unused conversation history."""


class QwenVlParaphraser:
    """Run the PRD-listed Qwen-VL model without network fallback."""

    def __init__(
        self,
        settings: QwenVlParaphraseSettings,
        *,
        tokenizer: Any | None = None,
        model: QwenVlChatModel | None = None,
    ) -> None:
        self.settings = settings
        self._tokenizer = tokenizer
        self._model = model
        self._torch: Any | None = None

    @property
    def generator_identity(self) -> str:
        """Return immutable model identity stored with every generated row."""
        return (
            f"{self.settings.model_name}@{self.settings.model_revision}"
            ":prd312-sampling-v2"
        )

    def load(self) -> None:
        """Load pinned local Qwen-VL assets and reject runtime downloads."""
        if self._tokenizer is not None and self._model is not None:
            return
        model_path = resolve_qwen_vl_model_path(self.settings.model_path)
        self._validate_local_assets(model_path)
        try:
            # Optional model dependencies are validated only on this route.
            # pylint: disable=import-outside-toplevel
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
                GenerationConfig,
            )
            # pylint: enable=import-outside-toplevel
        except ImportError as error:
            raise RuntimeError(
                "Transformers and Qwen-VL dependencies are required; run "
                "scripts/setup_qwen_vl_paraphrase_model.py."
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map=self.settings.device_map,
            local_files_only=True,
            trust_remote_code=True,
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
        self._tokenizer = tokenizer
        self._model = cast(QwenVlChatModel, model)
        self._torch = torch

    def paraphrase(self, job: ReferringParaphraseJob) -> ReferringParaphraseResult:
        """Generate one unreviewed result while preserving immutable provenance."""
        self.load()
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Qwen-VL paraphrase model did not initialize.")
        prompt = build_qwen_vl_paraphrase_prompt(job)
        last_error: Exception | None = None
        last_response = ""
        collected: list[str] = []
        for attempt in range(self.settings.retry_count + 1):
            try:
                attempt_prompt = prompt
                if attempt:
                    attempt_prompt += _retry_instruction(
                        last_error,
                        collected,
                        job.language,
                    )
                self._seed_generation(job, attempt)
                response, _ = self._model.chat(
                    self._tokenizer,
                    query=attempt_prompt,
                    history=None,
                    do_sample=True,
                    temperature=self.settings.temperature,
                    top_p=self.settings.top_p,
                    max_new_tokens=self.settings.max_new_tokens,
                )
                last_response = response
                candidates = collect_qwen_vl_paraphrase_candidates(
                    response,
                    source_query=job.source_query,
                    language=job.language,
                )
                for candidate in candidates:
                    if candidate not in collected:
                        collected.append(candidate)
                if len(collected) < job.requested_paraphrase_count:
                    last_error = ValueError(
                        f"Collected {len(collected)} distinct non-source "
                        f"paraphrases; need {job.requested_paraphrase_count}."
                    )
                    continue
                return ReferringParaphraseResult(
                    source_sample_id=job.source_sample_id,
                    source_fingerprint=job.source_fingerprint,
                    language=job.language,
                    generator_model=self.generator_identity,
                    review_status="unreviewed",
                    paraphrases=collected[: job.requested_paraphrase_count],
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                last_error = error
        error_name = type(last_error).__name__ if last_error else "unknown"
        error_message = str(last_error) if last_error else "unknown validation error"
        response_preview = json.dumps(last_response[:1000], ensure_ascii=False)
        raise RuntimeError(
            f"Qwen-VL returned no valid paraphrases for {job.source_sample_id!r}; "
            f"validation={error_name}: {error_message}; "
            f"response_preview={response_preview}"
        ) from last_error

    def _seed_generation(self, job: ReferringParaphraseJob, attempt: int) -> None:
        """Make sampled rewrites reproducible per immutable job and attempt."""
        if self._torch is None:
            return
        fingerprint_seed = int(job.source_fingerprint[:16], 16)
        seed = (fingerprint_seed + self.settings.generation_seed + attempt) % (2**31)
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)

    def _validate_local_assets(self, model_path: Path) -> None:
        """Require the pinned revision marker and every official INT4 shard."""
        revision_path = model_path / ".pinned_revision"
        if not revision_path.is_file():
            raise RuntimeError(
                "Pinned Qwen-VL assets are missing; run "
                "scripts/setup_qwen_vl_paraphrase_model.py."
            )
        actual_revision = revision_path.read_text(encoding="utf-8").strip()
        if actual_revision != self.settings.model_revision:
            raise RuntimeError(
                "Qwen-VL revision mismatch: "
                f"expected={self.settings.model_revision} actual={actual_revision}"
            )
        required_files = [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "quantize_config.json",
            "qwen.tiktoken",
        ] + [f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)]
        missing = [name for name in required_files if not (model_path / name).is_file()]
        if missing:
            raise RuntimeError(f"Qwen-VL snapshot is missing files: {missing}")


def load_qwen_vl_paraphrase_settings(
    config_path: str | Path = "configs/localization_qwen_vl_paraphrase.yaml",
) -> QwenVlParaphraseSettings:
    """Load one validated project-relative Qwen-VL configuration."""
    path = resolve_project_path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return cast(
        QwenVlParaphraseSettings,
        QwenVlParaphraseSettings.model_validate(raw),
    )


def resolve_qwen_vl_model_path(value: str | Path) -> Path:
    """Resolve an explicit data-volume path or a project-relative model path."""
    model_path = Path(value).expanduser()
    if model_path.is_absolute():
        return model_path.resolve()
    return Path(resolve_project_path(model_path))


def build_qwen_vl_paraphrase_prompt(job: ReferringParaphraseJob) -> str:
    """Request strict JSON while retaining all referent-changing constraints."""
    dimensions = ", ".join(job.dimensions)
    if job.language == "zh":
        return (
            "你正在为语言引导的服饰区域定位任务生成训练语料。\n"
            f"原始查询：{json.dumps(job.source_query, ensure_ascii=False)}\n"
            f"目标部件：{job.target_label}\n"
            f"目标实例数量：{job.target_count}\n"
            f"必须保留的语义维度：{dimensions}\n"
            f"任务要求：将原始查询改写为 {job.requested_paraphrase_count} 条中文表达。\n"
            "可以改成请求式、疑问式或定位式表达，但不能扩大或缩小目标集合；"
            "必须保留方位、属性、服装关系、单复数含义和参照系。"
            "每条表达都必须与原句有明显文字差异，不能照抄原句。"
            "只输出一个 JSON 字符串数组，不要输出 Markdown、字段名或解释；"
            f"数组必须恰好包含 {job.requested_paraphrase_count} 条互不重复的中文字符串。"
        )
    language_name = "English"
    return (
        "You create training data for language-guided fashion-region "
        "localization.\n"
        f"Source query: {json.dumps(job.source_query, ensure_ascii=False)}\n"
        f"Output language: {language_name}\n"
        f"Target part: {job.target_label}\n"
        f"Target instance count: {job.target_count}\n"
        f"Required meaning dimensions: {dimensions}\n"
        f"Task: {job.instruction}\n"
        "Never broaden or narrow the target set. Keep every direction, "
        "attribute, garment relation, singular/plural meaning, and reference "
        "frame unchanged. Do not copy the source sentence. Return only one "
        f"JSON array containing exactly {job.requested_paraphrase_count} "
        f"distinct {language_name} strings."
    )


def parse_qwen_vl_paraphrases(
    response: str,
    *,
    expected_count: int,
    source_query: str,
    language: Literal["zh", "en"],
) -> list[str]:
    """Parse and structurally validate one deterministic Qwen-VL response."""
    normalized = _parse_qwen_vl_string_array(response)
    if len(normalized) != expected_count:
        raise ValueError(
            f"Expected {expected_count} paraphrases, received {len(normalized)}."
        )
    if any(not item for item in normalized):
        raise ValueError("Qwen-VL paraphrases cannot be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Qwen-VL paraphrases cannot contain duplicates.")
    source_key = " ".join(source_query.strip().split()).casefold()
    if any(item.casefold() == source_key for item in normalized):
        raise ValueError("Qwen-VL copied the source query instead of rewriting it.")
    _validate_output_language(normalized, language)
    return normalized


def collect_qwen_vl_paraphrase_candidates(
    response: str,
    *,
    source_query: str,
    language: Literal["zh", "en"],
) -> list[str]:
    """Keep unique non-source candidates for accumulation across retries."""
    normalized = _parse_qwen_vl_string_array(response)
    _validate_output_language(normalized, language)
    source_key = " ".join(source_query.strip().split()).casefold()
    candidates: list[str] = []
    seen_keys: set[str] = set()
    for candidate in normalized:
        candidate_key = candidate.casefold()
        if candidate_key == source_key or candidate_key in seen_keys:
            continue
        seen_keys.add(candidate_key)
        candidates.append(candidate)
    return candidates


def _parse_qwen_vl_string_array(response: str) -> list[str]:
    """Decode one non-empty JSON string array without semantic filtering."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Qwen-VL response is empty.")
    payload = _extract_json_array(response)
    if not isinstance(payload, list) or any(
        not isinstance(item, str) for item in payload
    ):
        raise ValueError("Qwen-VL response must be a JSON array of strings.")
    normalized = [" ".join(item.strip().split()) for item in payload]
    if any(not item for item in normalized):
        raise ValueError("Qwen-VL paraphrases cannot be empty.")
    return normalized


def _extract_json_array(response: str) -> object:
    """Accept a bare array or one fenced JSON payload without surrounding prose."""
    stripped = response.strip()
    fenced = re.fullmatch(
        r"```[A-Za-z0-9_-]*\s*(\[.*\])\s*```",
        stripped,
        re.DOTALL,
    )
    candidate = fenced.group(1) if fenced else stripped
    return json.loads(candidate)


def _retry_instruction(
    last_error: Exception | None,
    collected: list[str],
    language: Literal["zh", "en"],
) -> str:
    """Turn the prior structural failure into one bounded correction prompt."""
    detail = str(last_error) if last_error else ""
    if language == "zh":
        instruction = (
            "\n上一次回答无效。只返回纯 JSON 字符串数组，不要使用 Markdown，"
            "不要返回对象、字段名或解释。"
        )
        if collected:
            excluded = json.dumps(collected, ensure_ascii=False)
            instruction += (
                " 不要重复原句或这些已接受的候选表达："
                f"{excluded}。请使用明显不同的句式继续改写。"
            )
        elif "copied the source query" in detail or "Collected 0" in detail:
            instruction += (
                " 不要照抄原句；可改用‘请指出’、‘帮我找出’、‘哪里是’等不同句式，"
                "但必须完整保留原句的方位、属性和服装关系。"
            )
        elif "duplicates" in detail:
            instruction += " 每条候选表达必须互不重复。"
        elif "Expected" in detail:
            instruction += " 必须返回要求数量的字符串。"
        return instruction
    instruction = (
        "\nThe prior response was invalid. Return only the bare JSON array "
        "with no Markdown or explanation."
    )
    if collected:
        excluded = json.dumps(collected, ensure_ascii=False)
        instruction += (
            " Do not repeat the source or these already accepted candidates: "
            f"{excluded}. Produce genuinely different wording."
        )
    elif "copied the source query" in detail:
        instruction += (
            " Replace every sentence that matches the source after ignoring "
            "capitalization; all candidates need genuinely different wording."
        )
    elif "duplicates" in detail:
        instruction += " Make every candidate distinct."
    elif "Expected" in detail:
        instruction += " Return exactly the requested number of strings."
    return instruction


def _validate_output_language(
    paraphrases: list[str],
    language: Literal["zh", "en"],
) -> None:
    """Reject obvious cross-language drift before human semantic review."""
    contains_cjk = [bool(re.search(r"[\u3400-\u9fff]", item)) for item in paraphrases]
    if language == "zh" and not all(contains_cjk):
        raise ValueError("Chinese paraphrases must contain Chinese characters.")
    if language == "en" and any(contains_cjk):
        raise ValueError("English paraphrases cannot contain Chinese characters.")
