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
from fashion_semantic_parser.dao.localization.taxonomy import (
    FASHIONPEDIA_PART_CATEGORIES,
    pluralize_english_part_name,
)

_SPATIAL_TERMS: dict[
    Literal["zh", "en"],
    dict[str, tuple[str, ...]],
] = {
    "en": {
        "left": ("left",),
        "right": ("right",),
        "upper": ("upper", "top", "above"),
        "lower": ("lower", "bottom", "below"),
    },
    "zh": {
        "left": ("左",),
        "right": ("右",),
        "upper": ("上", "顶部"),
        "lower": ("下", "底部"),
    },
}

_GARMENT_TERMS: dict[
    str,
    dict[Literal["zh", "en"], tuple[str, ...]],
] = {
    "shirt": {"en": ("shirt", "blouse"), "zh": ("衬衫",)},
    "top": {"en": ("top", "t-shirt", "sweatshirt"), "zh": ("上衣",)},
    "sweater": {"en": ("sweater",), "zh": ("毛衣",)},
    "cardigan": {"en": ("cardigan",), "zh": ("开衫",)},
    "jacket": {"en": ("jacket",), "zh": ("夹克",)},
    "vest": {"en": ("vest",), "zh": ("马甲",)},
    "pants": {"en": ("pants", "trousers", "jeans"), "zh": ("裤",)},
    "shorts": {"en": ("shorts",), "zh": ("短裤",)},
    "skirt": {"en": ("skirt",), "zh": ("裙",)},
    "coat": {"en": ("coat",), "zh": ("大衣", "外套")},
    "dress": {"en": ("dress",), "zh": ("连衣裙",)},
    "jumpsuit": {"en": ("jumpsuit",), "zh": ("连体裤",)},
    "cape": {"en": ("cape",), "zh": ("披肩",)},
}

_META_PREFIX = re.compile(
    r"^(?:请求式表达|疑问式表达|定位式表达|查询|request|question|location)"
    r"\s*[：:]\s*",
    re.IGNORECASE,
)

_FORBIDDEN_INTENT_TERMS = {
    "en": (
        "describe",
        "description",
        "fashion advice",
        "outfit",
        "feel confident",
        "beautiful",
        "elegance",
        "sophistication",
        "unzip",
        "zip up",
        "open the zipper",
        "close the zipper",
    ),
    "zh": ("描述", "穿搭", "好看", "拉开", "拉上", "解开", "打开拉链", "关闭拉链"),
}

_SPATIAL_REFERENCE_TERMS = {
    "en": (
        "garment",
        "clothing",
        "apparel",
        "dress",
        "shirt",
        "top",
        "sweater",
        "cardigan",
        "jacket",
        "vest",
        "pants",
        "trousers",
        "jeans",
        "shorts",
        "skirt",
        "coat",
        "jumpsuit",
        "cape",
    ),
    "zh": ("衣服", "服装", "衣物", "衬衫", "上衣", "裤", "裙", "大衣", "外套"),
}


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
    retry_count: int = Field(default=9, ge=0, le=10)
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
            ":prd312-semantic-gate-v11"
        )

    @property
    def resume_compatible_generator_identities(self) -> tuple[str, ...]:
        """Resume only rows produced by the exact active semantic policy."""
        return (self.generator_identity,)

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
                    job=job,
                )
                _append_unique_candidates(collected, candidates)
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
        if collected:
            return ReferringParaphraseResult(
                source_sample_id=job.source_sample_id,
                source_fingerprint=job.source_fingerprint,
                language=job.language,
                generator_model=self.generator_identity,
                review_status="unreviewed",
                paraphrases=collected,
            )
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
    constraints = _prompt_constraints(job)
    if job.language == "zh":
        return (
            "你正在为语言引导的服饰区域定位任务生成训练语料。\n"
            f"原始查询：{json.dumps(job.source_query, ensure_ascii=False)}\n"
            f"目标部件：{job.target_label}\n"
            f"目标实例数量：{job.target_count}\n"
            f"必须保留的语义维度：{dimensions}\n"
            f"必须逐条保留的明确约束：{constraints}\n"
            f"任务要求：将原始查询改写为 {job.requested_paraphrase_count} 条中文表达。\n"
            "可以改成请求式、疑问式或定位式表达，但不能扩大或缩小目标集合；"
            "必须保留方位、属性、服装关系、单复数含义和参照系。"
            "不能把‘这件/这条’改成任意的‘一件/一条’，也不能询问目标是什么。"
            "每条表达都必须与原句有明显文字差异，不能照抄原句。"
            "每条必须是简短的区域定位短语或定位问题，禁止输出说明、穿搭建议、"
            "描述任务或操作衣物的动作，也不能重复词组或句子片段。"
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
        f"Explicit constraints every rewrite must retain: {constraints}\n"
        f"Task: {job.instruction}\n"
        "Never broaden or narrow the target set. Keep every direction, "
        "attribute, garment relation, singular/plural meaning, and reference "
        "frame unchanged. Do not change a definite garment into an arbitrary "
        "a/an garment or ask what the target is. Every output must be a short "
        "region-locating phrase "
        "or question, never a description, fashion advice, or garment action. "
        "Do not repeat a phrase or sentence fragment inside one rewrite. "
        "Do not copy the source sentence. Return only one "
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
    if len({item.casefold() for item in normalized}) != len(normalized):
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
    job: ReferringParaphraseJob | None = None,
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
        if job is not None and not _preserves_semantic_constraints(candidate, job):
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
    normalized = [_normalize_candidate(item) for item in payload]
    if any(not item for item in normalized):
        raise ValueError("Qwen-VL paraphrases cannot be empty.")
    return normalized


def _normalize_candidate(value: str) -> str:
    """Remove model-added format labels before validating training text."""
    normalized = " ".join(value.strip().split())
    return _META_PREFIX.sub("", normalized).strip()


def _append_unique_candidates(
    collected: list[str],
    candidates: list[str],
) -> None:
    """Accumulate candidates with case-insensitive identity across attempts."""
    collected_keys = {candidate.casefold() for candidate in collected}
    for candidate in candidates:
        if candidate.casefold() not in collected_keys:
            collected.append(candidate)
            collected_keys.add(candidate.casefold())


def _prompt_constraints(job: ReferringParaphraseJob) -> str:
    """Render structured anchors explicitly instead of relying on prose alone."""
    constraints = [f"target={job.target_label}"]
    if job.spatial_modifier is not None:
        constraints.append(f"spatial={job.spatial_modifier}")
    if job.reference_category is not None:
        relation = job.reference_category
        if relation == "top":
            relation += " (the garment category, never the upper position)"
        constraints.append(f"garment_relation={relation}")
    if job.attribute_phrase is not None:
        constraints.append(f"attribute={job.attribute_phrase}")
    constraints.append(f"target_count={job.target_count}")
    return ", ".join(constraints)


def _preserves_semantic_constraints(
    candidate: str,
    job: ReferringParaphraseJob,
) -> bool:
    """Reject rewrites that drop a target, direction, attribute, or relation."""
    if not (
        _is_clean_localization_expression(candidate, job)
        and _contains_target_term(candidate, job)
        and _preserves_target_count(candidate, job)
    ):
        return False
    valid = True
    if job.spatial_modifier is not None:
        spatial_terms = _SPATIAL_TERMS[job.language][job.spatial_modifier]
        reference_terms = _SPATIAL_REFERENCE_TERMS[job.language]
        valid = valid and any(_contains_term(candidate, term) for term in spatial_terms)
        valid = valid and any(
            _contains_term(candidate, term) for term in reference_terms
        )
    if job.reference_category is not None:
        garment_terms = _GARMENT_TERMS.get(job.reference_category, {}).get(
            job.language,
            (job.reference_category,),
        )
        valid = valid and any(_contains_term(candidate, term) for term in garment_terms)
        if job.reference_category == "top" and job.language == "en":
            valid = valid and bool(
                re.search(r"\btop(?:\s+garment|'s)\b", candidate, re.IGNORECASE)
            )
    if job.attribute_phrase is not None:
        valid = valid and _contains_attribute_target_phrase(candidate, job)
    return valid


# Candidate-policy checks are intentionally named individually for auditability.
# pylint: disable-next=too-many-locals
def _is_clean_localization_expression(
    candidate: str,
    job: ReferringParaphraseJob,
) -> bool:
    """Block prose, garment actions, invented directions, and malformed text."""
    has_parentheses = "(" in candidate or ")" in candidate
    has_repeated_word = bool(
        re.search(r"\b([A-Za-z]+)\s+\1\b", candidate, re.IGNORECASE)
    )
    has_repeated_phrase = _has_repeated_phrase(candidate)
    too_long = (job.language == "en" and len(candidate.split()) > 20) or (
        job.language == "zh" and len(candidate) > 40
    )
    has_forbidden_intent = any(
        term.casefold() in candidate.casefold()
        for term in _FORBIDDEN_INTENT_TERMS[job.language]
    )
    target_is_late = job.language == "en" and not _english_target_appears_early(
        candidate, job
    )
    has_cross_language_label = job.language == "zh" and bool(
        re.search(r"[A-Za-z]{2,}", candidate)
    )
    has_confirmation_intent = _has_confirmation_intent(candidate, job.language)
    has_existence_intent = _has_existence_intent(candidate, job.language)
    adds_spatial = _adds_unrequested_spatial_modifier(candidate, job)
    adds_garment_relation = _adds_unrequested_garment_relation(candidate, job)
    adds_ordinal = _adds_unrequested_ordinal(candidate, job)
    adds_indefinite_relation = _adds_unrequested_indefinite_relation(candidate, job)
    has_formatting_quotes = job.language == "zh" and bool(
        re.search(r"['\"‘’“”]", candidate)
    )
    return not any(
        (
            has_parentheses,
            has_repeated_word,
            has_repeated_phrase,
            too_long,
            has_forbidden_intent,
            target_is_late,
            has_cross_language_label,
            has_confirmation_intent,
            has_existence_intent,
            adds_spatial,
            adds_garment_relation,
            adds_ordinal,
            adds_indefinite_relation,
            has_formatting_quotes,
        )
    )


def _has_repeated_phrase(candidate: str) -> bool:
    """Reject a non-overlapping token sequence repeated in one candidate."""
    tokens = re.findall(r"[A-Za-z0-9]+|[\u3400-\u9fff]", candidate.casefold())
    maximum_size = min(6, len(tokens) // 2)
    for size in range(2, maximum_size + 1):
        first_positions: dict[tuple[str, ...], int] = {}
        for index in range(len(tokens) - size + 1):
            phrase = tuple(tokens[index : index + size])
            first_position = first_positions.setdefault(phrase, index)
            if index - first_position >= size:
                return True
    return False


def _contains_attribute_target_phrase(
    candidate: str,
    job: ReferringParaphraseJob,
) -> bool:
    """Keep attributes attached to their target instead of a new object."""
    if job.attribute_phrase is None:
        return True
    candidate_key = candidate.casefold()
    attribute = re.escape(job.attribute_phrase.casefold())
    attribute_matches = re.findall(
        rf"(?<![a-z0-9]){attribute}(?![a-z0-9])",
        candidate_key,
    )
    if len(attribute_matches) != 1:
        return False
    for target_term in _localized_target_terms(job):
        target = re.escape(target_term.casefold())
        patterns = (
            rf"(?<![a-z0-9]){attribute}\s+{target}(?![a-z0-9])",
            rf"(?<![a-z0-9]){target}\s+with\s+"
            rf"(?:a\s+|an\s+|the\s+)?{attribute}(?![a-z0-9])",
        )
        if any(re.search(pattern, candidate_key) for pattern in patterns):
            return True
    return False


def _has_confirmation_intent(
    candidate: str,
    language: Literal["zh", "en"],
) -> bool:
    """Reject yes/no or object-identification questions instead of localization."""
    if language == "zh":
        return candidate.startswith(
            ("这是", "这是什么", "什么是", "哪些是", "哪个是", "是不是", "是否")
        )
    confirmation = bool(
        re.match(r"^(?:is|are|does|do|has|have)\b", candidate, re.IGNORECASE)
    )
    identification = bool(
        re.match(r"^what\b.*\b(?:is|are)\b", candidate, re.IGNORECASE)
    )
    location_terms = bool(
        re.search(r"\b(?:location|position|where)\b", candidate, re.IGNORECASE)
    )
    return confirmation or (identification and not location_terms)


def _has_existence_intent(
    candidate: str,
    language: Literal["zh", "en"],
) -> bool:
    """Reject presence claims or questions that do not request localization."""
    if language == "zh":
        return bool(re.search(r"(?:有没有|是否有|(?<!所)有)", candidate))
    return bool(
        re.match(
            r"^(?:is|are)\s+there\b|^there\s+(?:is|are)\b",
            candidate,
            re.IGNORECASE,
        )
    )


def _english_target_appears_early(
    candidate: str,
    job: ReferringParaphraseJob,
) -> bool:
    """Keep the requested region as the phrase head instead of the garment."""
    target_positions = [
        match.start()
        for term in _localized_target_terms(job)
        for match in [
            re.search(
                rf"(?<![a-z0-9]){re.escape(term.casefold())}(?![a-z0-9])",
                candidate.casefold(),
            )
        ]
        if match is not None
    ]
    if not target_positions:
        return False
    preceding_word_count = len(candidate[: min(target_positions)].split())
    return preceding_word_count <= 4


def _adds_unrequested_spatial_modifier(
    candidate: str,
    job: ReferringParaphraseJob,
) -> bool:
    """Reject invented positions on non-spatial source expressions."""
    if job.spatial_modifier is not None:
        return False
    terms: tuple[str, ...]
    if job.language == "en":
        terms = (
            "left",
            "right",
            "upper",
            "lower",
            "top",
            "bottom",
            "front",
            "back",
            "above",
            "below",
        )
    else:
        terms = (
            "左侧",
            "右侧",
            "左边",
            "右边",
            "上方",
            "下方",
            "前面",
            "后面",
            "底部",
            "顶部",
        )
    return any(
        _contains_term(candidate, term) and not _contains_term(job.source_query, term)
        for term in terms
    )


def _adds_unrequested_garment_relation(
    candidate: str,
    job: ReferringParaphraseJob,
) -> bool:
    """Reject a specific garment category invented by a non-relation query."""
    if job.reference_category is not None:
        return False
    for language_terms in _GARMENT_TERMS.values():
        for term in language_terms[job.language]:
            if _contains_term(candidate, term) and not _contains_term(
                job.source_query,
                term,
            ):
                return True
    return False


def _adds_unrequested_ordinal(
    candidate: str,
    job: ReferringParaphraseJob,
) -> bool:
    """Reject an invented instance index such as 'second rivet'."""
    if job.language == "zh":
        pattern = r"第(?:[一二三四五六七八九十百0-9]+)(?:个|颗|枚|处|条)?"
    else:
        pattern = r"\b(?:first|second|third|fourth|fifth|[0-9]+(?:st|nd|rd|th))\b"
    candidate_ordinals = {
        match.casefold()
        for match in re.findall(pattern, candidate, flags=re.IGNORECASE)
    }
    source_ordinals = {
        match.casefold()
        for match in re.findall(pattern, job.source_query, flags=re.IGNORECASE)
    }
    return not candidate_ordinals.issubset(source_ordinals)


def _adds_unrequested_indefinite_relation(
    candidate: str,
    job: ReferringParaphraseJob,
) -> bool:
    """Keep a definite garment reference from becoming an arbitrary instance."""
    if job.reference_category is None:
        return False
    if job.language == "zh":
        source_is_definite = bool(re.search(r"这(?:件|条|款|个)?", job.source_query))
        return source_is_definite and bool(
            re.search(r"(?:^|[^第])一(?:件|条|款|个)", candidate)
        )
    source_is_definite = bool(re.search(r"\bthe\b", job.source_query, re.IGNORECASE))
    if not source_is_definite:
        return False
    garment_terms = _GARMENT_TERMS.get(job.reference_category, {}).get(
        "en",
        (job.reference_category,),
    )
    return any(
        re.search(
            rf"\b(?:a|an)\s+{re.escape(term)}\b",
            candidate,
            re.IGNORECASE,
        )
        for term in garment_terms
    )


def _contains_target_term(candidate: str, job: ReferringParaphraseJob) -> bool:
    """Check the localized Fashionpedia target without collapsing its query."""
    return any(_contains_term(candidate, term) for term in _localized_target_terms(job))


def _preserves_target_count(candidate: str, job: ReferringParaphraseJob) -> bool:
    """Keep single-instance and all-instance queries semantically distinct."""
    if job.language == "zh":
        has_all_marker = any(marker in candidate for marker in ("所有", "全部"))
        return has_all_marker if job.target_count > 1 else not has_all_marker
    plural_terms = _localized_english_plural_target_terms(job)
    has_plural_target = any(_contains_term(candidate, term) for term in plural_terms)
    has_all_marker = bool(re.search(r"\b(?:all|both)\b", candidate, re.IGNORECASE))
    if job.target_count > 1:
        return has_plural_target
    return not (has_plural_target or has_all_marker)


def _localized_target_terms(job: ReferringParaphraseJob) -> list[str]:
    """Return accepted target aliases for the job language."""
    category = next(
        (
            item
            for item in FASHIONPEDIA_PART_CATEGORIES
            if item.english_name == job.target_label
        ),
        None,
    )
    if category is None:
        return [job.target_label]
    terms = [
        category.english_name,
        pluralize_english_part_name(category.english_name),
    ]
    if job.language == "zh":
        terms = [
            term
            for term in category.prompt_terms
            if re.search(r"[\u3400-\u9fff]", term)
        ]
        terms.append(category.chinese_name)
    return terms


def _localized_english_plural_target_terms(job: ReferringParaphraseJob) -> list[str]:
    """Return controlled English plural aliases for target-count validation."""
    category = next(
        (
            item
            for item in FASHIONPEDIA_PART_CATEGORIES
            if item.english_name == job.target_label
        ),
        None,
    )
    target = category.english_name if category is not None else job.target_label
    return [pluralize_english_part_name(target)]


def _contains_term(text: str, term: str) -> bool:
    """Match English tokens by boundaries and Chinese anchors by substring."""
    if re.search(r"[\u3400-\u9fff]", term):
        return term in text
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(term.casefold())}(?![a-z0-9])",
            text.casefold(),
        )
    )


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
