from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar
from urllib.parse import urlparse

import fitz
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

T = TypeVar("T", bound=BaseModel)
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{3,}")


class Question(BaseModel):
    text: str = Field(min_length=8)
    topic: str = Field(min_length=2)
    expected_points: list[str] = Field(min_length=2, max_length=4)
    source_excerpt: str = Field(min_length=8)
    source_id: int = Field(default=1, ge=1, le=6)


class CriterionAssessment(BaseModel):
    criterion_index: int = Field(ge=1, le=6)
    status: Literal["met", "partial", "missing"]
    explanation: str = Field(min_length=3)
    answer_evidence: str | None = None


class EvaluationJudgement(BaseModel):
    criteria: list[CriterionAssessment] = Field(min_length=1, max_length=6)


class AnswerEvaluation(BaseModel):
    score: int = Field(ge=0, le=100)
    feedback: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    evidence_excerpt: str
    criterion_results: list[CriterionAssessment] = Field(default_factory=list)
    follow_up_needed: bool = False
    follow_up_question: str | None = None


class SessionRecord(BaseModel):
    question: Question
    answer: str
    evaluation: AnswerEvaluation


class SessionSummary(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    mastered_topics: list[str]
    weak_topics: list[str]
    revision_plan: list[str]


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3:14b"
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("PROMETHEUS_API_KEY", ""),
            base_url=os.getenv(
                "PROMETHEUS_BASE_URL", "http://localhost:11434/v1"
            ),
            model=os.getenv("PROMETHEUS_MODEL", "qwen3:14b"),
        )


def is_local_api(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").casefold()
    return host in {"localhost", "127.0.0.1", "::1"}


def is_ollama_api(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return is_local_api(base_url) and parsed.port == 11434


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_material(file_name: str, content: bytes) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        with fitz.open(stream=content, filetype="pdf") as document:
            text = "\n\n".join(page.get_text("text") for page in document)
    elif suffix in {".txt", ".md"}:
        text = content.decode("utf-8-sig", errors="replace")
    else:
        raise ValueError("Поддерживаются файлы PDF, TXT и MD.")

    text = normalize_text(text)
    if len(text) < 200:
        raise ValueError(
            "В документе найдено слишком мало текста. "
            "Возможно, PDF состоит из сканов без текстового слоя."
        )
    return text


def chunk_material(text: str, max_chars: int = 1800) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        else:
            sentences = [paragraph]

        for unit in sentences:
            if current and current_size + len(unit) + 2 > max_chars:
                chunks.append("\n\n".join(current))
                current = []
                current_size = 0
            current.append(unit)
            current_size += len(unit) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text)}


def retrieve_chunks(query: str, chunks: list[str], limit: int = 4) -> list[str]:
    query_tokens = _tokens(query)
    scored: list[tuple[float, int, str]] = []
    for index, chunk in enumerate(chunks):
        chunk_tokens = _tokens(chunk)
        overlap = len(query_tokens & chunk_tokens)
        score = overlap / max(len(query_tokens), 1)
        scored.append((score, -index, chunk))
    scored.sort(reverse=True)
    selected = [chunk for score, _, chunk in scored[:limit] if score > 0]
    return selected or chunks[:limit]


def representative_context(chunks: list[str], limit_chars: int = 9000) -> str:
    return "\n\n---\n\n".join(
        representative_chunks(chunks, limit_chars)
    )


def representative_chunks(
    chunks: list[str], limit_chars: int = 9000
) -> list[str]:
    if not chunks:
        return []
    if len(chunks) <= 6:
        selected = chunks
    else:
        indexes = {
            round(index * (len(chunks) - 1) / 5)
            for index in range(6)
        }
        selected = [chunks[index] for index in sorted(indexes)]
    result: list[str] = []
    total = 0
    for chunk in selected:
        remaining = limit_chars - total
        if remaining <= 0:
            break
        result.append(chunk[:remaining])
        total += len(result[-1])
    return result


def grounded_excerpt(candidate: str, context_chunks: list[str]) -> str:
    material = "\n".join(context_chunks)
    compact_candidate = normalize_text(candidate).strip(" «»\"'")
    if compact_candidate and compact_candidate.casefold() in material.casefold():
        return compact_candidate

    fallback = normalize_text(context_chunks[0]) if context_chunks else ""
    if len(fallback) <= 320:
        return fallback
    sentence_end = re.search(r"[.!?](?:\s|$)", fallback[180:320])
    end = 180 + sentence_end.end() if sentence_end else 320
    return fallback[:end].strip()


def verified_answer_evidence(candidate: str | None, answer: str) -> str | None:
    if not candidate:
        return None
    compact_candidate = normalize_text(candidate).strip(" «»\"'")
    compact_answer = normalize_text(answer)
    if compact_candidate.casefold() in compact_answer.casefold():
        return compact_candidate
    return None


def select_next_passages(
    records: list[SessionRecord],
    passages: list[str],
    limit: int = 4,
) -> list[str]:
    if not passages:
        return []
    last = records[-1]
    if last.evaluation.follow_up_needed:
        last_source = next(
            (
                passage
                for passage in passages
                if last.question.source_excerpt.casefold()
                in passage.casefold()
            ),
            None,
        )
        retrieved = retrieve_chunks(
            last.question.topic + " " + " ".join(last.evaluation.gaps),
            passages,
            limit=limit,
        )
        ordered = ([last_source] if last_source else []) + retrieved
        return list(dict.fromkeys(ordered))[:limit]

    used_excerpts = [
        record.question.source_excerpt.casefold() for record in records
    ]
    unused = [
        passage
        for passage in passages
        if not any(excerpt in passage.casefold() for excerpt in used_excerpts)
    ]
    return representative_chunks(unused or passages, limit_chars=7000)[:limit]


def demo_question(
    language: str = "ru", difficulty: str = "standard"
) -> Question:
    if difficulty == "basic":
        if language == "en":
            return Question(
                text=(
                    "What is the difference between classification and "
                    "regression?"
                ),
                topic="Supervised learning tasks",
                expected_points=[
                    "Classification predicts a category",
                    "Regression predicts a numeric value",
                ],
                source_excerpt=(
                    "В задаче классификации ответ является категорией, "
                    "например «спам» или «не спам». В задаче регрессии "
                    "предсказывается числовое значение"
                ),
            )
        return Question(
            text="Чем классификация отличается от регрессии?",
            topic="Задачи обучения с учителем",
            expected_points=[
                "Классификация предсказывает категорию",
                "Регрессия предсказывает числовое значение",
            ],
            source_excerpt=(
                "В задаче классификации ответ является категорией, "
                "например «спам» или «не спам». В задаче регрессии "
                "предсказывается числовое значение"
            ),
        )

    if difficulty == "advanced":
        if language == "en":
            return Question(
                text=(
                    "Why is fitting data transformations before the train-test "
                    "split a form of data leakage?"
                ),
                topic="Data leakage",
                expected_points=[
                    "Test-set statistics influence training",
                    "The evaluation becomes overly optimistic",
                    "Transformations must be fitted on training data only",
                ],
                source_excerpt=(
                    "нормализация всех данных до разбиения позволяет "
                    "статистикам тестовой части попасть в обучение"
                ),
            )
        return Question(
            text=(
                "Почему нормализация до разбиения выборки является "
                "утечкой данных?"
            ),
            topic="Утечка данных",
            expected_points=[
                "Статистики тестовой части влияют на обучение",
                "Оценка качества становится завышенной",
                "Преобразования обучают только на тренировочных данных",
            ],
            source_excerpt=(
                "нормализация всех данных до разбиения позволяет "
                "статистикам тестовой части попасть в обучение"
            ),
        )

    if language == "en":
        return Question(
            text=(
                "Why can accuracy be misleading with imbalanced classes, "
                "and which metrics should be considered instead?"
            ),
            topic="Classification metrics",
            expected_points=[
                "The majority class can dominate the total accuracy",
                "A high accuracy can hide failures on the minority class",
                "Precision, recall, F1, and the confusion matrix add insight",
            ],
            source_excerpt=(
                "При сильном дисбалансе классов accuracy может вводить "
                "в заблуждение"
            ),
        )
    return Question(
        text=(
            "Почему accuracy может вводить в заблуждение при дисбалансе "
            "классов и какие метрики стоит анализировать?"
        ),
        topic="Метрики классификации",
        expected_points=[
            "Большой класс может доминировать в общей accuracy",
            "Высокая accuracy может скрывать ошибки на редком классе",
            "Нужны precision, recall, F1 и матрица ошибок",
        ],
        source_excerpt=(
            "При сильном дисбалансе классов accuracy может вводить "
            "в заблуждение"
        ),
    )


def build_answer_evaluation(
    expected_points: list[str],
    judgement: EvaluationJudgement,
    answer: str,
    evidence_excerpt: str,
    language: str = "ru",
) -> AnswerEvaluation:
    by_index: dict[int, CriterionAssessment] = {}
    for item in judgement.criteria:
        if item.criterion_index > len(expected_points):
            continue
        if item.criterion_index in by_index:
            raise ValueError("Модель продублировала критерий оценки.")
        item.answer_evidence = verified_answer_evidence(
            item.answer_evidence, answer
        )
        by_index[item.criterion_index] = item

    expected_indexes = set(range(1, len(expected_points) + 1))
    if set(by_index) != expected_indexes:
        raise ValueError("Модель оценила не все критерии ответа.")

    ordered = [by_index[index] for index in sorted(by_index)]
    weights = {"met": 1.0, "partial": 0.5, "missing": 0.0}
    score = round(
        100
        * sum(weights[item.status] for item in ordered)
        / len(expected_points)
    )
    strengths = [
        expected_points[item.criterion_index - 1]
        for item in ordered
        if item.status == "met"
    ]
    gaps = [
        expected_points[item.criterion_index - 1]
        for item in ordered
        if item.status != "met"
    ]

    if language == "en":
        if score >= 90:
            feedback = "The answer fully and correctly covers the rubric."
        elif score >= 70:
            feedback = (
                "The answer is mostly correct, with a few incomplete points."
            )
        elif score >= 40:
            feedback = "The main idea is present, but important gaps remain."
        else:
            feedback = (
                "The answer does not yet demonstrate enough understanding."
            )
        follow_up_template = "Explain this point in more detail: {gap}"
    else:
        if score >= 90:
            feedback = "Ответ полно и корректно раскрывает ожидаемые пункты."
        elif score >= 70:
            feedback = (
                "Ответ в целом верный, но отдельные пункты раскрыты не полностью."
            )
        elif score >= 40:
            feedback = (
                "Ответ частичный: основная идея присутствует, "
                "но есть важные пробелы."
            )
        else:
            feedback = (
                "Ответ пока не демонстрирует достаточного понимания темы."
            )
        follow_up_template = "Раскройте подробнее: {gap}"

    follow_up_needed = score < 70
    follow_up_question = (
        follow_up_template.format(gap=gaps[0])
        if follow_up_needed and gaps
        else None
    )
    return AnswerEvaluation(
        score=score,
        feedback=feedback,
        strengths=strengths,
        gaps=gaps,
        evidence_excerpt=evidence_excerpt,
        criterion_results=ordered,
        follow_up_needed=follow_up_needed,
        follow_up_question=follow_up_question,
    )


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Модель не вернула JSON-объект.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Ожидался JSON-объект.")
    return value


class VivaCoach:
    def __init__(
        self,
        material: str,
        config: LLMConfig,
        language: str = "ru",
        difficulty: str = "standard",
    ) -> None:
        if not config.api_key and not is_local_api(config.base_url):
            raise ValueError("Не задан API-ключ модели.")
        self.config = config
        self.language = language
        self.difficulty = difficulty
        self.chunks = chunk_material(material)
        all_passages = [
            part.strip() for part in material.split("\n\n") if part.strip()
        ]
        self.source_passages = [
            part for part in all_passages if len(part) >= 80
        ] or all_passages
        if not self.chunks:
            raise ValueError("Не удалось разбить материал на смысловые фрагменты.")

    def _request(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        reasoning: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> T:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        validation_error: Exception | None = None
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            for attempt in range(2):
                if is_ollama_api(self.config.base_url):
                    parsed = urlparse(self.config.base_url)
                    url = f"{parsed.scheme}://{parsed.netloc}/api/chat"
                    payload = {
                        "model": self.config.model,
                        "messages": messages,
                        "stream": False,
                        "think": reasoning,
                        "format": schema.model_json_schema(),
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                        "keep_alive": "10m",
                    }
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    response_data = response.json()
                    raw = response_data["message"]["content"]
                else:
                    url = (
                        f"{self.config.base_url.rstrip('/')}"
                        "/chat/completions"
                    )
                    payload = {
                        "model": self.config.model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                    }
                    headers = {"Content-Type": "application/json"}
                    if self.config.api_key:
                        headers["Authorization"] = (
                            f"Bearer {self.config.api_key}"
                        )
                    response = client.post(
                        url, headers=headers, json=payload
                    )
                    if (
                        response.status_code == 400
                        and "response_format" in response.text
                    ):
                        payload.pop("response_format")
                        response = client.post(
                            url, headers=headers, json=payload
                        )
                    response.raise_for_status()
                    response_data = response.json()
                    raw = response_data["choices"][0]["message"]["content"]

                try:
                    return schema.model_validate(parse_json_object(raw))
                except (
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                    ValidationError,
                ) as error:
                    validation_error = error
                    if attempt == 0:
                        messages.extend(
                            [
                                {
                                    "role": "assistant",
                                    "content": str(raw)[:4000],
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        "Repair the response so it exactly "
                                        "matches the required JSON schema. "
                                        "Return JSON only."
                                    ),
                                },
                            ]
                        )

        raise ValueError(
            "Ответ модели не соответствует ожидаемой структуре."
        ) from validation_error

    def _difficulty_instruction(self) -> str:
        if self.language == "en":
            return {
                "basic": "Use basic recall plus one simple explanation.",
                "advanced": (
                    "Use an advanced application or error-analysis question."
                ),
            }.get(
                self.difficulty,
                "Use an intermediate conceptual understanding question.",
            )
        return {
            "basic": "Задай базовый вопрос с простым объяснением.",
            "advanced": (
                "Задай продвинутый вопрос на применение или анализ ошибки."
            ),
        }.get(
            self.difficulty,
            "Задай вопрос среднего уровня на понимание концепции.",
        )

    def first_question(self) -> Question:
        selected_sources = representative_chunks(self.source_passages)
        context = "\n\n".join(
            f"[SOURCE {index}]\n{source}"
            for index, source in enumerate(selected_sources, start=1)
        )
        if self.language == "en":
            system_prompt = (
                "You are a supportive oral examiner. Use only the supplied "
                "study material. Return JSON only. Write the question, topic, "
                "and expected points in English, even when the source material "
                "is written in another language. Keep source_excerpt as an "
                "exact quote in the source's original language."
            )
            language_instruction = (
                "IMPORTANT: write the question, topic, and expected points in "
                "English. Copy source_excerpt verbatim in its original language."
            )
        else:
            system_prompt = (
                "Ты преподаватель, проводящий доброжелательный устный экзамен. "
                "Работай только по переданному материалу. Верни только JSON."
            )
            language_instruction = "Все поля заполни на русском языке."
        question = self._request(
            system=system_prompt,
            user=f"""
Сформулируй один содержательный вопрос на понимание, а не на запоминание.
Не спрашивай сразу несколько независимых вещей. Дай 2–4 непересекающихся,
проверяемых ожидаемых пункта. Каждый пункт должен прямо поддерживаться
материалом и не должен требовать знаний извне.
{language_instruction}
{self._difficulty_instruction()}
Весь вопрос и все expected_points должны опираться ровно на один SOURCE.
source_id должен быть номером этого SOURCE.

JSON:
{{
  "text": "вопрос",
  "topic": "краткая тема",
  "expected_points": ["ключевой пункт"],
  "source_excerpt": "точная короткая цитата из выбранного SOURCE",
  "source_id": 1
}}

МАТЕРИАЛ:
{context}
""".strip(),
            schema=Question,
        )
        source_index = min(question.source_id, len(selected_sources)) - 1
        relevant = [selected_sources[source_index]]
        question.source_excerpt = grounded_excerpt(
            question.source_excerpt, relevant
        )
        return question

    def next_question(self, records: list[SessionRecord]) -> Question:
        last = records[-1]
        relevant = select_next_passages(
            records, self.source_passages, limit=4
        )

        relevant_context = "\n\n".join(
            f"[SOURCE {index}]\n{source}"
            for index, source in enumerate(relevant, start=1)
        )
        history = "\n".join(
            f"- {record.question.topic}: {record.evaluation.score}/100; "
            f"пробелы: {', '.join(record.evaluation.gaps) or 'нет'}"
            for record in records
        )
        if last.evaluation.follow_up_needed:
            mode_instruction = (
                "Ask one focused follow-up about the last missing criterion."
                if self.language == "en"
                else (
                    "Задай один точечный уточняющий вопрос по последнему "
                    "пропущенному критерию."
                )
            )
        else:
            previous_topics = ", ".join(
                record.question.topic for record in records
            )
            mode_instruction = (
                "Choose a genuinely new topic from the supplied sources. "
                f"Do not ask about these previous topics: {previous_topics}."
                if self.language == "en"
                else (
                    "Выбери действительно новую тему из переданных SOURCE. "
                    f"Не спрашивай прошлые темы: {previous_topics}."
                )
            )
        if self.language == "en":
            system_prompt = (
                "You are an adaptive oral examiner. Use only the supplied "
                "material and history. Return JSON only. Write every generated "
                "field except source_excerpt in English. Keep source_excerpt "
                "as an exact quote in the source's original language."
            )
            language_instruction = (
                "IMPORTANT: write the question, topic, and expected points in "
                "English. Copy source_excerpt verbatim in its original language."
            )
        else:
            system_prompt = (
                "Ты адаптивный преподаватель. Работай только по материалу. "
                "Верни только JSON."
            )
            language_instruction = "Все поля заполни на русском языке."
        question = self._request(
            system=system_prompt,
            user=f"""
Сформулируй следующий один вопрос. {mode_instruction}
Не повторяй формулировки предыдущих вопросов.
Дай 2–4 непересекающихся, проверяемых ожидаемых пункта, прямо поддержанных
материалом.
{language_instruction}
{self._difficulty_instruction()}
Весь вопрос и все expected_points должны опираться ровно на один SOURCE.
source_id должен быть номером этого SOURCE.

ИСТОРИЯ:
{history}

JSON:
{{
  "text": "вопрос",
  "topic": "краткая тема",
  "expected_points": ["ключевой пункт"],
  "source_excerpt": "точная короткая цитата из выбранного SOURCE",
  "source_id": 1
}}

РЕЛЕВАНТНЫЙ МАТЕРИАЛ:
{relevant_context}
""".strip(),
            schema=Question,
        )
        source_index = min(question.source_id, len(relevant)) - 1
        question.source_excerpt = grounded_excerpt(
            question.source_excerpt, [relevant[source_index]]
        )
        return question

    def evaluate(self, question: Question, answer: str) -> AnswerEvaluation:
        relevant = retrieve_chunks(
            question.text
            + " "
            + " ".join(question.expected_points)
            + " "
            + question.source_excerpt,
            self.chunks,
            limit=4,
        )
        context = "\n\n---\n\n".join(relevant)
        numbered_criteria = "\n".join(
            f"{index}. {point}"
            for index, point in enumerate(question.expected_points, start=1)
        )
        if self.language == "en":
            system_prompt = (
                "You assess a student's answer against an explicit rubric. "
                "Match each criterion only against the student's answer and "
                "use the material only for factual verification. Do not add "
                "criteria. Return JSON only and write explanations in English."
            )
            language_instruction = (
                "IMPORTANT: write every explanation in English."
            )
        else:
            system_prompt = (
                "Ты проверяешь ответ студента по явной рубрике. "
                "Сопоставь каждый критерий только с ответом студента. "
                "Материал используй для проверки фактической корректности. "
                "Не добавляй собственные критерии. Верни только JSON."
            )
            language_instruction = "Пояснения пиши на русском языке."
        judgement = self._request(
            system=system_prompt,
            user=f"""
ВОПРОС:
{question.text}

КРИТЕРИИ:
{numbered_criteria}

ОТВЕТ СТУДЕНТА:
{answer}

СПРАВОЧНЫЙ МАТЕРИАЛ:
{context}

Для КАЖДОГО критерия верни ровно один результат:
- met — пункт раскрыт правильно и достаточно;
- partial — правильная идея есть, но пункт раскрыт неполно;
- missing — пункт отсутствует или фактически неверен.

answer_evidence — точная короткая цитата только из ОТВЕТА СТУДЕНТА,
которая подтверждает статус met или partial. Для missing используй null.
Не требуй примеров, определений и тем, которых нет в критериях.
Не выставляй итоговый балл — его вычислит программа.
{language_instruction}

JSON:
{{
  "criteria": [
    {{
      "criterion_index": 1,
      "status": "met",
      "explanation": "почему выбран этот статус",
      "answer_evidence": "точная цитата из ответа или null"
    }}
  ]
}}
""".strip(),
            schema=EvaluationJudgement,
            reasoning=True,
            temperature=0.0,
            max_tokens=2000,
        )
        evidence_excerpt = grounded_excerpt(
            question.source_excerpt, relevant
        )
        return build_answer_evaluation(
            question.expected_points,
            judgement,
            answer,
            evidence_excerpt,
            self.language,
        )


def build_summary(
    records: list[SessionRecord], language: str = "ru"
) -> SessionSummary:
    if not records:
        return SessionSummary(
            overall_score=0,
            mastered_topics=[],
            weak_topics=[],
            revision_plan=[
                "Answer at least one question."
                if language == "en"
                else "Пройти хотя бы один вопрос."
            ],
        )

    overall = round(
        sum(record.evaluation.score for record in records) / len(records)
    )
    mastered = [
        record.question.topic
        for record in records
        if record.evaluation.score >= 70
    ]
    weak = [
        record.question.topic
        for record in records
        if record.evaluation.score < 70
    ]
    gaps = [
        gap
        for record in records
        for gap in record.evaluation.gaps
    ]
    revision = list(dict.fromkeys(gaps))[:5]
    if not revision:
        revision = [
            (
                "Reinforce the material with another exam in 1–2 days."
                if language == "en"
                else "Закрепить материал повторным опросом через 1–2 дня."
            )
        ]
    return SessionSummary(
        overall_score=overall,
        mastered_topics=list(dict.fromkeys(mastered)),
        weak_topics=list(dict.fromkeys(weak)),
        revision_plan=revision,
    )


def build_markdown_report(
    records: list[SessionRecord],
    material_name: str,
    language: str = "ru",
) -> str:
    summary = build_summary(records, language)
    if language == "en":
        lines = [
            "# Viva AI — Exam Report",
            "",
            f"**Material:** {material_name}",
            f"**Overall score:** {summary.overall_score}/100",
            "",
            "## Question results",
        ]
        plan_title = "## Personal study plan"
        source_label = "Source"
        answer_label = "Answer"
    else:
        lines = [
            "# Viva AI — Отчёт об экзамене",
            "",
            f"**Материал:** {material_name}",
            f"**Итог:** {summary.overall_score}/100",
            "",
            "## Результаты по вопросам",
        ]
        plan_title = "## Персональный план повторения"
        source_label = "Источник"
        answer_label = "Ответ"

    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                "",
                f"### {index}. {record.question.topic} — "
                f"{record.evaluation.score}/100",
                "",
                record.question.text,
                "",
                f"**{answer_label}:** {record.answer}",
                "",
            ]
        )
        for criterion in record.evaluation.criterion_results:
            point = record.question.expected_points[
                criterion.criterion_index - 1
            ]
            lines.append(f"- **{criterion.status}:** {point}")
        lines.extend(
            [
                "",
                f"> **{source_label}:** "
                f"{record.evaluation.evidence_excerpt}",
            ]
        )

    lines.extend(["", plan_title, ""])
    lines.extend(
        f"{index}. {item}"
        for index, item in enumerate(summary.revision_plan, start=1)
    )
    return "\n".join(lines).strip() + "\n"
