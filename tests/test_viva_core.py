import json

import pytest

from ui_text import TRANSLATIONS, text
from viva_core import (
    AnswerEvaluation,
    CriterionAssessment,
    EvaluationJudgement,
    LLMConfig,
    Question,
    SessionRecord,
    VivaCoach,
    build_answer_evaluation,
    build_markdown_report,
    build_summary,
    chunk_material,
    extract_material,
    demo_question,
    grounded_excerpt,
    is_local_api,
    is_ollama_api,
    parse_json_object,
    retrieve_chunks,
    select_next_passages,
)


def test_extract_text_material() -> None:
    content = ("Машинное обучение. " * 20).encode()
    result = extract_material("notes.txt", content)
    assert result.startswith("Машинное")


def test_rejects_too_short_material() -> None:
    with pytest.raises(ValueError, match="слишком мало"):
        extract_material("notes.md", "Коротко".encode())


def test_chunking_preserves_content() -> None:
    text = "\n\n".join(["Первый абзац. " * 20, "Второй абзац. " * 20])
    chunks = chunk_material(text, max_chars=200)
    assert len(chunks) >= 2
    assert "Первый" in chunks[0]
    assert any("Второй" in chunk for chunk in chunks)


def test_retrieval_prefers_matching_chunk() -> None:
    chunks = [
        "Классификация предсказывает категорию объекта.",
        "Регуляризация помогает бороться с переобучением модели.",
    ]
    result = retrieve_chunks("Как регуляризация уменьшает переобучение?", chunks, 1)
    assert result == [chunks[1]]


def test_parse_json_from_code_fence() -> None:
    payload = {"score": 80}
    assert parse_json_object(f"```json\n{json.dumps(payload)}\n```") == payload


def test_hallucinated_excerpt_is_replaced_with_source_text() -> None:
    chunks = ["Precision показывает точность положительных предсказаний."]
    excerpt = grounded_excerpt("Несуществующая цитата.", chunks)
    assert excerpt == chunks[0]


def test_local_api_detection() -> None:
    assert is_local_api("http://localhost:11434/v1")
    assert is_local_api("http://127.0.0.1:11434/v1")
    assert not is_local_api("https://api.openai.com/v1")
    assert is_ollama_api("http://localhost:11434/v1")
    assert not is_ollama_api("http://localhost:1234/v1")


def test_deterministic_score_from_criterion_statuses() -> None:
    judgement = EvaluationJudgement(
        criteria=[
            CriterionAssessment(
                criterion_index=1,
                status="met",
                explanation="Раскрыто полностью.",
                answer_evidence="первый пункт",
            ),
            CriterionAssessment(
                criterion_index=2,
                status="partial",
                explanation="Раскрыто частично.",
                answer_evidence="второй пункт",
            ),
            CriterionAssessment(
                criterion_index=3,
                status="missing",
                explanation="В ответе отсутствует.",
            ),
        ]
    )
    evaluation = build_answer_evaluation(
        ["Критерий 1", "Критерий 2", "Критерий 3"],
        judgement,
        "Раскрыт первый пункт и упомянут второй пункт.",
        "Цитата из материала.",
    )
    assert evaluation.score == 50
    assert evaluation.follow_up_needed
    assert evaluation.strengths == ["Критерий 1"]
    assert evaluation.gaps == ["Критерий 2", "Критерий 3"]


def test_complete_answer_always_scores_100() -> None:
    judgement = EvaluationJudgement(
        criteria=[
            CriterionAssessment(
                criterion_index=index,
                status="met",
                explanation="Раскрыто.",
            )
            for index in range(1, 4)
        ]
    )
    evaluation = build_answer_evaluation(
        ["A", "B", "C"], judgement, "Полный ответ.", "Источник."
    )
    assert evaluation.score == 100
    assert not evaluation.follow_up_needed
    assert evaluation.gaps == []


def test_english_evaluation_feedback() -> None:
    judgement = EvaluationJudgement(
        criteria=[
            CriterionAssessment(
                criterion_index=1,
                status="met",
                explanation="Covered.",
            ),
            CriterionAssessment(
                criterion_index=2,
                status="met",
                explanation="Covered.",
            ),
        ]
    )
    evaluation = build_answer_evaluation(
        ["Criterion A", "Criterion B"],
        judgement,
        "Complete answer.",
        "Source evidence.",
        language="en",
    )
    assert evaluation.score == 100
    assert evaluation.feedback.startswith("The answer")


def test_ui_translations_have_matching_keys() -> None:
    assert set(TRANSLATIONS["ru"]) == set(TRANSLATIONS["en"])
    assert text("ru", "start_button") == "Начать экзамен"
    assert text("en", "start_button") == "Start the exam"


def test_incomplete_criterion_judgement_is_rejected() -> None:
    judgement = EvaluationJudgement(
        criteria=[
            CriterionAssessment(
                criterion_index=1,
                status="met",
                explanation="Раскрыто.",
            )
        ]
    )
    with pytest.raises(ValueError, match="не все критерии"):
        build_answer_evaluation(
            ["A", "B"], judgement, "Ответ.", "Источник."
        )


def test_build_summary_aggregates_scores_and_gaps() -> None:
    question = Question(
        text="Почему accuracy недостаточно при дисбалансе классов?",
        topic="Метрики",
        expected_points=["дисбаланс", "F1"],
        source_excerpt="Accuracy может вводить в заблуждение.",
    )
    evaluation = AnswerEvaluation(
        score=60,
        feedback="Частичный ответ.",
        strengths=["Упомянут дисбаланс"],
        gaps=["Повторить precision и recall"],
        evidence_excerpt="Полезнее анализировать precision, recall и F1.",
        follow_up_needed=True,
        follow_up_question="Чем отличаются precision и recall?",
    )
    summary = build_summary(
        [
            SessionRecord(
                question=question,
                answer="Accuracy скрывает редкий класс.",
                evaluation=evaluation,
            )
        ]
    )
    assert summary.overall_score == 60
    assert summary.weak_topics == ["Метрики"]
    assert summary.revision_plan == ["Повторить precision и recall"]


def test_new_topic_selection_excludes_used_source() -> None:
    passages = [
        "Классификация использует accuracy и F1.",
        "Переобучение ухудшает качество на новых данных.",
        "Утечка данных искажает тестовую оценку.",
    ]
    question = Question(
        text="Какие метрики используют?",
        topic="Метрики",
        expected_points=["Accuracy", "F1"],
        source_excerpt=passages[0],
    )
    evaluation = AnswerEvaluation(
        score=100,
        feedback="Полный ответ.",
        evidence_excerpt=passages[0],
    )
    selected = select_next_passages(
        [
            SessionRecord(
                question=question,
                answer="Accuracy и F1.",
                evaluation=evaluation,
            )
        ],
        passages,
    )
    assert passages[0] not in selected
    assert set(selected) == set(passages[1:])


def test_demo_question_is_bilingual() -> None:
    assert "accuracy" in demo_question("en").text.lower()
    assert "accuracy" in demo_question("ru").text.lower()
    assert demo_question("en", "basic").topic == "Supervised learning tasks"


def test_short_passages_remain_available_as_sources() -> None:
    material = "A" * 60 + "\n\n" + "B" * 60 + "\n\n" + "C" * 60
    coach = VivaCoach(
        material,
        LLMConfig(
            base_url="http://localhost:11434/v1",
            model="test-model",
            api_key="",
        ),
    )
    assert coach.source_passages == ["A" * 60, "B" * 60, "C" * 60]


def test_markdown_report_contains_results() -> None:
    question = Question(
        text="Что такое классификация?",
        topic="Классификация",
        expected_points=["Категория", "Размеченные данные"],
        source_excerpt="Классификация предсказывает категорию.",
    )
    evaluation = AnswerEvaluation(
        score=75,
        feedback="Хорошо.",
        gaps=["Размеченные данные"],
        evidence_excerpt=question.source_excerpt,
    )
    report = build_markdown_report(
        [
            SessionRecord(
                question=question,
                answer="Предсказание категории.",
                evaluation=evaluation,
            )
        ],
        "Конспект",
    )
    assert "# Viva AI — Отчёт" in report
    assert "Классификация — 75/100" in report
