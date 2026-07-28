"""Manual local-model regression check for the grading rubric."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from viva_core import LLMConfig, Question, VivaCoach  # noqa: E402


def main() -> int:
    material = (
        PROJECT_DIR / "demo_materials" / "machine_learning_basics.txt"
    ).read_text(encoding="utf-8")
    coach = VivaCoach(
        material,
        LLMConfig(
            api_key="",
            base_url="http://localhost:11434/v1",
            model="qwen3:14b",
            timeout_seconds=120,
        ),
    )
    question = Question(
        text=(
            "Почему accuracy может вводить в заблуждение при сильном "
            "дисбалансе классов и какие метрики лучше анализировать?"
        ),
        topic="Метрики классификации при дисбалансе",
        expected_points=[
            "Большой класс доминирует в общей доле правильных ответов",
            (
                "Accuracy может быть высокой, даже если модель плохо "
                "находит редкий положительный класс"
            ),
            "Нужно анализировать precision, recall, F1 и матрицу ошибок",
        ],
        source_excerpt=(
            "При сильном дисбалансе классов accuracy может вводить "
            "в заблуждение."
        ),
    )
    answers = [
        ("none", "Я не знаю ответа на этот вопрос."),
        (
            "vague",
            "Потому что классы несбалансированы и accuracy работает плохо.",
        ),
        (
            "partial",
            "Большой класс доминирует, поэтому accuracy может быть высокой, "
            "хотя редкий класс модель почти не находит.",
        ),
        (
            "full",
            "Accuracy считает общую долю верных ответов, поэтому при 95% "
            "объектов большинства модель может всегда выбирать этот класс "
            "и получить 95%, полностью пропустив редкий положительный класс. "
            "Нужно смотреть precision, recall, F1 и матрицу ошибок.",
        ),
        (
            "full_repeat",
            "Accuracy считает общую долю верных ответов, поэтому при 95% "
            "объектов большинства модель может всегда выбирать этот класс "
            "и получить 95%, полностью пропустив редкий положительный класс. "
            "Нужно смотреть precision, recall, F1 и матрицу ошибок.",
        ),
    ]

    scores: list[int] = []
    for label, answer in answers:
        result = coach.evaluate(question, answer)
        scores.append(result.score)
        print(
            json.dumps(
                {
                    "label": label,
                    "score": result.score,
                    "statuses": [
                        item.status for item in result.criterion_results
                    ],
                    "gaps": result.gaps,
                },
                ensure_ascii=False,
            )
        )

    passed = (
        scores == sorted(scores)
        and scores[0] == 0
        and 40 <= scores[2] <= 80
        and scores[-2:] == [100, 100]
    )
    print(
        json.dumps(
            {
                "scores": scores,
                "non_decreasing": scores == sorted(scores),
                "partial_in_expected_range": 40 <= scores[2] <= 80,
                "full_is_stable_100": scores[-2:] == [100, 100],
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
