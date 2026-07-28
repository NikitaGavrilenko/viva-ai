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
        language="en",
    )
    question = Question(
        text=(
            "Why can accuracy be misleading with a severe class imbalance, "
            "and which metrics should be analyzed instead?"
        ),
        topic="Classification metrics under class imbalance",
        expected_points=[
            "The majority class dominates the total fraction of correct answers",
            (
                "Accuracy can be high even when the model fails to identify "
                "the rare positive class"
            ),
            "Precision, recall, F1, and the confusion matrix should be analyzed",
        ],
        source_excerpt=(
            "With a severe class imbalance, accuracy can be misleading"
        ),
    )
    answers = [
        ("none", "I do not know the answer."),
        (
            "vague",
            "The classes are imbalanced, so accuracy does not work well.",
        ),
        (
            "partial",
            "The majority class dominates, so accuracy can remain high even "
            "when the model rarely identifies the minority class.",
        ),
        (
            "full",
            "Accuracy measures the total fraction of correct predictions. If "
            "95% of examples belong to the majority class, a model can always "
            "predict that class and receive 95% accuracy while completely "
            "missing the rare positive class. Precision, recall, F1, and the "
            "confusion matrix should be examined.",
        ),
        (
            "full_repeat",
            "Accuracy measures the total fraction of correct predictions. If "
            "95% of examples belong to the majority class, a model can always "
            "predict that class and receive 95% accuracy while completely "
            "missing the rare positive class. Precision, recall, F1, and the "
            "confusion matrix should be examined.",
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
