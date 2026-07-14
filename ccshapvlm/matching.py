"""Decide whether a model prediction counts as correct for a task."""

from __future__ import annotations

from .config import is_multiple_choice


def evaluate_prediction(prediction: str, correct_answer: str | list[str],
                        task: str) -> int:
    if is_multiple_choice(task):
        return int(prediction == correct_answer)
    prediction = prediction.lower()
    if isinstance(correct_answer, list):
        return int(any(answer.lower() in prediction for answer in correct_answer))
    return int(correct_answer in prediction)
