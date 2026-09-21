"""Общие подделки для тестов (SPEC.md, раздел 5).

Здесь живут реализации протоколов из `agent/types.py`, чтобы тесты не
ходили в сеть. `conftest.py` оборачивает их в фикстуры.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FakeLLM:
    """Отдаёт заранее заданные ответы по очереди.

    Не привязан к промптам: сохраняет все вызовы в `calls`, чтобы тест
    мог проверить, что именно ушло в модель.
    """

    def __init__(self, responses: list[str] | str) -> None:
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "json_schema": json_schema,
            }
        )
        if not self.responses:
            raise AssertionError("FakeLLM: запросов больше, чем заготовленных ответов")
        return self.responses.pop(0)


class FakeEmbedder:
    """Эмбеддер по ключевым словам: вектор — маска попаданий.

    Нужен, чтобы тестировать сам ретривер, а не качество модели.
    """

    min_score = 0.1

    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.keywords)))
        for row, text in enumerate(texts):
            lowered = text.lower()
            for col, word in enumerate(self.keywords):
                matrix[row, col] = float(lowered.count(word))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-9)
