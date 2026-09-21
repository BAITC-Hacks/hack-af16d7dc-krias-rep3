"""Классификация обращения: справка / жалоба / запись.

Решение принимает только модель. Предварительного отсева по ключевым
словам сознательно нет: обращения вида «жалуюсь на отказ и заодно прошу
справку» ключевые слова относят к первому совпадению, а не к основному
намерению заявителя.
"""

from __future__ import annotations

import json
import re

from agent.types import REQUEST_TYPES, Classification, LLMClient, LLMError

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")

SYSTEM = """Ты делопроизводитель управления обслуживания населения.
Определи тип обращения гражданина. Возможны только три типа:

- "справка" — заявитель просит выдать документ или сведения;
- "жалоба" — заявитель не согласен с действиями органа или работника,
  сообщает о нарушении, требует проверки;
- "запись" — заявитель хочет попасть на приём, спрашивает о времени,
  переносит или отменяет визит.

Если в обращении несколько намерений, выбери основное — то, ради чего
обращение написано. Отвечай только JSON без пояснений вокруг."""

SCHEMA = {
    "name": "classification",
    "schema": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": list(REQUEST_TYPES)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["type", "confidence", "reason"],
        "additionalProperties": False,
    },
}


def _parse(raw: str) -> dict:
    """Разбор ответа модели. Снимает markdown-заборы: их ставят почти все."""
    cleaned = FENCE.sub("", raw.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"классификатор вернул не JSON: {raw[:200]!r}") from exc
    if not isinstance(data, dict):
        raise LLMError(f"ожидался объект, получено {type(data).__name__}")
    return data


def classify(text: str, llm: LLMClient) -> Classification:
    """Определяет тип обращения.

    Raises:
        ValueError: пустое обращение.
        LLMError: модель вернула неразбираемый ответ или неизвестный тип.
    """
    if not text.strip():
        raise ValueError("пустое обращение")

    raw = llm.complete(
        system=SYSTEM,
        user=f"Обращение гражданина:\n\n{text.strip()}",
        temperature=0.0,
        json_schema=SCHEMA,
    )
    data = _parse(raw)

    kind = str(data.get("type", "")).strip().lower()
    if kind not in REQUEST_TYPES:
        raise LLMError(
            f"неизвестный тип обращения {kind!r}, допустимы {list(REQUEST_TYPES)}"
        )

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return Classification(
        type=kind,  # type: ignore[arg-type]
        confidence=min(1.0, max(0.0, confidence)),
        reason=str(data.get("reason", "")).strip() or "обоснование не указано",
    )
