"""Тесты классификации обращения."""

from __future__ import annotations

import json

import pytest

from agent.classify import classify
from agent.types import LLMError
from tests.fakes import FakeLLM


def _answer(kind: str, confidence: float = 0.9, reason: str = "явный запрос") -> str:
    return json.dumps(
        {"type": kind, "confidence": confidence, "reason": reason},
        ensure_ascii=False,
    )


def test_разбирает_корректный_ответ_модели():
    llm = FakeLLM([_answer("справка")])
    result = classify("Прошу выдать справку о составе семьи", llm)
    assert result.type == "справка"
    assert result.confidence == 0.9
    assert result.reason == "явный запрос"


def test_все_три_типа_проходят():
    for kind in ("справка", "жалоба", "запись"):
        assert classify("текст обращения", FakeLLM([_answer(kind)])).type == kind


def test_снимает_markdown_забор():
    """Модели почти всегда оборачивают JSON в ```json."""
    llm = FakeLLM(["```json\n" + _answer("жалоба") + "\n```"])
    assert classify("не согласен с отказом", llm).type == "жалоба"


def test_обращение_передаётся_в_промпт_и_схема_тоже():
    llm = FakeLLM([_answer("запись")])
    classify("Хочу записаться на приём", llm)
    call = llm.calls[0]
    assert "Хочу записаться на приём" in call["user"]
    assert call["json_schema"] is not None
    assert call["temperature"] == 0.0, "классификация должна быть детерминированной"


def test_неизвестный_тип_это_ошибка():
    llm = FakeLLM([_answer("консультация")])
    with pytest.raises(LLMError):
        classify("текст", llm)


def test_не_json_это_ошибка():
    with pytest.raises(LLMError):
        classify("текст", FakeLLM(["Думаю, это справка."]))


def test_json_не_объект_это_ошибка():
    with pytest.raises(LLMError):
        classify("текст", FakeLLM(['["справка"]']))


def test_уверенность_вне_диапазона_прижимается():
    assert classify("текст", FakeLLM([_answer("справка", 7.5)])).confidence == 1.0
    assert classify("текст", FakeLLM([_answer("справка", -3)])).confidence == 0.0


def test_нечисловая_уверенность_не_роняет_разбор():
    llm = FakeLLM(['{"type": "жалоба", "confidence": "высокая", "reason": "r"}'])
    assert classify("текст", llm).confidence == 0.0


def test_пустое_обоснование_заполняется():
    llm = FakeLLM([_answer("справка", 0.8, "   ")])
    assert classify("текст", llm).reason == "обоснование не указано"


def test_пустое_обращение_это_ошибка():
    with pytest.raises(ValueError):
        classify("   ", FakeLLM([_answer("справка")]))


def test_падение_модели_прокидывается():
    llm = FakeLLM([LLMError("провайдер недоступен")])
    with pytest.raises(LLMError):
        classify("текст", llm)
