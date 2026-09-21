"""Тесты оркестратора: порядок шагов, ветка записи, лог при падении."""

from __future__ import annotations

import json
from datetime import date

import pytest

from agent.pipeline import extract_preferred_date, run
from agent.trace import JsonTracer
from agent.types import Appointment, Clause, LLMError
from tests.fakes import FakeLLM

CLAUSES = [
    Clause(id="2.1", title="Виды справок", text="справка о составе семьи", score=0.5),
    Clause(id="2.3", title="Срок", text="три рабочих дня", score=0.4),
    Clause(id="2.2", title="Документы", text="удостоверение личности", score=0.3),
]


class StubRetriever:
    def __init__(self, low: bool = False) -> None:
        self.low = low
        self.queries: list[str] = []

    def search(self, query: str, k: int = 3) -> list[Clause]:
        self.queries.append(query)
        return CLAUSES[:k]

    def is_low_confidence(self, found: list[Clause]) -> bool:
        return self.low


def _llm(kind: str = "справка") -> FakeLLM:
    return FakeLLM(
        [
            json.dumps({"type": kind, "confidence": 0.9, "reason": "r"}, ensure_ascii=False),
            "Уважаемый заявитель! Согласно п. 2.3 срок составляет три рабочих дня.",
            "Құрметті өтініш беруші! п. 2.3 бойынша мерзім үш жұмыс күні.",
        ]
    )


def _tracer(tmp_path) -> JsonTracer:
    return JsonTracer("обращение", "fake-model", log_path=str(tmp_path / "log.json"))


def test_полный_прогон_возвращает_результат(tmp_path):
    result = run(
        "Прошу выдать справку",
        llm=_llm(),
        retriever=StubRetriever(),
        tracer=_tracer(tmp_path),
    )
    assert result.classification.type == "справка"
    assert len(result.clauses) == 3
    assert "п. 2.3" in result.draft.ru
    assert "п. 2.3" in result.draft.kk
    assert result.draft.cited == ["2.3"]
    assert result.appointment is None
    assert result.duration_ms >= 0


def test_лог_содержит_все_шаги_по_порядку(tmp_path):
    result = run(
        "Прошу выдать справку",
        llm=_llm(),
        retriever=StubRetriever(),
        tracer=_tracer(tmp_path),
    )
    data = json.loads(open(result.trace_path, encoding="utf-8").read())
    assert [s["action"] for s in data["steps"]] == [
        "classify",
        "retrieve",
        "draft_ru",
        "translate_kk",
    ]
    assert [s["step"] for s in data["steps"]] == [1, 2, 3, 4]
    assert data["request_id"] == result.request_id


def test_ветка_записи_вызывает_бронирование(tmp_path):
    booked = Appointment(
        slot_iso="2026-09-24T10:30:00+05:00",
        office="Есильское отделение",
        service="запись на приём",
        ticket="A-042",
    )
    calls: list[dict] = []

    def fake_book(**kwargs) -> Appointment:
        calls.append(kwargs)
        return booked

    result = run(
        "Хочу записаться на приём",
        llm=_llm("запись"),
        retriever=StubRetriever(),
        tracer=_tracer(tmp_path),
        book=fake_book,
    )
    assert result.appointment == booked
    assert len(calls) == 1
    data = json.loads(open(result.trace_path, encoding="utf-8").read())
    assert [t["name"] for t in data["tool_calls"]] == ["book_appointment"]


def test_справка_не_вызывает_бронирование(tmp_path):
    def fake_book(**kwargs):
        raise AssertionError("бронирование не должно вызываться для справки")

    run(
        "Прошу выдать справку",
        llm=_llm(),
        retriever=StubRetriever(),
        tracer=_tracer(tmp_path),
        book=fake_book,
    )


def test_слабый_поиск_прокидывается_в_черновик_и_лог(tmp_path):
    llm = _llm()
    result = run(
        "непонятное обращение",
        llm=llm,
        retriever=StubRetriever(low=True),
        tracer=_tracer(tmp_path),
    )
    assert "уточнение" in llm.calls[1]["user"]
    data = json.loads(open(result.trace_path, encoding="utf-8").read())
    assert data["steps"][1]["output"]["low_confidence"] is True


def test_падение_модели_сохраняет_лог_и_прокидывает_ошибку(tmp_path):
    tracer = _tracer(tmp_path)
    with pytest.raises(LLMError):
        run(
            "Прошу выдать справку",
            llm=FakeLLM([LLMError("провайдер недоступен")]),
            retriever=StubRetriever(),
            tracer=tracer,
        )
    data = json.loads(open(tracer.log_path, encoding="utf-8").read())
    assert data["steps"][0]["action"] == "classify"
    assert "error" in data["steps"][0]
    assert "output" not in data["steps"][0]


def test_падение_на_переводе_отмечает_нужный_шаг(tmp_path):
    tracer = _tracer(tmp_path)
    llm = FakeLLM(
        [
            json.dumps({"type": "справка", "confidence": 0.9, "reason": "r"}, ensure_ascii=False),
            "Согласно п. 2.3 срок три дня.",
            "Аударма без ссылок.",
            "Аударма снова без ссылок.",
        ]
    )
    with pytest.raises(LLMError):
        run("Прошу справку", llm=llm, retriever=StubRetriever(), tracer=tracer)
    data = json.loads(open(tracer.log_path, encoding="utf-8").read())
    assert data["steps"][-1]["action"] == "translate_kk"
    assert "error" in data["steps"][-1]


def test_тип_обращения_подмешивается_в_поисковый_запрос(tmp_path):
    """Иначе смешанное обращение находит пункты только одного раздела."""
    retriever = StubRetriever()
    run(
        "Хочу записаться, чтобы получить справку",
        llm=_llm("запись"),
        retriever=retriever,
        tracer=_tracer(tmp_path),
        book=lambda **kw: Appointment(
            slot_iso="2026-09-24T10:30:00+05:00",
            office="Есильское",
            service="приём",
            ticket="A-001",
        ),
    )
    assert "талон" in retriever.queries[0], "подсказка по типу не добавлена"
    assert "получить справку" in retriever.queries[0], "текст обращения потерян"


def test_пустое_обращение_это_ошибка(tmp_path):
    with pytest.raises(ValueError):
        run("   ", llm=_llm(), retriever=StubRetriever(), tracer=_tracer(tmp_path))


def test_дата_из_обращения_передаётся_в_бронирование(tmp_path):
    args: list[dict] = []
    run(
        "Хочу записаться на приём 24.09.2026",
        llm=_llm("запись"),
        retriever=StubRetriever(),
        tracer=_tracer(tmp_path),
        book=lambda **kw: (
            args.append(kw)
            or Appointment(
                slot_iso="2026-09-24T10:30:00+05:00",
                office="Есильское",
                service="приём",
                ticket="A-001",
            )
        ),
    )
    assert args[0]["preferred_date"] == "2026-09-24"


def test_без_даты_в_обращении_параметр_не_передаётся(tmp_path):
    args: list[dict] = []
    run(
        "Хочу записаться на приём",
        llm=_llm("запись"),
        retriever=StubRetriever(),
        tracer=_tracer(tmp_path),
        book=lambda **kw: (
            args.append(kw)
            or Appointment(
                slot_iso="2026-09-24T10:30:00+05:00",
                office="Есильское",
                service="приём",
                ticket="A-001",
            )
        ),
    )
    assert "preferred_date" not in args[0]


def test_ошибка_загрузки_зависимостей_попадает_в_лог(tmp_path):
    """Иначе сбой конфигурации не оставляет следа в трассировке."""

    class BrokenRetriever:
        def search(self, query, k=3):
            raise RuntimeError("индекс не собрался")

    tracer = _tracer(tmp_path)
    with pytest.raises(RuntimeError):
        run("Прошу справку", llm=_llm(), retriever=BrokenRetriever(), tracer=tracer)
    data = json.loads(open(tracer.log_path, encoding="utf-8").read())
    assert data["steps"][-1]["action"] == "retrieve"
    assert "индекс не собрался" in data["steps"][-1]["error"]


def test_извлечение_даты_понимает_форматы():
    today = date(2026, 9, 21)
    assert extract_preferred_date("приём 2026-09-24", today) == "2026-09-24"
    assert extract_preferred_date("запишите на 24.09.2026", today) == "2026-09-24"
    assert extract_preferred_date("запишите на 24.09", today) == "2026-09-24"
    assert extract_preferred_date("хочу 24 сентября", today) == "2026-09-24"
    assert extract_preferred_date("можно 3 февраля?", today) == "2027-02-03"
    assert extract_preferred_date("хочу на приём", today) is None
    assert extract_preferred_date("31.02.2026", today) is None
