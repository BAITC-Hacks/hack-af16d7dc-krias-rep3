"""Оркестратор: собирает шаги агента и пишет трассировку.

Обычная функция, а не граф: на четырёх шагах граф прячет логику, а по
заданию главное доказательство агентности — читаемый лог.
"""

from __future__ import annotations

import os
import re
from datetime import date
from time import perf_counter
from typing import Callable

from agent.classify import classify
from agent.draft import build_draft, draft_ru, translate_kk
from agent.retrieve import ReglamentRetriever, build_query
from agent.types import (
    AgentResult,
    Appointment,
    LLMClient,
    Retriever,
    Tracer,
)

Booker = Callable[..., Appointment]


MONTHS = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "ма": 5, "июн": 6,
    "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
}
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DOTTED_DATE = re.compile(r"\b(\d{1,2})[.](\d{1,2})(?:[.](\d{4}|\d{2}))?\b")
WORDY_DATE = re.compile(r"\b(\d{1,2})\s+([а-яё]{3,})", re.IGNORECASE)


def _ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def extract_preferred_date(text: str, today: date | None = None) -> str | None:
    """Достаёт желаемую дату приёма из обращения в формате YYYY-MM-DD.

    Понимает «2026-09-24», «24.09», «24.09.2026» и «24 сентября».
    Год без указания берётся текущий, а если дата уже прошла — следующий:
    «запишите на 3 февраля», сказанное в декабре, означает февраль
    следующего года.
    """
    today = today or date.today()

    match = ISO_DATE.search(text)
    if match:
        year, month, day = (int(g) for g in match.groups())
        return _safe_date(year, month, day)

    match = DOTTED_DATE.search(text)
    if match:
        day, month, year_raw = match.group(1), match.group(2), match.group(3)
        year = int(year_raw) if year_raw else None
        if year is not None and year < 100:
            year += 2000
        return _with_year(int(day), int(month), year, today)

    match = WORDY_DATE.search(text)
    if match:
        day, word = int(match.group(1)), match.group(2).lower()
        for prefix, month in MONTHS.items():
            if word.startswith(prefix):
                return _with_year(day, month, None, today)
    return None


def _safe_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _with_year(day: int, month: int, year: int | None, today: date) -> str | None:
    if year is not None:
        return _safe_date(year, month, day)
    guess = _safe_date(today.year, month, day)
    if guess is None:
        return None
    return guess if date.fromisoformat(guess) >= today else _safe_date(
        today.year + 1, month, day
    )


def run(
    text: str,
    *,
    llm: LLMClient | None = None,
    retriever: Retriever | None = None,
    tracer: Tracer | None = None,
    book: Booker | None = None,
) -> AgentResult:
    """Прогоняет обращение через агента.

    Зависимости можно подменить — так пайплайн тестируется без сети.
    По умолчанию берутся реальные реализации.

    Raises:
        ValueError: пустое обращение.
        LLMError: провайдер недоступен или ответ не прошёл проверку.
    """
    if not text.strip():
        raise ValueError("пустое обращение")

    started = perf_counter()

    # Трассировщик создаётся первым: ошибки конфигурации модели и загрузки
    # регламента тоже должны попасть в лог, а не пропасть до его открытия.
    if tracer is None:
        from agent.trace import new_tracer

        tracer = new_tracer(text, os.getenv("LLM_MODEL", "unknown"))

    action = "setup"
    try:
        step = perf_counter()
        if llm is None:
            from agent.llm import get_client

            llm = get_client()
        if retriever is None:
            retriever = ReglamentRetriever()

        action = "classify"
        step = perf_counter()
        classification = classify(text, llm)
        tracer.step(
            "classify",
            inp={"text": text},
            out={
                "type": classification.type,
                "confidence": classification.confidence,
                "reason": classification.reason,
            },
            duration_ms=_ms(step),
        )

        action = "retrieve"
        step = perf_counter()
        # Тип подмешивается в запрос подсказкой из предметной лексики:
        # смешанное обращение («записаться, чтобы получить справку») иначе
        # находит только пункты про справки, и черновику нечем обосновать
        # запись на приём.
        query = build_query(classification.type, text)
        clauses = retriever.search(query, k=3)
        low = bool(getattr(retriever, "is_low_confidence", lambda _: False)(clauses))
        tracer.step(
            "retrieve",
            inp={"query": query, "k": 3},
            out={
                "clauses": [c.id for c in clauses],
                "scores": [c.score for c in clauses],
                "low_confidence": low,
            },
            duration_ms=_ms(step),
        )

        appointment: Appointment | None = None
        if classification.type == "запись":
            action = "book_appointment"
            if book is None:
                from agent.tools import book_appointment

                book = book_appointment
            step = perf_counter()
            args: dict[str, str] = {"service": "запись на приём"}
            preferred = extract_preferred_date(text)
            if preferred:
                args["preferred_date"] = preferred
            appointment = book(**args)
            tracer.tool_call(
                "book_appointment",
                args=args,
                result={
                    "slot_iso": appointment.slot_iso,
                    "office": appointment.office,
                    "ticket": appointment.ticket,
                },
                duration_ms=_ms(step),
            )

        action = "draft_ru"
        step = perf_counter()
        ru = draft_ru(
            text,
            classification,
            clauses,
            llm,
            low_confidence=low,
            appointment=appointment,
        )
        tracer.step(
            "draft_ru",
            inp={"type": classification.type, "clauses": [c.id for c in clauses]},
            out={"text": ru},
            duration_ms=_ms(step),
        )

        action = "translate_kk"
        step = perf_counter()
        kk = translate_kk(ru, llm)
        tracer.step(
            "translate_kk",
            inp={"chars_ru": len(ru)},
            out={"text": kk},
            duration_ms=_ms(step),
        )
    except Exception as exc:
        # Лог обязан сохраниться даже при падении: SPEC 3.2.
        tracer.error(action, inp={"text": text}, message=str(exc))
        raise

    draft = build_draft(ru, kk)
    trace_path = tracer.save()

    return AgentResult(
        request_id=tracer.request_id,
        input_text=text,
        classification=classification,
        clauses=clauses,
        draft=draft,
        appointment=appointment,
        trace_path=trace_path,
        duration_ms=_ms(started),
    )
