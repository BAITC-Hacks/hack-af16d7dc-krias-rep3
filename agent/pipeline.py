"""Оркестратор: собирает шаги агента и пишет трассировку.

Обычная функция, а не граф: на четырёх шагах граф прячет логику, а по
заданию главное доказательство агентности — читаемый лог.
"""

from __future__ import annotations

import os
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


def _ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


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

    if llm is None:
        from agent.llm import get_client

        llm = get_client()
    if retriever is None:
        retriever = ReglamentRetriever()
    if tracer is None:
        from agent.trace import new_tracer

        tracer = new_tracer(text, os.getenv("LLM_MODEL", "unknown"))

    action = "classify"
    try:
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
            appointment = book(service="запись на приём")
            tracer.tool_call(
                "book_appointment",
                args={"service": "запись на приём"},
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
