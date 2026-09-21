"""Офлайн-демо: полный прогон агента без обращения к модели.

Нужен для двух вещей: показать заполненный logs/agent_log.json, когда
ключа под рукой нет, и быстро проверить, что половины проекта стыкуются.
Ответы модели заранее заданы, поиск лексический, инструмент бронирования
настоящий.

    python scripts/demo_offline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import run  # noqa: E402
from agent.reglament import parse_reglament  # noqa: E402
from agent.retrieve import LexicalEmbedder, ReglamentRetriever  # noqa: E402
from agent.trace import JsonTracer  # noqa: E402
from tests.fakes import FakeLLM  # noqa: E402

TEXT = "Хочу записаться на приём 24 сентября для получения справки о составе семьи"

ANSWERS = [
    json.dumps(
        {
            "type": "запись",
            "confidence": 0.88,
            "reason": "заявитель просит время приёма",
        },
        ensure_ascii=False,
    ),
    "Уважаемый заявитель!\n\nВаша запись на приём оформлена. Согласно п. 3.2 "
    "регламента приём ведётся в интервалах по тридцать минут с 09:00 до 17:00, "
    "перерыв с 13:00 до 14:00. Запись подтверждена талоном, указанным ниже; "
    "согласно п. 3.3 перенести или отменить запись можно не позднее чем за два "
    "часа до начала приёма.\n\nУправление обслуживания населения города Астаны",
    "Құрметті өтініш беруші!\n\nСіздің қабылдауға жазылуыңыз рәсімделді. "
    "Регламенттің п. 3.2 бойынша қабылдау 09:00-ден 17:00-ге дейін отыз "
    "минуттық аралықтармен жүргізіледі, үзіліс 13:00-14:00. п. 3.3 бойынша "
    "жазылуды қабылдау басталуына екі сағат қалғанға дейін ауыстыруға болады."
    "\n\nАстана қаласы халыққа қызмет көрсету басқармасы",
]


def main() -> int:
    clauses = parse_reglament()
    retriever = ReglamentRetriever(
        clauses=clauses,
        embedder=LexicalEmbedder([f"{c.title}. {c.text}" for c in clauses]),
    )
    result = run(
        TEXT,
        llm=FakeLLM(ANSWERS),
        retriever=retriever,
        tracer=JsonTracer(TEXT, "offline-demo"),
    )
    print(f"Тип: {result.classification.type} ({result.classification.confidence:.0%})")
    for clause in result.clauses:
        print(f"  п. {clause.id} ({clause.score:.2f}) — {clause.title}")
    if result.appointment:
        print(
            f"Талон: {result.appointment.ticket}, {result.appointment.slot_iso}, "
            f"{result.appointment.office}"
        )
    print(f"Ссылки в ответе: {result.draft.cited}")
    print(f"Лог: {result.trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
