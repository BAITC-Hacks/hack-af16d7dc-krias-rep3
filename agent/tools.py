"""Демонстрационный календарь приёма: состояние только в памяти процесса."""

from datetime import date, datetime, time, timedelta
from hashlib import sha256
from threading import Lock
from zoneinfo import ZoneInfo

from agent.types import AgentError, Appointment

TZ = ZoneInfo("Asia/Almaty")
OFFICES = ("Астана — Есиль", "Астана — Алматы", "Астана — Сарыарка")
_LOCK = Lock()
_BOOKED: set[tuple[str, str]] = set()
_TICKETS: set[tuple[str, str]] = set()


def _today() -> date:
    return datetime.now(TZ).date()


def _workday(day: date) -> date:
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def book_appointment(*, service: str, preferred_date: str | None = None) -> Appointment:
    service = service.strip()
    if not service:
        raise AgentError("Для записи нужно указать услугу")
    day = _workday(_today() + timedelta(days=1))
    if preferred_date is not None:
        try:
            preferred = date.fromisoformat(preferred_date)
            if preferred.isoformat() != preferred_date:
                raise ValueError
        except (ValueError, TypeError):
            raise AgentError("Дата записи должна быть в формате YYYY-MM-DD") from None
        day = _workday(max(day, preferred))
    seed = int.from_bytes(sha256(service.encode("utf-8")).digest()[:4], "big") % 1000
    with _LOCK:
        while True:
            for hour in range(9, 17):
                if hour == 13:
                    continue
                for minute in (0, 30):
                    slot = datetime.combine(day, time(hour, minute), TZ).isoformat()
                    for index, office in enumerate(OFFICES):
                        if (office, slot) in _BOOKED:
                            continue
                        number = seed
                        ticket = f"{chr(65 + index)}-{number:03d}"
                        while (day.isoformat(), ticket) in _TICKETS:
                            number = (number + 1) % 1000
                            ticket = f"{chr(65 + index)}-{number:03d}"
                        _BOOKED.add((office, slot))
                        _TICKETS.add((day.isoformat(), ticket))
                        return Appointment(slot_iso=slot, office=office, service=service, ticket=ticket)
            day = _workday(day + timedelta(days=1))


TOOL_SPECS = [{
    "type": "function",
    "function": {
        "name": "book_appointment",
        "description": "Демонстрационная запись на приём. Возвращает фактическую дату, отделение и талон; занятые дни и выходные пропускаются.",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "minLength": 1, "description": "Запрашиваемая услуга"},
                "preferred_date": {"type": ["string", "null"], "format": "date", "description": "Желаемая дата YYYY-MM-DD, если известна"},
            },
            "required": ["service"],
            "additionalProperties": False,
        },
    },
}]
