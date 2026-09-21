from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
import re

import pytest

from agent import tools
from agent.types import AgentError


@pytest.fixture(autouse=True)
def calendar(monkeypatch):
    monkeypatch.setattr(tools, "_BOOKED", set())
    monkeypatch.setattr(tools, "_TICKETS", set())
    monkeypatch.setattr(tools, "_today", lambda: date(2026, 9, 25))  # пятница


@pytest.mark.parametrize("preferred,expected", [
    (None, "2026-09-28"), ("2026-09-01", "2026-09-28"),
    ("2026-10-03", "2026-10-05"), ("2026-09-30", "2026-09-30"),
])
def test_booking_dates_and_reproducible_ticket(preferred, expected):
    first = tools.book_appointment(service="Справка", preferred_date=preferred)
    assert first.slot_iso == expected + "T09:00:00+05:00"
    assert re.fullmatch(r"[ABC]-\d{3}", first.ticket)
    tools._BOOKED.clear()
    tools._TICKETS.clear()
    assert tools.book_appointment(service="Справка", preferred_date=preferred) == first


def test_full_day_and_concurrent_reservations():
    # 14 интервалов × 3 отделения; 43-я запись должна перейти на вторник.
    with ThreadPoolExecutor(max_workers=4) as pool:
        bookings = list(pool.map(lambda _: tools.book_appointment(service="Справка"), range(43)))
    assert len({(b.office, b.slot_iso) for b in bookings}) == 43
    assert len({(b.slot_iso[:10], b.ticket) for b in bookings}) == 43
    assert sum(b.slot_iso.startswith("2026-09-28") for b in bookings) == 42
    assert max(b.slot_iso for b in bookings) == "2026-09-29T09:00:00+05:00"
    for booking in bookings:
        slot = datetime.fromisoformat(booking.slot_iso)
        assert slot.weekday() < 5 and 9 <= slot.hour < 17 and slot.hour != 13
        assert slot.minute in (0, 30)


@pytest.mark.parametrize("kwargs", [
    {"service": " "}, {"service": "x", "preferred_date": "20260928"},
    {"service": "x", "preferred_date": "2026-02-30"},
])
def test_invalid_booking(kwargs):
    with pytest.raises(AgentError):
        tools.book_appointment(**kwargs)
