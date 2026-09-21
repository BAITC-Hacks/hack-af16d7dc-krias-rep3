"""Контракты AI-секретаря.

ЕДИНСТВЕННЫЙ источник истины для обеих половин проекта.
Менять только через правку SPEC.md (см. раздел 6) — иначе рассыпается
сборка на стыке модулей.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# --- Тип обращения -----------------------------------------------------------

RequestType = Literal["справка", "жалоба", "запись"]

REQUEST_TYPES: tuple[RequestType, ...] = ("справка", "жалоба", "запись")

# --- Данные ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Clause:
    """Пункт регламента."""

    id: str  # "3.4"
    title: str
    text: str
    score: float = 0.0  # косинусная близость к обращению, 0..1


@dataclass(frozen=True, slots=True)
class Classification:
    type: RequestType
    confidence: float  # 0..1
    reason: str  # одно предложение: почему именно этот тип


@dataclass(frozen=True, slots=True)
class Appointment:
    """Результат бронирования приёма (только для типа 'запись')."""

    slot_iso: str  # "2026-09-24T10:30:00+05:00"
    office: str
    service: str
    ticket: str  # "A-042"


@dataclass(frozen=True, slots=True)
class Draft:
    ru: str
    kk: str
    cited: list[str] = field(default_factory=list)  # ["3.1", "3.4"]


@dataclass(frozen=True, slots=True)
class AgentResult:
    request_id: str
    input_text: str
    classification: Classification
    clauses: list[Clause]  # ровно 3
    draft: Draft
    appointment: Appointment | None
    trace_path: str
    duration_ms: int


# --- Интерфейсы --------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Обёртка над OpenAI-совместимым API.

    Если передан json_schema, реализация ОБЯЗАНА вернуть строку с валидным
    JSON по этой схеме: либо structured output провайдера, либо ретрай
    с починкой. Вызывающая сторона доверяет этому и не парсит мусор.
    """

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_schema: dict[str, Any] | None = None,
    ) -> str: ...


@runtime_checkable
class Tracer(Protocol):
    """Трассировщик шагов. Формат файла — SPEC.md, раздел 3.2."""

    request_id: str
    """Тот же идентификатор попадает в AgentResult — оркестратор берёт его
    отсюда, а не генерирует второй раз."""

    def step(
        self,
        action: str,
        *,
        inp: dict[str, Any],
        out: dict[str, Any],
        duration_ms: int,
        meta: dict[str, Any] | None = None,
    ) -> None: ...

    def tool_call(
        self,
        name: str,
        *,
        args: dict[str, Any],
        result: dict[str, Any],
        duration_ms: int,
    ) -> None: ...

    def error(self, action: str, *, inp: dict[str, Any], message: str) -> None: ...

    def save(self) -> str:
        """Записывает лог и возвращает путь к файлу."""
        ...


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[Clause]: ...


# --- Ошибки ------------------------------------------------------------------


class AgentError(Exception):
    """Базовая ошибка пайплайна."""


class ReglamentError(AgentError):
    """Регламент не найден или нарушен формат заголовков (SPEC 3.3)."""


class LLMError(AgentError):
    """Провайдер недоступен или вернул невалидный ответ."""
