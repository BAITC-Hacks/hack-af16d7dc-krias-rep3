"""Общие фикстуры для офлайн-тестов обеих частей проекта."""

import pytest


@pytest.fixture(autouse=True)
def isolated_retriever_cache():
    """Кэш процесса не должен переносить зависимости между тестами."""
    import sys

    def clear():
        pipeline = sys.modules.get("agent.pipeline")
        cached = getattr(pipeline, "_cached_retriever", None)
        if cached is not None:
            cached.cache_clear()

    clear()
    yield
    clear()


@pytest.fixture
def fake_llm():
    """Фабрика: llm = fake_llm(['первый ответ', 'второй ответ'])."""
    from tests.fakes import FakeLLM

    return FakeLLM


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Любая случайная попытка теста выйти в сеть немедленно падает."""
    import socket

    def deny(*args, **kwargs):
        raise AssertionError("Сетевые подключения в тестах запрещены")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)


@pytest.fixture
def result_factory():
    from agent.types import AgentResult, Classification, Clause, Draft

    def make(text="Обращение", kind="справка"):
        return AgentResult(
            request_id="abc123", input_text=text,
            classification=Classification(kind, .95, "Основание"),
            clauses=[Clause(f"2.{i}", "Название", "Пункт", .8) for i in range(1, 4)],
            draft=Draft("Русский ответ " * 30, "Қазақша жауап " * 30, ["2.1"]),
            appointment=None, trace_path="logs/agent_log.json", duration_ms=10,
        )
    return make
