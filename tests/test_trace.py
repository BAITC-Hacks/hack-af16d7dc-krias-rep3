"""Приёмочные сценарии SPEC 3.2 и TASKS_CODEX 2; сеть не нужна."""

import json
import re
from datetime import datetime, timedelta

import pytest

from agent import trace
from agent.types import Tracer


@pytest.fixture
def recording(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(trace, "perf_counter", lambda: clock[0])
    path = tmp_path / "nested" / "agent_log.json"
    tracer = trace.JsonTracer("Өтініш: справка", "test-model", str(path))
    return tracer, path, clock


def test_complete_trace_contract(recording):
    tracer, path, clock = recording
    assert isinstance(tracer, Tracer)
    outputs = [
        ("classify", {"type": "запись", "confidence": 0.94}),
        ("retrieve", {"clauses": ["3.1", "3.4", "7.2"], "scores": [.8, .7, .6]}),
        ("draft_ru", {"text": "я" * 812}),
        ("translate_kk", {"chars": 794}),
    ]
    for action, out in outputs:
        tracer.step(action, inp={"text": "Өтініш"}, out=out,
                    duration_ms=12, meta={"source": "test"})
    tracer.tool_call("book_appointment", args={"service": "справка"},
                     result={"ticket": "A-042"}, duration_ms=5)
    clock[0] = 14.31
    assert tracer.save() == str(path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert set(data) == {"request_id", "created_at", "input", "model", "steps",
                         "tool_calls", "result", "duration_ms"}
    assert re.fullmatch(r"[0-9a-f]{6}", data["request_id"])
    assert data["request_id"] == tracer.request_id
    assert datetime.fromisoformat(data["created_at"]).utcoffset() == timedelta(hours=5)
    assert data["input"] == "Өтініш: справка"
    assert data["model"] == "test-model"
    assert [s["step"] for s in data["steps"]] == [1, 2, 3, 4]
    assert [s["action"] for s in data["steps"]] == [a for a, _ in outputs]
    assert all(s["duration_ms"] == 12 for s in data["steps"])
    assert data["steps"][0]["meta"] == {"source": "test"}
    assert data["result"] == {"type": "запись", "clauses": ["3.1", "3.4", "7.2"],
                              "draft_ru_chars": 812, "draft_kk_chars": 794}
    assert data["tool_calls"] == [{"name": "book_appointment", "args": {"service": "справка"},
                                    "result": {"ticket": "A-042"}, "duration_ms": 5}]
    assert data["duration_ms"] == 4310
    assert '  "input": "Өтініш: справка"' in raw


@pytest.mark.parametrize("length", [199, 200, 201, 1000])
def test_recursive_truncation_and_snapshot(recording, length):
    tracer, path, _ = recording
    text = "ә" * length
    expected = text if length <= 200 else text[:199] + "…"
    payload = {"nested": [{"text": text}], "number": 7, "flag": True, "empty": None}
    tracer.step("custom", inp=payload, out=payload, duration_ms=0)
    tracer.tool_call("custom", args=payload, result=payload, duration_ms=0)
    assert payload["nested"][0]["text"] == text
    payload["nested"][0]["text"] = "изменено после записи"
    tracer.save()
    data = json.loads(path.read_text())
    for snapshot in (data["steps"][0]["input"], data["steps"][0]["output"],
                     data["tool_calls"][0]["args"], data["tool_calls"][0]["result"]):
        assert snapshot == {"nested": [{"text": expected}], "number": 7,
                            "flag": True, "empty": None}


def test_error_is_persisted_without_explicit_save(recording):
    tracer, path, clock = recording
    tracer.step("classify", inp={}, out={"type": "жалоба"}, duration_ms=1)
    clock[0] = 11.0
    tracer.error("retrieve", inp={"text": "ә" * 250}, message="Ошибка поиска")
    data = json.loads(path.read_text())
    assert [step["step"] for step in data["steps"]] == [1, 2]
    assert data["steps"][1] == {"step": 2, "action": "retrieve",
                                "input": {"text": "ә" * 199 + "…"},
                                "error": "Ошибка поиска"}
    assert data["duration_ms"] == 1000
    tracer.save()
    assert json.loads(path.read_text()) == data


def test_factory_defaults_and_failed_write_preserve_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracer = trace.new_tracer("ә" * 300, "test-model")
    assert isinstance(tracer, Tracer)
    assert tracer.save() == "logs/agent_log.json"
    path = tmp_path / "logs" / "agent_log.json"
    previous = path.read_bytes()
    assert len(json.loads(previous)["input"]) == 200

    def fail_replace(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(trace.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        tracer.save()
    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]
