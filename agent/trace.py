"""JSON-трассировка по SPEC 3.2; один экземпляр на обращение."""

from __future__ import annotations

import json
import os
from pathlib import Path
from secrets import token_hex
from tempfile import NamedTemporaryFile
from time import perf_counter
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from agent.types import Tracer


def _compact(value: Any) -> Any:
    """Снимок JSON-данных без изменения объектов вызывающей стороны."""
    if isinstance(value, str):
        return value if len(value) <= 200 else value[:199] + "…"
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value]
    return value


class JsonTracer:
    def __init__(
        self,
        input_text: str,
        model: str,
        log_path: str = "logs/agent_log.json",
    ) -> None:
        self._started = perf_counter()
        self.request_id = token_hex(3)
        self.log_path = log_path
        self._data: dict[str, Any] = {
            "request_id": self.request_id,
            "created_at": datetime.now(ZoneInfo("Asia/Almaty")).isoformat(),
            "input": _compact(input_text),
            "model": model,
            "steps": [],
            "tool_calls": [],
            "result": {},
            "duration_ms": 0,
        }

    def step(
        self,
        action: str,
        *,
        inp: dict[str, Any],
        out: dict[str, Any],
        duration_ms: int,
        meta: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "step": len(self._data["steps"]) + 1,
            "action": action,
            "input": _compact(inp),
            "output": _compact(out),
            "duration_ms": duration_ms,
        }
        if meta is not None:
            entry["meta"] = _compact(meta)
        self._data["steps"].append(entry)

        # Сводка из форматов шагов SPEC 3.2 и README, без нового метода
        # в замороженном протоколе Tracer.
        result = self._data["result"]
        if action == "classify" and "type" in out:
            result["type"] = _compact(out["type"])
        elif action == "retrieve" and "clauses" in out:
            result["clauses"] = _compact(out["clauses"])
        elif action in ("draft_ru", "translate_kk"):
            language = "ru" if action == "draft_ru" else "kk"
            if "chars" in out:
                result[f"draft_{language}_chars"] = out["chars"]
            elif isinstance(out.get("text"), str):
                result[f"draft_{language}_chars"] = len(out["text"])

    def tool_call(
        self,
        name: str,
        *,
        args: dict[str, Any],
        result: dict[str, Any],
        duration_ms: int,
    ) -> None:
        self._data["tool_calls"].append({
            "name": name,
            "args": _compact(args),
            "result": _compact(result),
            "duration_ms": duration_ms,
        })

    def error(self, action: str, *, inp: dict[str, Any], message: str) -> None:
        self._data["steps"].append({
            "step": len(self._data["steps"]) + 1,
            "action": action,
            "input": _compact(inp),
            "error": _compact(message),
        })
        # Не полагаемся на то, что вызывающий код дойдёт до save().
        self.save()

    def save(self) -> str:
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._data["duration_ms"] = int((perf_counter() - self._started) * 1000)
        payload = json.dumps(self._data, ensure_ascii=False, indent=2)
        # Замена целого файла сохраняет предыдущий лог при сбое записи.
        temporary: str | None = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = stream.name
                stream.write(payload + "\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        return str(path)


def new_tracer(input_text: str, model: str) -> Tracer:
    return JsonTracer(input_text, model)
