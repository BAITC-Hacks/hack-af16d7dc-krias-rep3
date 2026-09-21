"""Синхронный клиент /chat/completions с проверкой JSON Schema."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from time import sleep
from typing import Any

import httpx
from dotenv import load_dotenv
from jsonschema import SchemaError, ValidationError
from jsonschema.validators import validator_for

from agent.types import LLMClient, LLMError

log = logging.getLogger(__name__)


class _HTTPError(LLMError):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"LLM-провайдер вернул HTTP {status}")


class _ResponseError(LLMError):
    """Успешный HTTP-ответ, из которого нельзя извлечь текст."""


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        if not base_url.strip() or not api_key.strip() or not model.strip():
            raise LLMError("Задайте LLM_BASE_URL, LLM_API_KEY и LLM_MODEL")
        if not math.isfinite(timeout) or timeout <= 0:
            raise LLMError("LLM_TIMEOUT должен быть положительным числом")
        try:
            url = httpx.URL(base_url)
        except httpx.InvalidURL:
            raise LLMError("Некорректный LLM_BASE_URL") from None
        if url.scheme not in ("http", "https") or not url.host:
            raise LLMError("LLM_BASE_URL должен быть HTTP(S)-адресом API")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._api_key = api_key

    def _request(self, payload: dict[str, Any]) -> str:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for attempt in range(3):
                    response = client.post(
                        f"{self.base_url}/chat/completions", json=payload,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                    )
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt < 2:
                            log.warning("LLM HTTP %s, повтор %s/3", response.status_code, attempt + 2)
                            sleep(2 ** attempt)
                            continue
                    if response.is_error:
                        raise _HTTPError(response.status_code)
                    try:
                        content = response.json()["choices"][0]["message"]["content"]
                    except (ValueError, KeyError, IndexError, TypeError):
                        raise _ResponseError("LLM-провайдер вернул некорректный ответ") from None
                    if not isinstance(content, str) or not content.strip():
                        raise _ResponseError("LLM-провайдер вернул пустой текст")
                    return content
        except httpx.TimeoutException:
            raise LLMError("Истекло время ожидания LLM-провайдера") from None
        except httpx.RequestError:
            raise LLMError("Не удалось соединиться с LLM-провайдером") from None
        raise LLMError("LLM-провайдер не вернул ответ")

    def complete(self, *, system: str, user: str, temperature: float = 0.2,
                 json_schema: dict | None = None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_schema is None:
            return self._request(payload)
        # classify.py передаёт именованную обёртку; также принимаем чистую схему.
        name = "response"
        if isinstance(json_schema.get("schema"), dict) and "name" in json_schema:
            name = json_schema["name"]
            json_schema = json_schema["schema"]
        try:
            validator_type = validator_for(json_schema)
            validator_type.check_schema(json_schema)
            validator = validator_type(json_schema)
        except SchemaError:
            raise LLMError("Передана некорректная JSON Schema") from None
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": json_schema},
        }
        for attempt in range(2):
            try:
                content = self._request(payload)
            except _HTTPError as exc:
                if attempt or exc.status not in (400, 422):
                    raise
            except _ResponseError:
                if attempt:
                    raise
            else:
                try:
                    value = json.loads(content, parse_constant=_reject_constant)
                    validator.validate(value)
                    return content
                except (ValueError, ValidationError):
                    if attempt:
                        raise LLMError("LLM дважды вернул JSON, не соответствующий схеме") from None
            log.info("Повтор генерации JSON со схемой в промпте")
            payload.pop("response_format", None)
            payload["temperature"] = 0
            payload["messages"] = [
                {"role": "system", "content": system + "\nВерни только валидный JSON, без Markdown, строго по схеме:\n" + json.dumps(json_schema, ensure_ascii=False)},
                {"role": "user", "content": user},
            ]
        raise LLMError("LLM не смог сформировать JSON")


def _reject_constant(value: str) -> None:
    raise ValueError(f"Недопустимая константа JSON: {value}")


def get_client() -> LLMClient:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    try:
        timeout = float(os.getenv("LLM_TIMEOUT", "60"))
    except ValueError:
        raise LLMError("LLM_TIMEOUT должен быть положительным числом") from None
    return OpenAICompatibleClient(
        base_url=os.getenv("LLM_BASE_URL", ""), api_key=os.getenv("LLM_API_KEY", ""),
        model=os.getenv("LLM_MODEL", ""), timeout=timeout,
    )
