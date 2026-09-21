import json

import httpx
import pytest

from agent import llm
from agent.types import LLMError

SCHEMA = {"type": "object", "properties": {"value": {"type": "integer"}},
          "required": ["value"], "additionalProperties": False}


@pytest.fixture
def provider(monkeypatch):
    original_client = httpx.Client

    def make(responses):
        pending = iter(responses)
        calls, pauses = [], []

        def handle(request):
            calls.append(json.loads(request.content))
            item = next(pending)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, int):
                return httpx.Response(item, json={"error": "provider error"})
            return httpx.Response(200, json={"choices": [{"message": {"content": item}}]})

        transport = httpx.MockTransport(handle)
        monkeypatch.setattr(llm.httpx, "Client", lambda **kw: original_client(transport=transport, **kw))
        monkeypatch.setattr(llm, "sleep", pauses.append)
        return llm.OpenAICompatibleClient("https://llm.invalid/v1", "test-key", "test-model"), calls, pauses
    return make


@pytest.mark.parametrize("responses,expected_calls", [
    (['{"value":2}'], 1), ([400, '{"value":2}'], 2),
    ([422, '{"value":2}'], 2), (["не JSON", '{"value":2}'], 2),
    ([None, '{"value":2}'], 2),
    (['{"value":"wrong"}', '{"value":2}'], 2),
])
def test_schema_and_one_repair(provider, responses, expected_calls):
    client, calls, pauses = provider(responses)
    assert json.loads(client.complete(system="system", user="user", json_schema=SCHEMA)) == {"value": 2}
    assert len(calls) == expected_calls and pauses == []
    assert calls[0]["response_format"]["json_schema"]["schema"] == SCHEMA
    if expected_calls == 2:
        assert "response_format" not in calls[1]
        assert calls[1]["temperature"] == 0
        assert '"required": ["value"]' in calls[1]["messages"][0]["content"]


@pytest.mark.parametrize("responses,schema,count,pauses_expected", [
    ([429, 503, "готово"], None, 3, [1, 2]),
    ([503, 503, 503], None, 3, [1, 2]),
    ([401], SCHEMA, 1, []),
    (["bad", "bad again"], SCHEMA, 2, []),
    ([httpx.ReadTimeout("таймаут")], None, 1, []),
    ([httpx.ConnectError("нет сети")], None, 1, []),
])
def test_retry_budgets_and_readable_errors(provider, responses, schema, count, pauses_expected):
    client, calls, pauses = provider(responses)
    if responses[-1] == "готово":
        assert client.complete(system="s", user="u") == "готово"
    else:
        with pytest.raises(LLMError):
            client.complete(system="s", user="u", json_schema=schema)
    assert len(calls) == count
    assert pauses == pauses_expected


def test_classification_schema_wrapper_is_validated(provider):
    from agent.classify import SCHEMA as classification_schema, classify
    valid = '{"type":"справка","confidence":0.9,"reason":"Запрос документа"}'
    client, calls, _ = provider(['{"type":"справка"}', valid])
    assert classify("Выдайте справку", client).confidence == .9
    assert len(calls) == 2
    assert calls[0]["response_format"]["json_schema"]["schema"] == classification_schema["schema"]


@pytest.mark.parametrize("timeout", ["0", "nan", "abc"])
def test_invalid_config(monkeypatch, timeout):
    monkeypatch.setattr(llm, "load_dotenv", lambda *a: None)
    for name, value in {"LLM_BASE_URL": "https://llm.invalid/v1", "LLM_API_KEY": "test",
                        "LLM_MODEL": "model", "LLM_TIMEOUT": timeout}.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(LLMError, match="LLM_TIMEOUT"):
        llm.get_client()
