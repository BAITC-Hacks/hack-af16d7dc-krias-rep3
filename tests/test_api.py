from dataclasses import asdict

from fastapi.testclient import TestClient
import pytest

import api
from agent.types import AgentError, LLMError


@pytest.fixture
def client():
    with TestClient(api.app) as client:
        yield client
    api.app.dependency_overrides.clear()


def test_health_and_full_result(client, monkeypatch, result_factory):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    api.app.dependency_overrides[api.get_runner] = lambda: result_factory
    assert client.get("/health").json() == {"status": "ok", "model": "test-model"}
    response = client.post("/appeal", json={"text": "Өтініш"})
    assert response.status_code == 200
    assert response.json() == asdict(result_factory("Өтініш"))
    assert len(response.json()["draft"]["kk"]) > 200


@pytest.mark.parametrize("error,status", [(AgentError("ошибка"), 422), (LLMError("провайдер"), 503)])
def test_pipeline_error_mapping(client, error, status):
    def fail(text):
        raise error
    api.app.dependency_overrides[api.get_runner] = lambda: fail
    response = client.post("/appeal", json={"text": "Обращение"})
    assert response.status_code == status
    assert response.json() == {"detail": str(error)}


@pytest.mark.parametrize("body", [{}, {"text": " "}, {"text": 123}])
def test_invalid_request(client, body, result_factory):
    api.app.dependency_overrides[api.get_runner] = lambda: result_factory
    assert client.post("/appeal", json=body).status_code == 422
