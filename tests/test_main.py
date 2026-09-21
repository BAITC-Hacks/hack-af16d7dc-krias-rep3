from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import pytest

import main
from agent.types import LLMError


@pytest.mark.parametrize("as_json", [False, True])
def test_single_output(monkeypatch, capsys, result_factory, as_json):
    monkeypatch.setattr(main, "run", result_factory)
    assert main.main(["--text", "Өтініш"] + (["--json"] if as_json else [])) == 0
    captured = capsys.readouterr()
    if as_json:
        assert json.loads(captured.out) == asdict(result_factory("Өтініш"))
    else:
        assert all(text in captured.out for text in ("справка", "п. 2.1", "Русский", "Қазақша", "logs/agent_log.json"))
    assert not captured.err


def test_batch_continues_after_error(monkeypatch, tmp_path, capsys, result_factory):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([{"id": str(i), "text": t} for i, t in enumerate(["ok", "bad", "ok"])]))
    def run(text):
        if text == "bad":
            raise LLMError("модель недоступна")
        return result_factory(text)
    monkeypatch.setattr(main, "run", run)
    assert main.main(["--batch", str(path), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["summary"] == {"total": 3, "types": {"справка": 2}, "failed": 1, "average_duration_ms": 10}
    assert len(data["results"]) == 3
    assert data["results"][1]["error"] == "модель недоступна"


def test_samples_balance():
    samples = json.loads((Path(__file__).resolve().parents[1] / "data/samples.json").read_text())
    assert Counter(s["expected_type"] for s in samples) == {"справка": 4, "жалоба": 4, "запись": 4}
    assert len({s["id"] for s in samples}) == 12
    assert sum("пограничный" in s["note"] for s in samples) == 2


def test_batch_integrates_real_pipeline_offline(monkeypatch, tmp_path, capsys, fake_llm):
    """Все 12 примеров через CLI, поиск, календарь и трассировку; только LLM подменён."""
    from agent import llm, pipeline, tools
    from agent.reglament import parse_reglament
    from agent.retrieve import LexicalEmbedder, ReglamentRetriever, build_query

    sample_path = Path(__file__).resolve().parents[1] / "data/samples.json"
    samples = json.loads(sample_path.read_text())
    clauses = parse_reglament()
    retriever = ReglamentRetriever(clauses=clauses, embedder=LexicalEmbedder([f"{c.title}. {c.text}" for c in clauses]))
    clients = []
    for sample in samples:
        clause = retriever.search(build_query(sample["expected_type"], sample["text"]))[0].id
        clients.append(fake_llm([
            json.dumps({"type": sample["expected_type"], "confidence": .9, "reason": "Тест"}),
            f"Уважаемый заявитель! Ответ согласно п. {clause}.",
            f"Құрметті өтініш беруші! п. {clause} бойынша жауап.",
        ]))
    pending = iter(clients)
    monkeypatch.setattr(llm, "get_client", lambda: next(pending))
    monkeypatch.setattr(pipeline, "ReglamentRetriever", lambda: retriever)
    monkeypatch.setattr(tools, "_BOOKED", set())
    monkeypatch.setattr(tools, "_TICKETS", set())
    monkeypatch.chdir(tmp_path)
    assert main.main(["--batch", str(sample_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["failed"] == 0
    assert data["summary"]["types"] == {"справка": 4, "жалоба": 4, "запись": 4}
    for item in data["results"]:
        result = item["result"]
        assert len(result["clauses"]) == 3
        assert (result["appointment"] is not None) == (result["classification"]["type"] == "запись")
    trace = json.loads((tmp_path / "logs/agent_log.json").read_text())
    assert trace["request_id"] == data["results"][-1]["result"]["request_id"]
    assert trace["result"]["draft_kk_chars"] == len(data["results"][-1]["result"]["draft"]["kk"])
