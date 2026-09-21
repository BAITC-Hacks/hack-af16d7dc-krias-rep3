"""Тесты поиска по регламенту (SPEC 3.4)."""

from __future__ import annotations

import pytest

from agent.retrieve import LexicalEmbedder, ReglamentRetriever
from agent.types import Clause
from tests.fakes import FakeEmbedder

# Реальные обращения и раздел регламента, который обязан найтись.
CASES = [
    ("Прошу выдать справку о составе семьи", "2"),
    ("Нужна справка о месте проживания для школы", "2"),
    ("Сколько ждать справку о социальных выплатах?", "2"),
    ("Хочу записаться на приём в отделение", "3"),
    ("Можно перенести запись, у меня номер талона A-12", "3"),
    ("Не согласен с действиями работника, прошу провести проверку", "4"),
    ("Жалоба на нарушение при рассмотрении моего обращения", "4"),
]


@pytest.fixture
def retriever() -> ReglamentRetriever:
    """Лексический эмбеддер: тесты не должны тянуть модель из сети."""
    clauses = ReglamentRetriever(embedder=FakeEmbedder(["x"])).clauses
    corpus = [f"{c.title}. {c.text}" for c in clauses]
    return ReglamentRetriever(clauses=clauses, embedder=LexicalEmbedder(corpus))


def test_возвращается_ровно_три_пункта(retriever):
    found = retriever.search("Прошу выдать справку о составе семьи")
    assert len(found) == 3


def test_пункты_отсортированы_по_убыванию_близости(retriever):
    found = retriever.search("Жалоба на работника управления")
    scores = [c.score for c in found]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_нужный_раздел_регламента_попадает_в_топ_3(retriever):
    for text, section in CASES:
        found = retriever.search(text)
        sections = {c.id.split(".")[0] for c in found}
        assert section in sections, f"{text!r}: нашлись {[c.id for c in found]}"


def test_лучший_пункт_из_нужного_раздела(retriever):
    """Более строгая проверка: раздел угадан не случайно."""
    hits = sum(retriever.search(t)[0].id.startswith(s) for t, s in CASES)
    assert hits >= len(CASES) - 1, f"первым пунктом угадано только {hits} из {len(CASES)}"


def test_пустой_запрос_это_ошибка(retriever):
    with pytest.raises(ValueError):
        retriever.search("   ")


def test_k_ограничен_числом_пунктов():
    clauses = [
        Clause(id="1.1", title="Справка", text="порядок изготовления справки"),
        Clause(id="2.1", title="Жалоба", text="порядок рассмотрения жалобы"),
    ]
    r = ReglamentRetriever(clauses=clauses, embedder=FakeEmbedder(["справк", "жалоб"]))
    assert len(r.search("справка", k=5)) == 2
    assert len(r.search("справка", k=1)) == 1


def test_слабый_результат_помечается_как_ненадёжный():
    clauses = [
        Clause(id="1.1", title="Справка", text="порядок изготовления справки"),
        Clause(id="2.1", title="Жалоба", text="порядок рассмотрения жалобы"),
    ]
    r = ReglamentRetriever(
        clauses=clauses, embedder=FakeEmbedder(["справк", "жалоб"]), min_score=0.5
    )
    assert r.is_low_confidence(r.search("погода в Астане")) is True
    assert r.is_low_confidence(r.search("нужна справка")) is False


def test_порог_берётся_из_эмбеддера_если_не_задан():
    clauses = [Clause(id="1.1", title="А", text="текст")]
    r = ReglamentRetriever(clauses=clauses, embedder=FakeEmbedder(["текст"]))
    assert r.min_score == FakeEmbedder.min_score


def test_пустой_регламент_это_ошибка():
    with pytest.raises(ValueError):
        ReglamentRetriever(clauses=[], embedder=FakeEmbedder(["x"]))


def test_посторонний_запрос_помечается_как_ненадёжный(retriever):
    """Главная защита от выдуманных оснований в ответе гражданину."""
    found = retriever.search("Какая погода в Астане завтра")
    assert retriever.is_low_confidence(found), [
        (c.id, c.score) for c in found
    ]


def test_служебные_слова_не_дают_ложной_близости(retriever):
    found = retriever.search("это как то что при или для")
    assert retriever.is_low_confidence(found)
