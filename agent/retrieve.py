"""Поиск релевантных пунктов регламента (SPEC.md, раздел 3.4).

Эмбеддер подключаемый. По умолчанию — BGE-m3 через sentence-transformers,
но если модель недоступна (нет сети, нет GPU, чужая машина на защите),
используется лексический эмбеддер на numpy. Это не «запасной костыль»:
регламент на две страницы с разведённой лексикой ищется лексически
достаточно уверенно, а прототип остаётся запускаемым где угодно.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import replace
from typing import Protocol

import numpy as np

from agent.reglament import parse_reglament
from agent.types import Clause

log = logging.getLogger(__name__)

WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
STEM_LEN = 6  # грубое усечение вместо морфологии: «справки»/«справка» → «справк»
MIN_TOKEN_LEN = 3  # без этого служебные слова накручивают близость постороннему
# запросу, и он перестаёт помечаться как ненадёжный.
STOPWORDS = frozenset(
    """
    или что как при это так для его мне без бы же ли уже еще ещё был была были
    его её их там тут где когда кто чем чём том тем этот эта эти ваш мой наш
    """.split()
)


class Embedder(Protocol):
    min_score: float

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


def _tokens(text: str) -> list[str]:
    words = (w.lower() for w in WORD.findall(text))
    return [
        w[:STEM_LEN]
        for w in words
        if len(w) >= MIN_TOKEN_LEN and w not in STOPWORDS
    ]


class LexicalEmbedder:
    """TF-IDF по усечённым словам. Словарь строится по корпусу регламента."""

    min_score = 0.12

    def __init__(self, corpus: list[str]) -> None:
        docs = [_tokens(text) for text in corpus]
        vocab = sorted({token for doc in docs for token in doc})
        self._vocab = {token: i for i, token in enumerate(vocab)}
        df = np.zeros(len(vocab))
        for doc in docs:
            for token in set(doc):
                df[self._vocab[token]] += 1
        self._idf = np.log((1 + len(docs)) / (1 + df)) + 1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self._vocab)))
        for row, text in enumerate(texts):
            for token in _tokens(text):
                index = self._vocab.get(token)
                if index is not None:
                    matrix[row, index] += 1.0
        return _normalize(matrix * self._idf)


class SentenceTransformerEmbedder:
    """BGE-m3. Модель грузится лениво — импорт не должен ничего тянуть."""

    min_score = 0.35

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.getenv("EMBED_MODEL", "BAAI/bge-m3")
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(self._model.encode(texts), dtype=float))


def default_embedder(corpus: list[str]) -> Embedder:
    """BGE-m3, если доступна; иначе лексический эмбеддер."""
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:  # нет пакета, нет модели, нет сети
        log.warning("BGE-m3 недоступна (%s), используется лексический поиск", exc)
        return LexicalEmbedder(corpus)


class ReglamentRetriever:
    """Поиск по регламенту. Индекс в памяти: пунктов десятки, не миллионы."""

    def __init__(
        self,
        clauses: list[Clause] | None = None,
        embedder: Embedder | None = None,
        min_score: float | None = None,
    ) -> None:
        self.clauses = clauses if clauses is not None else parse_reglament()
        if not self.clauses:
            raise ValueError("нечего индексировать: пустой регламент")
        corpus = [f"{c.title}. {c.text}" for c in self.clauses]
        self.embedder = embedder or default_embedder(corpus)
        env = os.getenv("RETRIEVE_MIN_SCORE")
        self.min_score = (
            min_score
            if min_score is not None
            else float(env)
            if env
            else getattr(self.embedder, "min_score", 0.35)
        )
        self._matrix = self.embedder.encode(corpus)

    def search(self, query: str, k: int = 3) -> list[Clause]:
        """Возвращает k пунктов по убыванию близости, со заполненным score."""
        if not query.strip():
            raise ValueError("пустой запрос")
        vector = self.embedder.encode([query])[0]
        scores = self._matrix @ vector
        k = max(1, min(k, len(self.clauses)))
        top = np.argsort(-scores)[:k]
        return [
            replace(self.clauses[i], score=round(float(scores[i]), 4)) for i in top
        ]

    def is_low_confidence(self, found: list[Clause]) -> bool:
        """True, если лучший пункт слабый: в черновик пойдёт оговорка.

        Номера пунктов при этом не выдумываются — возвращается то, что
        нашлось, но ответ честно помечается как требующий уточнения.
        """
        return not found or found[0].score < self.min_score
