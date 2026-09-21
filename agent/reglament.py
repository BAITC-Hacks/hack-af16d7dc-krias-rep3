"""Разбор регламента в список пунктов.

Формат заголовка строго `## N.N Название` (SPEC.md, раздел 3.3).
Парсер намеренно строгий: молча проглоченный пункт означает, что агент
сошлётся на несуществующее основание, а это худшая из возможных ошибок
в этом проекте.
"""

from __future__ import annotations

import re
from pathlib import Path

from agent.types import Clause, ReglamentError

CLAUSE_HEADING = re.compile(r"^##\s+(\d+\.\d+)\s+(\S.*?)\s*$")
SECTION_HEADING = re.compile(r"^#{1,2}\s+")

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "reglament.md"


def parse_reglament(path: str | Path | None = None) -> list[Clause]:
    """Читает markdown-регламент и возвращает пункты в порядке следования.

    Raises:
        ReglamentError: файла нет, пунктов нет, номера дублируются
            или у пункта пустой текст.
    """
    src = Path(path) if path is not None else DEFAULT_PATH
    try:
        raw = src.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ReglamentError(f"регламент не найден: {src}") from exc

    clauses: list[Clause] = []
    current_id: str | None = None
    current_title = ""
    buffer: list[str] = []

    def flush() -> None:
        if current_id is None:
            return
        text = " ".join(" ".join(buffer).split())
        if not text:
            raise ReglamentError(f"пункт {current_id} без текста")
        clauses.append(Clause(id=current_id, title=current_title, text=text))

    for line in raw.splitlines():
        match = CLAUSE_HEADING.match(line)
        if match:
            flush()
            current_id, current_title = match.group(1), match.group(2)
            buffer = []
            continue
        if SECTION_HEADING.match(line):
            # Заголовок документа или раздела: закрывает текущий пункт.
            flush()
            current_id, current_title, buffer = None, "", []
            continue
        if current_id is not None:
            buffer.append(line)

    flush()

    if not clauses:
        raise ReglamentError(
            f"в {src} нет пунктов формата '## N.N Название' (SPEC 3.3)"
        )

    seen: dict[str, int] = {}
    for index, clause in enumerate(clauses):
        if clause.id in seen:
            raise ReglamentError(f"номер пункта {clause.id} встречается дважды")
        seen[clause.id] = index

    return clauses


def clause_index(clauses: list[Clause]) -> dict[str, Clause]:
    """Пункты по номеру — для проверки ссылок в черновике ответа."""
    return {clause.id: clause for clause in clauses}
