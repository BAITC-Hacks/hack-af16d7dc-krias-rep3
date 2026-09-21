"""Тесты парсера регламента (SPEC 3.3)."""

from __future__ import annotations

import re

import pytest

from agent.reglament import clause_index, parse_reglament
from agent.types import ReglamentError

ID_PATTERN = re.compile(r"^\d+\.\d+$")


def test_регламент_разбирается_и_содержит_достаточно_пунктов():
    clauses = parse_reglament()
    assert len(clauses) >= 12, "задание требует содержательный регламент"


def test_у_каждого_пункта_валидный_номер_название_и_текст():
    for clause in parse_reglament():
        assert ID_PATTERN.match(clause.id), f"плохой номер: {clause.id!r}"
        assert clause.title.strip()
        assert len(clause.text) > 40, f"пункт {clause.id} слишком короткий"


def test_номера_пунктов_уникальны():
    clauses = parse_reglament()
    ids = [c.id for c in clauses]
    assert len(ids) == len(set(ids))
    assert len(clause_index(clauses)) == len(ids)


def test_покрыты_все_три_типа_обращений():
    """Иначе ретриверу нечего будет находить для части обращений."""
    text = " ".join(c.title.lower() + " " + c.text.lower() for c in parse_reglament())
    assert "справк" in text
    assert "жалоб" in text
    assert "талон" in text


def test_текст_пункта_склеивается_из_нескольких_строк(tmp_path):
    path = tmp_path / "r.md"
    path.write_text(
        "# Заголовок\n\n## 1.1 Название\n\nПервая строка\nвторая строка.\n",
        encoding="utf-8",
    )
    clauses = parse_reglament(path)
    assert len(clauses) == 1
    assert clauses[0].text == "Первая строка вторая строка."
    assert clauses[0].title == "Название"


def test_заголовок_раздела_не_становится_пунктом(tmp_path):
    path = tmp_path / "r.md"
    path.write_text(
        "## Общие положения\n\nтекст\n\n## 2.1 Пункт\n\nтело пункта\n",
        encoding="utf-8",
    )
    clauses = parse_reglament(path)
    assert [c.id for c in clauses] == ["2.1"]


def test_пустой_пункт_это_ошибка(tmp_path):
    path = tmp_path / "r.md"
    path.write_text("## 1.1 Название\n\n## 1.2 Другое\n\nтекст\n", encoding="utf-8")
    with pytest.raises(ReglamentError):
        parse_reglament(path)


def test_дубль_номера_это_ошибка(tmp_path):
    path = tmp_path / "r.md"
    path.write_text(
        "## 1.1 А\n\nтекст а\n\n## 1.1 Б\n\nтекст б\n", encoding="utf-8"
    )
    with pytest.raises(ReglamentError):
        parse_reglament(path)


def test_файл_без_пунктов_это_ошибка(tmp_path):
    path = tmp_path / "r.md"
    path.write_text("# Просто текст\n\nбез пунктов\n", encoding="utf-8")
    with pytest.raises(ReglamentError):
        parse_reglament(path)


def test_отсутствующий_файл_это_ошибка(tmp_path):
    with pytest.raises(ReglamentError):
        parse_reglament(tmp_path / "нет.md")
