"""Тесты черновика ответа (SPEC 3.5)."""

from __future__ import annotations

import pytest

from agent.draft import build_draft, draft_ru, translate_kk
from agent.types import Appointment, Classification, Clause, LLMError
from tests.fakes import FakeLLM

CLAUSES = [
    Clause(id="2.1", title="Виды справок", text="справка о составе семьи", score=0.5),
    Clause(id="2.3", title="Срок изготовления", text="три рабочих дня", score=0.4),
    Clause(id="2.2", title="Документы", text="документ, удостоверяющий личность", score=0.3),
]
SPRAVKA = Classification(type="справка", confidence=0.9, reason="просит документ")


def test_черновик_со_ссылкой_принимается():
    llm = FakeLLM(["Уважаемый заявитель! Согласно п. 2.3 справка готовится три дня."])
    ru = draft_ru("Нужна справка", SPRAVKA, CLAUSES, llm)
    assert "п. 2.3" in ru


def test_черновик_без_ссылок_переспрашивается():
    llm = FakeLLM(
        ["Ответим в ближайшее время.", "Согласно п. 2.1 справка выдаётся на бланке."]
    )
    ru = draft_ru("Нужна справка", SPRAVKA, CLAUSES, llm)
    assert "п. 2.1" in ru
    assert len(llm.calls) == 2
    assert "нет ни одной ссылки" in llm.calls[1]["user"]


def test_выдуманный_пункт_переспрашивается():
    """Ссылка на несуществующее основание — главное, что нельзя пропустить."""
    llm = FakeLLM(["Согласно п. 9.9 отказываем.", "Согласно п. 2.2 нужен документ."])
    ru = draft_ru("Нужна справка", SPRAVKA, CLAUSES, llm)
    assert "п. 2.2" in ru
    assert "9.9" in llm.calls[1]["user"]


def test_две_неудачи_подряд_это_ошибка():
    llm = FakeLLM(["Согласно п. 9.9 отказ.", "Согласно п. 8.1 отказ."])
    with pytest.raises(LLMError):
        draft_ru("Нужна справка", SPRAVKA, CLAUSES, llm)


def test_данные_записи_попадают_в_промпт():
    appointment = Appointment(
        slot_iso="2026-09-24T10:30:00+05:00",
        office="Есильское отделение",
        service="запись на приём",
        ticket="A-042",
    )
    llm = FakeLLM(["Ваш талон A-042, приём согласно п. 2.1."])
    draft_ru(
        "Хочу на приём",
        Classification(type="запись", confidence=0.9, reason="просит приём"),
        CLAUSES,
        llm,
        appointment=appointment,
    )
    assert "A-042" in llm.calls[0]["user"]
    assert "Есильское отделение" in llm.calls[0]["user"]


def test_слабый_поиск_добавляет_требование_оговорки():
    llm = FakeLLM(["Согласно п. 2.1 требуется уточнение."])
    draft_ru("непонятное обращение", SPRAVKA, CLAUSES, llm, low_confidence=True)
    assert "уточнение" in llm.calls[0]["user"]


def test_без_пунктов_черновик_не_генерируется():
    with pytest.raises(ValueError):
        draft_ru("Нужна справка", SPRAVKA, [], FakeLLM(["текст"]))


def test_только_разрешённые_пункты_в_промпте():
    llm = FakeLLM(["Согласно п. 2.1 справка выдаётся."])
    draft_ru("Нужна справка", SPRAVKA, CLAUSES, llm)
    user = llm.calls[0]["user"]
    assert "п. 2.1" in user and "п. 2.3" in user and "п. 2.2" in user


def test_перевод_сохраняет_номера_пунктов():
    llm = FakeLLM(["п. 2.3 бойынша анықтама үш жұмыс күні ішінде дайындалады."])
    kk = translate_kk("Согласно п. 2.3 справка готовится три дня.", llm)
    assert "п. 2.3" in kk


def test_перевод_потерявший_пункты_переспрашивается():
    llm = FakeLLM(["Анықтама дайындалады.", "п. 2.3 бойынша анықтама дайындалады."])
    kk = translate_kk("Согласно п. 2.3 справка готовится.", llm)
    assert "п. 2.3" in kk
    assert len(llm.calls) == 2


def test_перевод_дважды_потерявший_пункты_это_ошибка():
    llm = FakeLLM(["Анықтама дайындалады.", "Анықтама дайын."])
    with pytest.raises(LLMError):
        translate_kk("Согласно п. 2.3 справка готовится.", llm)


def test_пустой_текст_не_переводится():
    with pytest.raises(ValueError):
        translate_kk("   ", FakeLLM(["текст"]))


def test_build_draft_фиксирует_использованные_пункты():
    draft = build_draft("Согласно п. 2.3 и п. 2.1 ответ.", "п. 2.3 және п. 2.1 жауап.")
    assert draft.cited == ["2.1", "2.3"]


def test_ссылка_на_существующий_пункт_вне_найденных_допустима():
    """п. 3.3 есть в регламенте, но не попал в топ-3 — это не галлюцинация."""
    llm = FakeLLM(["Согласно п. 2.3 срок три дня, перенос — по п. 3.3."])
    ru = draft_ru("Нужна справка", SPRAVKA, CLAUSES, llm, known_ids={"3.3"})
    assert "п. 3.3" in ru
    assert len(llm.calls) == 1, "повторный запрос не нужен"


def test_ссылка_только_вне_найденных_переспрашивается():
    """Ответ обязан опираться на то, что нашёл поиск."""
    llm = FakeLLM(["Согласно п. 3.3 перенос возможен.", "Согласно п. 2.3 три дня."])
    ru = draft_ru("Нужна справка", SPRAVKA, CLAUSES, llm, known_ids={"3.3"})
    assert "п. 2.3" in ru
    assert "нет ни одной ссылки на найденные" in llm.calls[1]["user"]
