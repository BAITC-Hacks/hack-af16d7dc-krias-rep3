"""Черновик ответа на русском и казахском (SPEC.md, раздел 3.5).

Казахская версия — перевод русской, а не независимая генерация:
иначе номера пунктов, сроки и суммы в двух версиях расходятся, и это
первое, что заметит проверяющий.

Ссылки на пункты проверяются программно: выдуманное основание в ответе
гражданину — самая дорогая ошибка этого проекта. Правило мягче, чем
«только найденные три»: сослаться можно на любой существующий пункт
регламента, но хотя бы одна ссылка обязана быть из найденных. Иначе
черновик отвергался за ссылку на настоящий смежный пункт — проверка
ловила бы не галлюцинацию, а полезное уточнение.
"""

from __future__ import annotations

import re

from agent.types import Appointment, Classification, Clause, Draft, LLMClient, LLMError

CITATION = re.compile(r"п\.\s*(\d+\.\d+)")

SYSTEM_RU = """Ты готовишь черновик официального ответа заявителю от
управления обслуживания населения. Требования к тексту:

- обращение к заявителю, деловой тон, без канцелярских штампов ради объёма;
- ссылка минимум на один пункт регламента в виде «п. 2.3»;
- ссылаться можно ТОЛЬКО на пункты, приведённые ниже, дословно по номеру;
- указать срок, если он есть в приведённых пунктах;
- подпись: «Управление обслуживания населения города Астаны»;
- 150–250 слов, без markdown-разметки.

Не придумывай номера пунктов, сроки, суммы и фамилии."""

SYSTEM_KK = """Сен ресми жауаптың қазақ тіліндегі нұсқасын дайындайсың.
Орыс тіліндегі мәтінді аудар. Талаптар:

- пункт нөмірлерін («п. 2.3» түрінде), мерзімдерді және сандарды
  дәл сол қалпында сақта;
- мағынасын өзгертпе, ештеңе қоспа және алып тастама;
- ресми іскерлік стиль, markdown белгілерін қолданба."""


def _clauses_block(clauses: list[Clause]) -> str:
    return "\n\n".join(f"п. {c.id} {c.title}\n{c.text}" for c in clauses)


def _cited(text: str) -> set[str]:
    return set(CITATION.findall(text))


def draft_ru(
    text: str,
    classification: Classification,
    clauses: list[Clause],
    llm: LLMClient,
    *,
    low_confidence: bool = False,
    appointment: Appointment | None = None,
    known_ids: set[str] | None = None,
) -> str:
    """Генерирует русскую версию ответа со ссылками на найденные пункты.

    Args:
        known_ids: номера всех пунктов регламента. Ссылка на существующий
            пункт вне найденных допустима, на несуществующий — нет.
    """
    if not clauses:
        raise ValueError("нет пунктов регламента для обоснования ответа")

    allowed = {c.id for c in clauses}
    permitted = allowed | (known_ids or set())
    parts = [
        f"Тип обращения: {classification.type}.",
        f"Текст обращения:\n{text.strip()}",
        f"Пункты регламента, на которые можно ссылаться:\n\n{_clauses_block(clauses)}",
    ]
    if appointment is not None:
        parts.append(
            "Запись оформлена: отделение "
            f"{appointment.office}, услуга {appointment.service}, "
            f"время {appointment.slot_iso}, талон {appointment.ticket}. "
            "Укажи эти данные в ответе."
        )
    if low_confidence:
        parts.append(
            "Найденные пункты слабо связаны с обращением. Добавь абзац о том, "
            "что для точного ответа требуется уточнение предмета обращения, "
            "и не утверждай лишнего."
        )
    user = "\n\n".join(parts)

    for attempt in range(2):
        result = llm.complete(system=SYSTEM_RU, user=user, temperature=0.3)
        cited = _cited(result)
        invented = cited - permitted
        if cited & allowed and not invented:
            return result.strip()
        user += (
            "\n\nПредыдущая попытка отклонена: "
            + (
                f"использованы несуществующие пункты {sorted(invented)}."
                if invented
                else "в тексте нет ни одной ссылки на найденные пункты."
            )
            + f" Обязательно сошлись хотя бы на один из {sorted(allowed)}. Исправь."
        )

    raise LLMError(
        "черновик дважды сослался на несуществующие пункты либо не сослался "
        "ни на один из найденных; ответ без основания отдавать нельзя"
    )


def translate_kk(ru: str, llm: LLMClient) -> str:
    """Переводит русскую версию на казахский, сохраняя номера пунктов."""
    if not ru.strip():
        raise ValueError("нечего переводить")

    expected = _cited(ru)
    user = ru
    for attempt in range(2):
        result = llm.complete(system=SYSTEM_KK, user=user, temperature=0.2)
        if _cited(result) == expected:
            return result.strip()
        user = (
            f"{ru}\n\n(Предыдущий перевод потерял ссылки на пункты "
            f"{sorted(expected)} — сохрани их дословно.)"
        )

    raise LLMError(
        f"перевод не сохранил ссылки на пункты {sorted(expected)}: "
        "расхождение версий ответа недопустимо"
    )


def build_draft(ru: str, kk: str) -> Draft:
    """Собирает Draft, фиксируя фактически использованные пункты."""
    return Draft(ru=ru, kk=kk, cited=sorted(_cited(ru)))
