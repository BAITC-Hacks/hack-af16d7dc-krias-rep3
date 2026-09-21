"""CLI для одного обращения и пакетного прогона примеров."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

from agent.types import AgentResult


def run(text: str) -> AgentResult:
    from agent.pipeline import run as run_pipeline
    return run_pipeline(text)


def _display(result: AgentResult) -> None:
    print(f"Тип: {result.classification.type} ({result.classification.confidence:.0%})")
    for clause in result.clauses:
        print(f"  п. {clause.id} — {clause.title}")
    print(f"\nРусский:\n{result.draft.ru}\n\nҚазақша:\n{result.draft.kk}")
    if result.appointment:
        appointment = result.appointment
        print(f"\nПриём: {appointment.slot_iso}, {appointment.office}, талон {appointment.ticket}")
    print(f"\nЛог: {result.trace_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--batch", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        if args.batch:
            samples = json.loads(args.batch.read_text(encoding="utf-8"))
            if not isinstance(samples, list) or not samples:
                raise ValueError("Пакет должен быть непустым JSON-массивом обращений")
            if any(not isinstance(s, dict) or not isinstance(s.get("text"), str) for s in samples):
                raise ValueError("Каждое обращение должно содержать строку text")
        else:
            samples = [{"text": args.text}]
    except (OSError, ValueError) as exc:
        print(f"Ошибка входных данных: {exc}", file=sys.stderr)
        return 1
    counts = Counter()
    elapsed = []
    outcomes = []
    failures = 0
    for sample in samples:
        try:
            if not sample["text"].strip():
                raise ValueError("Текст обращения не может быть пустым")
            result = run(sample["text"])
            counts[result.classification.type] += 1
            elapsed.append(result.duration_ms)
            outcomes.append({"id": sample.get("id"), "result": asdict(result)})
            if not args.as_json:
                _display(result)
        except Exception as exc:
            failures += 1
            outcomes.append({"id": sample.get("id"), "error": str(exc)})
            if not args.as_json:
                print(f"Ошибка {sample.get('id', '')}: {exc}", file=sys.stderr)
    summary = {"total": len(samples), "types": dict(counts), "failed": failures,
               "average_duration_ms": round(sum(elapsed) / len(elapsed), 2) if elapsed else 0}
    if args.as_json:
        output = {"results": outcomes, "summary": summary} if args.batch else outcomes[0].get("result", outcomes[0])
        print(json.dumps(output, ensure_ascii=False, indent=2))
    elif args.batch:
        print(f"\nИтого: {len(samples)}, ошибки: {failures}; типы: {dict(counts)}; среднее: {summary['average_duration_ms']} мс")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
