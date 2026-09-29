#!/usr/bin/env python3
"""Run eval cases against ask() / agent pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb_agent_lab.answer import ask  # noqa: E402


def _tool_names(traces: list) -> set[str]:
    names = set()
    for t in traces or []:
        if t.get("type") == "call" and t.get("name"):
            names.add(t["name"])
        elif t.get("name"):
            names.add(t["name"])
    return names


def run_case(case: dict) -> dict:
    query = case["query"]
    result = ask(query)
    answer = result.get("answer") or ""
    mode = result.get("mode")
    ok = True
    reasons = []

    for token in case.get("expect_contains") or []:
        if token not in answer:
            ok = False
            reasons.append(f"missing in answer: {token!r}")

    modes = case.get("expect_mode_in")
    if modes and mode not in modes:
        # soft: agent may fall back to rag_llm
        if mode == "rag_llm" and "agent" in (modes or []) and result.get("fallback_from") == "agent":
            reasons.append(f"agent fell back to rag_llm: {result.get('agent_error')}")
        elif mode not in modes:
            ok = False
            reasons.append(f"mode={mode} not in {modes}")

    expect_tool = case.get("expect_tool")
    if expect_tool:
        names = _tool_names(result.get("tool_traces") or [])
        if expect_tool not in names:
            # If fell back to pure RAG, tool expectation fails
            ok = False
            reasons.append(f"expected tool {expect_tool!r}, got {sorted(names)}")

    return {
        "id": case.get("id"),
        "ok": ok,
        "mode": mode,
        "tools": sorted(_tool_names(result.get("tool_traces") or [])),
        "reasons": reasons,
        "answer_head": answer[:180],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run kb_agent_lab evals")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "evals" / "cases.json",
        help="cases json path",
    )
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    rows = [run_case(c) for c in cases]
    passed = sum(1 for r in rows if r["ok"])
    report = {
        "total": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass_rate": round(passed / len(rows), 3) if rows else 0.0,
        "results": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
