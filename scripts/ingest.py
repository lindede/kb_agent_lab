#!/usr/bin/env python3
"""Ingest docs/ into Chroma."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kb_agent_lab.rag import ingest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest docs into Chroma")
    parser.add_argument("--docs", type=Path, default=None, help="docs directory")
    parser.add_argument("--no-reset", action="store_true", help="append without wiping collection")
    args = parser.parse_args()
    result = ingest(docs_dir=args.docs, reset=not args.no_reset)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
