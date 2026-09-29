"""Agent tools: search_kb / http_get / save_note."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, List
from urllib.parse import urlparse

import httpx
from langchain_core.tools import StructuredTool

from kb_agent_lab import config
from kb_agent_lab.rag import retrieve


def search_kb(query: str) -> str:
    """Search the local knowledge base and return relevant passages with sources."""
    query = (query or "").strip()
    if not query:
        return "error: empty query"
    hits = retrieve(query, k=config.RETRIEVE_K)
    if not hits:
        return "no results"
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] source={h.get('source')} distance={h.get('distance')}\n{h.get('text', '')}"
        )
    return "\n\n".join(blocks)


def http_get(url: str) -> str:
    """Fetch a public http(s) URL and return truncated plain text."""
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "error: only http/https URLs are allowed"
    try:
        with httpx.Client(timeout=12.0, follow_redirects=True) as client:
            resp = client.get(
                url,
                headers={"User-Agent": "kb_agent_lab/0.1 (+local-demo)"},
            )
        text = resp.text
        if len(text) > 8000:
            text = text[:8000] + "\n…[truncated]"
        return f"status={resp.status_code}\n{text}"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def save_note(text: str) -> str:
    """Append a note to the local notes file and return the saved path."""
    text = (text or "").strip()
    if not text:
        return "error: empty note"
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.NOTES_DIR / "notes.md"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {stamp}\n\n{text}\n")
    return f"saved: {path}"


def get_langchain_tools() -> List[StructuredTool]:
    """Tools bound for LangGraph / ChatOpenAI tool calling."""

    def _wrap(fn: Callable[..., str], name: str, description: str) -> StructuredTool:
        return StructuredTool.from_function(func=fn, name=name, description=description)

    return [
        _wrap(
            search_kb,
            "search_kb",
            "检索本地知识库 docs（向量检索），返回带 source 的相关片段。回答项目/文档问题优先用它。",
        ),
        _wrap(
            http_get,
            "http_get",
            "对公开 http/https URL 发起 GET，返回截断后的文本。需要联网页面内容时使用。",
        ),
        _wrap(
            save_note,
            "save_note",
            "把一段笔记追加写入本地 data/notes/notes.md。用户要求记录/保存时使用。",
        ),
    ]


def tools_manifest() -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description}
        for t in get_langchain_tools()
    ]
