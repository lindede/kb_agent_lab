"""Answer a question with retrieved context + optional LLM."""

from __future__ import annotations

from typing import Any

from kb_agent_lab import config
from kb_agent_lab.rag import retrieve


SYSTEM_PROMPT = """你是知识库助手。只根据给定「检索片段」回答用户问题。
规则：
1. 优先使用片段中的事实；不确定就明确说资料不足。
2. 回答使用简洁中文。
3. 在答案末尾用 [1][2] 形式标注引用编号（对应片段编号）。
"""


def _format_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] 来源: {h.get('source', '')}\n{h.get('text', '')}")
    return "\n\n".join(blocks)


def ask(query: str, k: int | None = None) -> dict[str, Any]:
    hits = retrieve(query, k=k)
    citations = [
        {
            "index": i + 1,
            "source": h.get("source", ""),
            "chunk": h.get("chunk"),
            "excerpt": (h.get("text") or "")[:240],
            "distance": h.get("distance"),
        }
        for i, h in enumerate(hits)
    ]

    if not hits:
        return {
            "query": query,
            "answer": "知识库中没有检索到相关内容。请先放入 docs/ 并运行 scripts/ingest.py。",
            "citations": [],
            "mode": "empty",
        }

    if not config.has_llm_key():
        # Retrieval-only fallback for local smoke without API key.
        preview = "\n\n".join(
            f"[{c['index']}] ({c['source']}) {c['excerpt']}" for c in citations
        )
        return {
            "query": query,
            "answer": "当前未配置 OPENAI_API_KEY，仅返回检索结果：\n\n" + preview,
            "citations": citations,
            "mode": "retrieval_only",
        }

    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
    user_content = f"问题：{query}\n\n检索片段：\n{_format_context(hits)}"
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    answer = (resp.choices[0].message.content or "").strip()
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "mode": "rag_llm",
        "model": config.OPENAI_MODEL,
    }
