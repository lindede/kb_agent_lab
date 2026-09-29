"""Answer a question with retrieved context + optional LLM / Agent."""

from __future__ import annotations

from typing import Any, Optional

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


def _build_user_prompt(query: str, hits: list[dict[str, Any]]) -> str:
    return f"问题：{query}\n\n检索片段：\n{_format_context(hits)}"


def _retrieved_items(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for i, h in enumerate(hits, start=1):
        items.append(
            {
                "index": i,
                "source": h.get("source", ""),
                "chunk": h.get("chunk"),
                "text": h.get("text", ""),
                "distance": h.get("distance"),
            }
        )
    return items


def _prompt_payload(system: Optional[str], user: Optional[str]) -> dict[str, Any]:
    return {
        "system": system,
        "user": user,
        "messages": (
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            if system is not None and user is not None
            else []
        ),
    }


def ask_rag(query: str, k: Optional[int] = None) -> dict[str, Any]:
    """Classic RAG: retrieve then (optional) LLM answer. No tool calling."""
    hits = retrieve(query, k=k)
    retrieved = _retrieved_items(hits)
    citations = [
        {
            "index": item["index"],
            "source": item["source"],
            "chunk": item["chunk"],
            "excerpt": (item["text"] or "")[:240],
            "distance": item["distance"],
        }
        for item in retrieved
    ]

    if not hits:
        return {
            "query": query,
            "mode": "empty",
            "retrieved": [],
            "prompt": _prompt_payload(None, None),
            "tool_traces": [],
            "llm_response": None,
            "answer": "知识库中没有检索到相关内容。请先放入 docs/ 并运行 scripts/ingest.py。",
            "citations": [],
        }

    user_prompt = _build_user_prompt(query, hits)

    if not config.has_llm_key():
        preview = "\n\n".join(
            f"[{c['index']}] ({c['source']}) {c['excerpt']}" for c in citations
        )
        return {
            "query": query,
            "mode": "retrieval_only",
            "retrieved": retrieved,
            "prompt": _prompt_payload(None, None),
            "tool_traces": [],
            "llm_response": None,
            "answer": "当前未配置 OPENAI_API_KEY，仅返回检索结果：\n\n" + preview,
            "citations": citations,
        }

    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=config.OPENAI_TEMPERATURE,
        )
        msg = resp.choices[0].message
        answer = (msg.content or "").strip()
        llm_response: dict[str, Any] = {
            "content": answer,
            "model": config.OPENAI_MODEL,
            "usage": resp.usage.model_dump() if resp.usage else None,
        }
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            llm_response["reasoning_content"] = reasoning

        return {
            "query": query,
            "mode": "rag_llm",
            "model": config.OPENAI_MODEL,
            "retrieved": retrieved,
            "prompt": _prompt_payload(SYSTEM_PROMPT, user_prompt),
            "tool_traces": [],
            "llm_response": llm_response,
            "answer": answer,
            "citations": citations,
        }
    except Exception as exc:  # noqa: BLE001
        preview = "\n\n".join(
            f"[{c['index']}] ({c['source']}) {c['excerpt']}" for c in citations
        )
        return {
            "query": query,
            "mode": "retrieval_fallback",
            "model": config.OPENAI_MODEL,
            "retrieved": retrieved,
            "prompt": _prompt_payload(SYSTEM_PROMPT, user_prompt),
            "tool_traces": [],
            "llm_response": None,
            "answer": "大模型调用失败，已回退为检索结果。\n"
            f"原因：{exc}\n"
            f"请检查 .env 里 OPENAI_BASE_URL / OPENAI_MODEL（当前 model={config.OPENAI_MODEL}）。\n\n"
            + preview,
            "citations": citations,
            "error": str(exc),
        }


def ask(query: str, k: Optional[int] = None) -> dict[str, Any]:
    """
    Default entry: agent mode (LangGraph + tools) when configured,
    otherwise classic RAG. Agent failures fall back to RAG.
    """
    use_agent = config.has_llm_key() and config.ASK_MODE in {"agent", "tools", "react"}
    if use_agent:
        try:
            from kb_agent_lab.graph import run_agent

            return run_agent(query, k=k)
        except Exception as exc:  # noqa: BLE001
            fallback = ask_rag(query, k=k)
            fallback["agent_error"] = str(exc)
            fallback["fallback_from"] = "agent"
            return fallback
    return ask_rag(query, k=k)
