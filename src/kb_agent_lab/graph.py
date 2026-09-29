"""LangGraph ReAct agent: retrieve context + tool calling + answer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from kb_agent_lab import config
from kb_agent_lab.rag import retrieve
from kb_agent_lab.tools import get_langchain_tools, tools_manifest

AGENT_SYSTEM = """你是知识库 Agent，可以调用工具完成任务。
工具：
- search_kb: 检索本地知识库（回答项目/文档问题优先使用）
- http_get: 抓取公开网页文本
- save_note: 把内容追加保存到本地笔记

规则：
1. 涉及本项目或 docs 内容时，先 search_kb 再回答。
2. 只有确实需要外部网页时才 http_get。
3. 用户明确要求记下/保存时才 save_note。
4. 最终用简洁中文回答；若依据知识库片段，用 [1][2] 标注。
5. 不要编造工具结果里没有的事实。
"""


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        temperature=config.OPENAI_TEMPERATURE,
    )


def build_graph():
    """Build a ReAct-style LangGraph agent with our tools."""
    return create_react_agent(
        _build_llm(),
        get_langchain_tools(),
        prompt=AGENT_SYSTEM,
    )


def _retrieved_items(hits: List[dict]) -> List[dict]:
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


def _extract_tool_traces(messages: List[BaseMessage]) -> List[dict]:
    traces: List[dict] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                traces.append(
                    {
                        "type": "call",
                        "name": tc.get("name"),
                        "args": tc.get("args") or {},
                        "id": tc.get("id"),
                    }
                )
        elif isinstance(msg, ToolMessage):
            content = msg.content
            if not isinstance(content, str):
                content = str(content)
            traces.append(
                {
                    "type": "result",
                    "name": msg.name,
                    "content": content[:4000],
                    "tool_call_id": msg.tool_call_id,
                }
            )
    return traces


def _final_ai_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and (msg.content or "") and not getattr(msg, "tool_calls", None):
            return str(msg.content).strip()
        if isinstance(msg, AIMessage) and msg.content and getattr(msg, "tool_calls", None):
            # some models put final text alongside tool_calls; keep looking
            continue
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content).strip()
    return ""


def run_agent(query: str, k: Optional[int] = None) -> Dict[str, Any]:
    """
    Run retrieve (for UI) + LangGraph ReAct agent.
    Returns a response payload compatible with the try-it UI.
    """
    query = (query or "").strip()
    hits = retrieve(query, k=k or config.RETRIEVE_K)
    retrieved = _retrieved_items(hits)

    prompt = {
        "system": AGENT_SYSTEM,
        "user": query,
        "tools": tools_manifest(),
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": query},
        ],
    }

    graph = build_graph()
    # recursion_limit roughly bounds tool loops
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": max(4, config.AGENT_MAX_ITERATIONS * 2 + 2)},
    )
    messages: List[BaseMessage] = list(result.get("messages") or [])
    tool_traces = _extract_tool_traces(messages)
    answer = _final_ai_text(messages)

    llm_response: Dict[str, Any] = {
        "content": answer,
        "model": config.OPENAI_MODEL,
        "message_count": len(messages),
    }

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

    return {
        "query": query,
        "mode": "agent",
        "model": config.OPENAI_MODEL,
        "retrieved": retrieved,
        "prompt": prompt,
        "tool_traces": tool_traces,
        "llm_response": llm_response,
        "answer": answer or "（模型未返回最终文本）",
        "citations": citations,
    }
