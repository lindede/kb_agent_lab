"""FastAPI entry. Week 1: /health + /ask stub."""

from fastapi import FastAPI

app = FastAPI(title="kb_agent_lab", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True, "service": "kb_agent_lab"}


@app.post("/ask")
def ask(payload: dict):
    # Week 1: RAG answer with citations
    return {
        "answer": "Not implemented yet — week-1 milestone.",
        "query": payload.get("query", ""),
        "citations": [],
    }
