"""FastAPI entry: /health, /ask, /ingest."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kb_agent_lab.answer import ask as ask_question
from kb_agent_lab.rag import ingest

app = FastAPI(title="kb_agent_lab", version="0.1.0")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    k: Optional[int] = Field(None, ge=1, le=20, description="检索条数")


class IngestRequest(BaseModel):
    reset: bool = True


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "kb_agent_lab"}


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    return ask_question(payload.query, k=payload.k)


@app.post("/ingest")
def ingest_endpoint(payload: Optional[IngestRequest] = None) -> dict:
    reset = True if payload is None else payload.reset
    try:
        return ingest(reset=reset)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
