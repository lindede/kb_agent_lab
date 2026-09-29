"""FastAPI entry: /health, /ask, /ingest, and a simple browser UI."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from kb_agent_lab.answer import ask as ask_question
from kb_agent_lab.rag import ingest

app = FastAPI(title="kb_agent_lab", version="0.2.0")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    k: Optional[int] = Field(None, ge=1, le=20, description="检索条数")


class IngestRequest(BaseModel):
    reset: bool = True


_UI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>kb_agent_lab 试问</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif;
           max-width: 860px; margin: 40px auto; padding: 0 16px; color: #222; }
    h1 { font-size: 1.25rem; }
    h2 { font-size: 1rem; margin: 22px 0 8px; color: #1f3a5f; }
    textarea { width: 100%; min-height: 88px; font-size: 15px; padding: 10px; box-sizing: border-box; }
    button { margin-top: 10px; margin-right: 8px; padding: 8px 16px; font-size: 14px; cursor: pointer; }
    pre { background: #f5f7fa; border: 1px solid #e3e8ef; padding: 12px; overflow: auto;
          white-space: pre-wrap; word-break: break-word; font-size: 13px; margin: 0; }
    .panel { border: 1px solid #e3e8ef; border-left: 4px solid #2c5f8a; background: #fafbfd;
             padding: 10px 12px; margin-bottom: 12px; border-radius: 2px; }
    .panel.retrieved { border-left-color: #2a9d8f; }
    .panel.tools { border-left-color: #9b5de5; }
    .panel.prompt { border-left-color: #e9c46a; }
    .panel.llm { border-left-color: #e76f51; }
    .meta { color: #667; font-size: 12.5px; margin-bottom: 6px; }
    .hint { color: #667; font-size: 13px; margin: 8px 0 16px; }
    a { color: #2c5f8a; }
    .hit { margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px dashed #dde3ea; }
    .hit:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
  </style>
</head>
<body>
  <h1>kb_agent_lab · 试问</h1>
  <p class="hint">①预检索 ②工具轨迹 ③提示词 ④模型回答。默认 Agent 模式（LangGraph + Tools）。API：<a href="/docs">/docs</a></p>
  <textarea id="q" placeholder="例如：请先查知识库，这个项目要证明哪些能力？"></textarea>
  <div>
    <button id="askBtn" type="button">提问</button>
    <button id="ingestBtn" type="button">重新入库</button>
  </div>

  <h2>① 预检索内容 <span class="meta" id="modeMeta"></span></h2>
  <div class="panel retrieved" id="retrievedBox"><pre>（尚未提问）</pre></div>

  <h2>② 工具调用轨迹</h2>
  <div class="panel tools" id="toolsBox"><pre>（尚未提问）</pre></div>

  <h2>③ 传给大模型的提示词</h2>
  <div class="panel prompt" id="promptBox"><pre>（尚未提问）</pre></div>

  <h2>④ 大模型返回的内容</h2>
  <div class="panel llm" id="llmBox"><pre>（尚未提问）</pre></div>

  <h2>原始 JSON</h2>
  <pre id="rawOut">（尚未提问）</pre>

  <script>
    const q = document.getElementById('q');
    const retrievedBox = document.getElementById('retrievedBox');
    const toolsBox = document.getElementById('toolsBox');
    const promptBox = document.getElementById('promptBox');
    const llmBox = document.getElementById('llmBox');
    const rawOut = document.getElementById('rawOut');
    const modeMeta = document.getElementById('modeMeta');

    function esc(s) {
      return String(s ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
    }

    function renderRetrieved(items) {
      if (!items || !items.length) {
        retrievedBox.innerHTML = '<pre>（无预检索结果；Agent 仍可能通过 search_kb 自查）</pre>';
        return;
      }
      retrievedBox.innerHTML = items.map(it => `
        <div class="hit">
          <div class="meta">[${it.index}] ${esc(it.source)} · chunk=${it.chunk} · distance=${it.distance}</div>
          <pre>${esc(it.text)}</pre>
        </div>`).join('');
    }

    function renderTools(traces) {
      if (!traces || !traces.length) {
        toolsBox.innerHTML = '<pre>（本次无工具调用；若 mode=rag_llm 可能是回退到纯 RAG）</pre>';
        return;
      }
      toolsBox.innerHTML = traces.map((t, i) => {
        if (t.type === 'call') {
          return `<div class="hit"><div class="meta">#${i+1} CALL ${esc(t.name)}</div>
            <pre>${esc(JSON.stringify(t.args || {}, null, 2))}</pre></div>`;
        }
        return `<div class="hit"><div class="meta">#${i+1} RESULT ${esc(t.name)}</div>
          <pre>${esc(t.content || '')}</pre></div>`;
      }).join('');
    }

    function renderPrompt(prompt) {
      if (!prompt || (!prompt.system && !prompt.user)) {
        promptBox.innerHTML = '<pre>（未调用大模型，无提示词）</pre>';
        return;
      }
      let html = `
        <div class="meta">system</div>
        <pre>${esc(prompt.system || '')}</pre>
        <div class="meta" style="margin-top:10px">user</div>
        <pre>${esc(prompt.user || '')}</pre>`;
      if (prompt.tools && prompt.tools.length) {
        html += `<div class="meta" style="margin-top:10px">tools</div>
          <pre>${esc(JSON.stringify(prompt.tools, null, 2))}</pre>`;
      }
      promptBox.innerHTML = html;
    }

    function renderLlm(data) {
      const lr = data.llm_response;
      if (!lr) {
        llmBox.innerHTML = `<pre>${esc(data.answer || '（无大模型返回）')}</pre>
          <div class="meta">${esc(data.error || data.agent_error || data.mode || '')}</div>`;
        return;
      }
      let html = `<div class="meta">model=${esc(lr.model || data.model || '')}</div>
        <div class="meta">最终回答 content</div>
        <pre>${esc(lr.content || data.answer || '')}</pre>`;
      if (lr.reasoning_content) {
        html += `<div class="meta" style="margin-top:10px">reasoning_content</div>
          <pre>${esc(lr.reasoning_content)}</pre>`;
      }
      if (lr.usage) {
        html += `<div class="meta" style="margin-top:10px">usage</div>
          <pre>${esc(JSON.stringify(lr.usage, null, 2))}</pre>`;
      }
      if (data.agent_error) {
        html += `<div class="meta" style="margin-top:10px">agent_error</div>
          <pre>${esc(data.agent_error)}</pre>`;
      }
      llmBox.innerHTML = html;
    }

    document.getElementById('askBtn').onclick = async () => {
      const query = q.value.trim();
      if (!query) { rawOut.textContent = '请先输入问题'; return; }
      retrievedBox.innerHTML = toolsBox.innerHTML = promptBox.innerHTML = llmBox.innerHTML = '<pre>请求中…</pre>';
      rawOut.textContent = '请求中…';
      try {
        const r = await fetch('/ask?' + new URLSearchParams({ query }));
        const text = await r.text();
        let data;
        try { data = JSON.parse(text); }
        catch (_) {
          rawOut.textContent = 'HTTP ' + r.status + '\\n' + text;
          return;
        }
        modeMeta.textContent = `mode=${data.mode || ''} model=${data.model || ''}`;
        renderRetrieved(data.retrieved);
        renderTools(data.tool_traces);
        renderPrompt(data.prompt);
        renderLlm(data);
        rawOut.textContent = JSON.stringify(data, null, 2);
      } catch (e) {
        rawOut.textContent = String(e);
      }
    };

    document.getElementById('ingestBtn').onclick = async () => {
      rawOut.textContent = '入库中…';
      try {
        const r = await fetch('/ingest', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ reset: true })
        });
        rawOut.textContent = JSON.stringify(await r.json(), null, 2);
      } catch (e) {
        rawOut.textContent = String(e);
      }
    };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return _UI_HTML


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "kb_agent_lab"}


@app.get("/ask")
def ask_get(
    query: str = Query(..., min_length=1, description="用户问题"),
    k: Optional[int] = Query(None, ge=1, le=20, description="检索条数"),
) -> dict:
    return ask_question(query, k=k)


@app.post("/ask")
def ask_post(payload: AskRequest) -> dict:
    return ask_question(payload.query, k=payload.k)


@app.post("/ingest")
def ingest_endpoint(payload: Optional[IngestRequest] = None) -> dict:
    reset = True if payload is None else payload.reset
    try:
        return ingest(reset=reset)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
