# kb_agent_lab

案头知识库 Agent 实验项目：用一个可演示的小产品，验证 **RAG + 向量库 + Tool Calling + LangGraph 编排 + 评测**。

- 本地：`~/work/mps/kb_agent_lab`
- 远程：https://github.com/lindede/kb_agent_lab

## 当前进度

- [x] **第 1 周**：语料入库（Chroma）+ `POST /ask` 检索带引用
- [x] **第 2 周**：Tool Calling（search_kb / http_get / save_note）+ LangGraph ReAct + `evals/` 回归

## 快速开始

```bash
cd ~/work/mps/kb_agent_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env   # 填入 OPENAI_API_KEY / BASE_URL / MODEL
# ASK_MODE=agent 默认走 LangGraph+Tools；改成 rag 则退回纯 RAG

python scripts/ingest.py
uvicorn kb_agent_lab.api:app --app-dir src --reload --port 8765
```

试问页：http://127.0.0.1:8765/  
评测：`python evals/run_eval.py`

## 架构（简图）

```
用户问题
  ├─ 预检索 Chroma → UI「① retrieved」
  └─ LangGraph ReAct Agent
        ├─ search_kb / http_get / save_note
        ├─ tool_traces → UI「②」
        └─ 最终回答 → UI「④」
```

Agent 失败时自动回退到经典 RAG（`mode=rag_llm`，带 `agent_error`）。

## 工具

| 工具 | 作用 |
| --- | --- |
| `search_kb` | 本地知识库向量检索 |
| `http_get` | 公开 URL GET（截断文本） |
| `save_note` | 追加写入 `data/notes/notes.md` |

## 两周里程碑

**第 1 周 — RAG + API**

- [x] 语料入库（`docs/` → Chroma）
- [x] `POST /ask`：检索 + 带引用回答
- [x] `.env` 配置模型 Key

**第 2 周 — Tools + 编排 + 评测**

- [x] 接入 ≥3 个 Tool
- [x] LangGraph 多步图跑通
- [x] `evals/run_eval.py` 可回归
- [ ] README 录屏（可选，自行补）
