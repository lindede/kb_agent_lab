# kb_agent_lab

案头知识库 Agent 实验项目：用一个可演示的小产品，验证 **RAG + 向量库 + Tool Calling + LangGraph 编排 + 评测**。

- 本地：`~/work/mps/kb_agent_lab`
- 远程：https://github.com/lindede/kb_agent_lab

## 当前进度

- [x] **第 1 周**：语料入库（Chroma）+ `POST /ask` 检索带引用（有 Key 则 LLM 作答）
- [ ] **第 2 周**：Tool Calling + LangGraph + `evals/` 回归

## 快速开始

```bash
cd ~/work/mps/kb_agent_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env   # 可选：填入 OPENAI_API_KEY / BASE_URL / MODEL

# 把 Markdown 放入 docs/ 后入库
python scripts/ingest.py

# 启动 API
uvicorn kb_agent_lab.api:app --app-dir src --reload --port 8765
```

另开终端试问：

```bash
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/ask \
  -H 'content-type: application/json' \
  -d '{"query":"这个项目要证明哪些能力？"}'
```

未配置 API Key 时，`/ask` 仍返回检索片段（`mode=retrieval_only`）；配置后为 `mode=rag_llm`，答案带 `[1][2]` 引用。

也可通过接口重新入库：`POST /ingest`，body `{"reset": true}`。

## 目录

```
kb_agent_lab/
├── docs/                 # 知识库语料
├── src/kb_agent_lab/
│   ├── api.py            # FastAPI：/health /ask /ingest
│   ├── answer.py         # 检索 + LLM 作答
│   ├── rag.py            # 切分 / Chroma 入库 / 检索
│   ├── tools.py          # Week 2
│   └── graph.py          # Week 2
├── scripts/ingest.py
└── evals/                # Week 2
```

## 两周里程碑

**第 1 周 — RAG + API**（已完成骨架可跑）

- [x] 语料入库（`docs/` → Chroma）
- [x] `POST /ask`：检索 + 带引用回答
- [x] `.env` 配置模型 Key

**第 2 周 — Tools + 编排 + 评测**

- [ ] 接入 ≥3 个 Tool
- [ ] LangGraph 多步图跑通
- [ ] `evals/run_eval.py` 可回归
- [ ] README 补架构图 + 录屏
