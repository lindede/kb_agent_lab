# kb_agent_lab 项目说明

`kb_agent_lab` 是一个案头知识库 Agent 实验项目，用来验证求职所需的 Agent 工程能力。

## 要证明的能力

1. **RAG**：文档切分、向量入库、检索后带引用回答。
2. **Tool Calling**：至少提供 `search_kb`、`http_get`、`save_note` 等工具。
3. **编排**：用 LangGraph（或等价状态机）完成「检索 → 调工具 → 作答 → 失败重试」。
4. **评测**：`evals/` 固定题集，可一键回归。

## 技术栈

- Python + FastAPI
- Chroma 本地向量库
- 兼容 OpenAI 接口的 LLM API
- LangGraph（第二周接入）

## 使用方式

把 Markdown / 文本放入 `docs/`，运行 `scripts/ingest.py` 入库，再调用 `POST /ask` 提问。
