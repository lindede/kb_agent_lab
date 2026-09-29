# kb_agent_lab

案头知识库 Agent 实验项目：用一个可演示的小产品，验证 **RAG + 向量库 + Tool Calling + LangGraph 编排 + 评测**。

面向求职作品集；范围刻意做小，两周内能跑通主链路即可。

## 目标能力

| 模块 | 内容 |
| --- | --- |
| RAG | 文档切分 → embedding → 向量检索 → 带引用回答 |
| Tools | ≥3 个工具：`search_kb`、`http_get`、`save_note`（可再加 calculator） |
| 编排 | LangGraph：检索 → 是否调工具 → 作答 → 失败重试 |
| 评测 | `evals/` 固定题集 + 一键回归脚本 |

## 技术栈（首期）

- Python 3.11+
- FastAPI
- LangGraph
- 向量库：Chroma（本地）
- 任意兼容 OpenAI 接口的 LLM API

## 目录

```
kb_agent_lab/
├── README.md
├── requirements.txt
├── .env.example
├── docs/                 # 放入你的知识库语料（简历说明、项目笔记等）
├── src/
│   └── kb_agent_lab/
│       ├── __init__.py
│       ├── api.py        # FastAPI 入口
│       ├── graph.py      # LangGraph 编排
│       ├── rag.py        # 切分 / 建库 / 检索
│       ├── tools.py      # Tool Calling
│       └── config.py
├── evals/
│   ├── cases.json        # 评测题
│   └── run_eval.py
└── scripts/
    └── ingest.py         # 语料入库
```

## 两周里程碑

**第 1 周 — RAG + API**

- [ ] 语料入库（`docs/` → Chroma）
- [ ] `POST /ask`：检索 + 带引用回答
- [ ] `.env` 配置模型 Key

**第 2 周 — Tools + 编排 + 评测**

- [ ] 接入 ≥3 个 Tool
- [ ] LangGraph 多步图跑通
- [ ] `evals/run_eval.py` 可回归
- [ ] README 补架构说明 + 录屏链接

## 本地与远程

- 本地：`~/work/mps/kb_agent_lab`
- 远程：`https://github.com/lindede/kb_agent_lab`（Public）

## 快速开始（骨架就绪后）

```bash
cd ~/work/mps/kb_agent_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填入 API Key
```

（实现代码按里程碑逐步补齐。）
