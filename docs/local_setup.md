# 本地运行备忘

## 环境

- 本地目录：`~/work/mps/kb_agent_lab`
- 远程仓库：`https://github.com/lindede/kb_agent_lab`

## 配置

复制 `.env.example` 为 `.env`，填入：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（可用国内兼容网关）
- `OPENAI_MODEL`

若暂时没有 Key，`/ask` 仍可返回检索片段（retrieval-only），但不生成完整 LLM 回答。
