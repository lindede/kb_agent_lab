# 本地运行备忘

## 环境

- 本地目录：`~/work/mps/kb_agent_lab`
- 远程仓库：`https://github.com/lindede/kb_agent_lab`

## 配置

复制 `.env.example` 为 `.env`，填入：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（可用国内兼容网关）
- `OPENAI_MODEL`
- `EMBEDDING_PROVIDER=local`（默认；很多网关不开 `/embeddings`，入库请用本地向量。若网关明确支持 embedding，再改成 `openai`）

未配置 Key 时，`/ask` 仍返回检索片段；有 Key 时用对话模型生成带引用回答（与 embedding 是否走远程无关）。
