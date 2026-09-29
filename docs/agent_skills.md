# Agent 关键技术要点

面向 AI Agent 开发岗位，常见硬技能包括：

## RAG 与向量库

- 文档切分（chunk）与重叠窗口
- Embedding 向量化
- 向量库检索（如 Chroma、pgvector、Milvus）
- 回答时附带引用（citations），降低幻觉

## Tool Calling

- 把搜索、HTTP、写笔记、计算器等封装成工具
- 由模型决定何时调用、传什么参数
- 工具结果再喂回模型生成最终答案

## 编排（Orchestration）

- LangGraph / 状态机：多步任务、分支、重试
- 典型链路：理解问题 → 检索 → 调工具 → 汇总回答

## 工程化

- FastAPI 服务化
- 日志、耗时、评测集回归
- API Key 与配置通过环境变量管理
