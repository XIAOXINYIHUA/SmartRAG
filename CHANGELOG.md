# Changelog

## v0.3.0 (2026-05-29)

### 修复

- **内联 src 模块依赖** — 将 config / loader / splitter / embedding / chain / agent 全部内联到 `app.py`，彻底解决 `from src.*` 导入失败问题，部署时无需拷贝 `src/` 目录
- **修复 DashScope Embedding batch_size 限制** — 批量大小从 25 降至 10，避免 API 返回 400 错误
- **修复 Pydantic BaseRetriever 兼容性** — 用 `RunnableLambda` 替代继承 `BaseRetriever`，解决 LangChain 新版 Pydantic v2 类型校验错误
- **api.py 独立化** — 内联 `load_files_safe`，减少对 `src/` 模块的依赖

---

## v0.2.0 (2026-05-27)

### 新增

- **6 大 LLM 平台支持**
  - 阿里云 DashScope（通义千问）
  - OpenAI（GPT-4o / GPT-4o-mini）
  - DeepSeek（deepseek-chat / deepseek-reasoner）
  - 硅基流动 SiliconFlow（Qwen3-235B / DeepSeek-V3 / Llama-3.3）
  - 智谱 AI GLM（glm-4-flash / glm-4-plus）
  - Ollama 本地模型
- **多知识库** — 支持命名知识库（collection），侧边栏切换，互相独立
- **流式输出** — 基于 LangChain streaming + Streamlit `st.write_stream` 实时逐 token 输出
- **检索多样性控制**
  - 相似度检索（similarity）
  - MMR（Maximal Marginal Relevance）多样性检索
- **重排序 (Reranker)**
  - DashScope gte-rerank 在线重排
  - 本地 Cross-Encoder（BAAI/bge-reranker-base）备选
- **嵌入去重** — MD5 哈希比对，已存在的文档片段自动跳过，避免重复向量化
- **Agent 自动路由** — 关键词匹配判断问题是否需要检索，支持 LLM Router 模式
- **可配置分块** — 侧边栏实时调整 chunk_size / chunk_overlap（200~4000 / 0~500）
- **侧边栏 API Key 输入** — 无需 .env 文件，直接在 UI 输入各平台密钥
- **REST API** — FastAPI 封装，提供 `/chat` / `/load` / `/upload` / `/kb` / `/providers` 接口
- **Docker / docker-compose 支持** — 一键启动 Web UI + API 两个服务
- **单元测试** — pytest 覆盖 loader / splitter / config / embedding / agent

### 变更

- 项目架构从单文件扩展为 `src/` 模块化（v0.2.0），后于 v0.3.0 重归内联

---

## v0.1.0 (2026-05-27)

### 新增

- SmartRAG 初始版本
- 基于 LangChain + ChromaDB 的文档问答系统
- 支持 PDF / TXT / Markdown 文档导入
- Streamlit Web UI
- Agent 自动决策（检索知识库或直接回答）
- 多轮对话记忆
