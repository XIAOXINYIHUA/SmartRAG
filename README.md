# 📚 SmartRAG - 智能文档问答助手

[English](#english) | [中文](#中文)

---

## 中文

### 简介

SmartRAG 是一个基于 LangChain + ChromaDB 的智能文档问答系统。支持 6 大 LLM 平台，自动导入 PDF / TXT / Markdown / 网页文档构建向量知识库，并通过 Agent 自动决策是检索知识库还是直接回答。

### 功能特性

- **🌐 6 大 LLM 平台支持**
  - 阿里云 DashScope（通义千问）
  - OpenAI（GPT-4o / GPT-4o-mini）
  - DeepSeek（deepseek-chat / deepseek-reasoner）
  - 硅基流动 SiliconFlow（Qwen3-235B / DeepSeek-V3 / Llama-3.3）
  - 智谱 AI GLM（glm-4-flash / glm-4-plus）
  - Ollama 本地模型（无需 API Key）
- **📄 多格式文档导入**：支持 PDF / TXT / Markdown / 网页 URL
- **🧠 智能检索增强**：基于 ChromaDB 向量相似度检索，精准定位答案来源
- **🤖 Agent 自动决策**：关键词 / LLM 双模式路由，判断问题是否需要检索
- **💬 流式输出**：实时逐 token 流式显示，响应更快更流畅
- **📚 多知识库**：支持命名知识库，侧边栏一键切换，互相独立
- **🔍 多种检索方式**：相似度检索 + MMR 多样性检索
- **🎯 重排序 (Reranker)**：DashScope gte-rerank 在线重排 / 本地 BGE reranker 备选
- **🧹 嵌入去重**：MD5 哈希自动去重，避免重复向量化
- **✂️ 可配置分块**：实时调整 chunk_size / chunk_overlap
- **💻 Web UI**：基于 Streamlit 的友好界面，侧边栏集中配置
- **🔌 REST API**：FastAPI 封装，支持 /chat /load /upload /kb 等接口
- **🐳 Docker 支持**：Dockerfile + docker-compose，一键启动 Web + API 服务
- **✅ 单元测试**：pytest 覆盖核心模块

### 架构

```
用户提问
  │
  ▼
┌──────────────┐    retrieve     ┌──────────────────┐
│ Agent Router  │ ───────────────▶│   RAG Chain      │
│ (keyword/LLM) │                 │  检索 → 重排 → 生成 │
└──────────────┘                 └────────┬─────────┘
  │                                       │
  │ direct                                │
  ▼                                       ▼
┌──────────────┐                 ┌──────────────────┐
│  直接回答     │                 │ 📖 基于文档的回答  │
│  LLM 通用知识  │                 │ + 来源标注        │
└──────────────┘                 └──────────────────┘
```

### 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/XIAOXINYIHUA/SmartRAG.git
cd SmartRAG

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 API Key（至少配置一个平台）

# 4. 启动 Web UI
streamlit run app.py
```

### Docker 部署

```bash
# 启动 Web UI（http://localhost:8501）
docker compose up smartrag-web

# 同时启动 API（http://localhost:8000）
docker compose up -d
```

### REST API

```bash
# 启动 API 服务
uvicorn api:app --host 0.0.0.0 --port 8000

# 查看 API 文档
# 浏览器打开 http://localhost:8000/docs
```

#### 问答接口

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "文档中提到了什么？",
    "provider": "dashscope",
    "kb_name": "smartrag"
  }'
```

#### 上传文档

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@report.pdf" \
  -F "kb_name=my-knowledge"
```

### 使用方式

1. 在左侧侧边栏选择 LLM 平台并输入 API Key
2. 配置检索参数（top-k、检索方式、重排序）
3. 上传文档或输入 URL，点击"加载文档"构建知识库
4. 在对话框中提问，系统自动判断是否需要检索

### 支持平台一览

| 平台 | LLM | Embedding | 配置项 |
|------|-----|-----------|--------|
| DashScope | qwen-plus / qwen-max / qwen-turbo | text-embedding-v3 / v2 / v1 | `DASHSCOPE_API_KEY` |
| OpenAI | gpt-4o / gpt-4o-mini / gpt-4-turbo | text-embedding-3-small / 3-large | `OPENAI_API_KEY` |
| DeepSeek | deepseek-chat / deepseek-reasoner | ❌（回退 DashScope） | `DEEPSEEK_API_KEY` |
| SiliconFlow | Qwen3-235B / DeepSeek-V3 / Llama-3.3 | bge-large-zh-v1.5 / bge-m3 | `SILICONFLOW_API_KEY` |
| Zhipu AI | glm-4-flash / glm-4-plus / glm-4-long | embedding-2 | `ZHIPU_API_KEY` |
| Ollama | qwen3 / llama3 / deepseek-r1 | nomic-embed-text / bge-m3 | 无需 Key |

### 项目结构

```
SmartRAG/
├── app.py                 # Streamlit Web UI（已内联所有模块，单文件部署）
├── api.py                 # FastAPI REST API
├── requirements.txt       # Python 依赖
├── Dockerfile             # Docker 镜像构建
├── docker-compose.yml     # 多服务编排
├── CHANGELOG.md           # 版本日志
├── .env.example           # 环境变量模板
├── src/                   # 模块源码（可选，app.py 已内联所有功能）
│   ├── loader.py
│   ├── splitter.py
│   ├── embedding.py
│   ├── chain.py
│   ├── agent.py
│   └── config.py
├── tests/
│   └── test_core.py       # 单元测试
└── data/
    └── sample.md          # 示例文档
```

---

## English

### Introduction

SmartRAG is an intelligent document Q&A system built with LangChain and ChromaDB. It supports 6 LLM platforms, imports documents (PDF, TXT, Markdown, URLs), builds a vector knowledge base, and uses an Agent to decide whether to retrieve from the knowledge base or answer directly.

### Features

- **6 LLM Providers**: DashScope, OpenAI, DeepSeek, SiliconFlow, Zhipu AI, Ollama
- **Multi-format Import**: PDF / TXT / Markdown / Web URL
- **RAG-powered QA**: ChromaDB vector similarity search for precise answers
- **Agent Decision Router**: Keyword-based or LLM-based routing to RAG or direct answer
- **Streaming Output**: Real-time token-by-token streaming
- **Multi-Knowledge-Base**: Named collections, switch independently
- **Search Modes**: Similarity search + MMR diverse retrieval
- **Reranker**: DashScope gte-rerank online or local BGE Cross-Encoder
- **Embedding Dedup**: MD5 hash-based deduplication
- **Configurable Chunking**: Adjustable chunk_size / chunk_overlap
- **Streamlit Web UI**: Clean, interactive interface with sidebar config
- **REST API**: FastAPI endpoints for chat, document load, upload, KB management
- **Docker Support**: Dockerfile + docker-compose for Web UI and API
- **Tests**: pytest coverage

### Quick Start

```bash
git clone https://github.com/XIAOXINYIHUA/SmartRAG.git
cd SmartRAG
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key
streamlit run app.py
```

---

## License

MIT License
