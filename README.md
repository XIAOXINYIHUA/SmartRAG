# 📚 SmartRAG - 智能文档问答助手

[English](#english) | [中文](#中文)

---

## 中文

### 简介

SmartRAG 是一个基于 LangChain + ChromaDB 的智能文档问答系统。它能自动导入 PDF、TXT、Markdown 等文档，构建向量知识库，并通过 Agent 自动决策是检索知识库还是直接回答。

### 功能特性

- **多格式文档导入**：支持 PDF / TXT / Markdown / 网页 URL
- **智能检索增强**：基于 ChromaDB 向量相似度检索，精准定位答案来源
- **Agent 自动决策**：Agent 判断问题是否需要检索文档，通用问题直接回答
- **多轮对话记忆**：保留对话上下文，支持追问和上下文引用
- **Web UI**：基于 Streamlit 的友好界面，开箱即用
- **多 LLM 支持**：可切换 OpenAI / 通义千问

### 架构

```
用户提问
  │
  ▼
┌──────────┐    retrieve     ┌──────────────┐
│  Agent   │ ───────────────▶│  RAG Chain   │
│ (Router) │                 │  检索 + 生成  │
└──────────┘                 └──────┬───────┘
  │                                │
  │ direct                         │
  ▼                                ▼
┌──────────┐              ┌────────────────┐
│  直接回答 │              │ 向量库检索结果  │
└──────────┘              │ + LLM 生成回答  │
                          └────────────────┘
```

### 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/your-username/SmartRAG.git
cd SmartRAG

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 OPENAI_API_KEY

# 4. 启动
streamlit run app.py
```

### 使用方式

1. 在左侧侧边栏上传文档或输入 URL
2. 点击"加载文档"构建知识库
3. 在对话框中提问，系统自动判断是否需要检索

### 项目结构

```
SmartRAG/
├── app.py                 # Streamlit Web UI 入口
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量模板
├── src/
│   ├── __init__.py
│   ├── loader.py          # 文档加载（PDF/TXT/URL）
│   ├── splitter.py        # 文本分割（RecursiveCharacterTextSplitter）
│   ├── embedding.py       # 向量化 + ChromaDB 存储
│   ├── chain.py           # RAG Chain 组装
│   └── agent.py           # Agent 决策层（Router → RAG/Direct）
└── data/
    └── sample.md          # 示例文档
```

---

## English

### Introduction

SmartRAG is an intelligent document Q&A system built with LangChain and ChromaDB. It imports documents (PDF, TXT, Markdown, URLs), builds a vector knowledge base, and uses an Agent to decide whether to retrieve from the knowledge base or answer directly.

### Features

- **Multi-format Import**: PDF / TXT / Markdown / Web URL
- **RAG-powered QA**: ChromaDB vector similarity search for precise answers
- **Agent Decision Router**: Auto-detects if a question needs document retrieval
- **Multi-turn Memory**: Maintains conversation context across interactions
- **Streamlit Web UI**: Clean, interactive interface
- **Multi-LLM Support**: Switch between OpenAI and DashScope

### Quick Start

```bash
git clone https://github.com/your-username/SmartRAG.git
cd SmartRAG
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
streamlit run app.py
```

---

## License

MIT License
