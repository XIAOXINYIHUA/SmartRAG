"""向量化 + ChromaDB 向量存储 — 多平台支持"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .config import get_provider_config, get_api_key, get_embedding_provider


def get_embeddings(provider: str = "dashscope", model: str = None) -> OpenAIEmbeddings:
    """获取嵌入模型 — 统一走 OpenAI 兼容协议，支持所有平台"""
    emb_provider = get_embedding_provider(provider)
    cfg = get_provider_config(emb_provider)
    model = model or cfg["default_embedding"]

    return OpenAIEmbeddings(
        model=model,
        openai_api_key=get_api_key(emb_provider),
        openai_api_base=cfg["base_url"],
    )


def create_vectorstore(
    documents: List[Document],
    embeddings: OpenAIEmbeddings = None,
    persist_dir: str = "./chroma_db",
) -> Chroma:
    """从文档创建向量库"""
    embeddings = embeddings or get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def load_vectorstore(
    persist_dir: str = "./chroma_db",
    embeddings: OpenAIEmbeddings = None,
) -> Optional[Chroma]:
    """加载已有向量库"""
    if not os.path.exists(persist_dir):
        return None
    embeddings = embeddings or get_embeddings()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
