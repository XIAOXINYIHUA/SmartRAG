"""向量化 + ChromaDB 向量存储"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma


def get_embeddings(model: str = "openai"):
    """获取嵌入模型（支持 OpenAI / 通义千问 DashScope）"""
    if model == "dashscope" or os.getenv("EMBEDDING_PROVIDER") == "dashscope":
        return DashScopeEmbeddings(model="text-embedding-v3")
    return OpenAIEmbeddings(model="text-embedding-3-small")


def create_vectorstore(
    documents: List[Document],
    embeddings=None,
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
    embeddings=None,
) -> Optional[Chroma]:
    """加载已有向量库"""
    if not os.path.exists(persist_dir):
        return None
    embeddings = embeddings or get_embeddings()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
