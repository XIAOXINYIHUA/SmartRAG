"""向量化 + ChromaDB 向量存储 — 直接使用 chromadb 原生 API"""

import hashlib
import logging
import os
from typing import List, Optional

from dashscope import TextEmbedding
from langchain_core.documents import Document

from .config import get_provider_config, get_api_key, get_embedding_provider

logger = logging.getLogger(__name__)

# 嵌入 API 批量大小
EMBEDDING_BATCH_SIZE = 25


def _content_hash(text: str) -> str:
    """对文本内容生成 MD5 hash，用作去重 ID"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class SmartRAGVectorStore:
    """基于 chromadb 的向量存储，直接管理 embedding 计算"""

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "smartrag"):
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = None
        self._collection_name = collection_name
        self._provider = "dashscope"
        self._model = "text-embedding-v3"
        self._api_key = None
        self._persist_dir = persist_dir

    def _get_emb_provider(self, provider: str) -> str:
        return get_embedding_provider(provider)

    def _compute_embeddings(self, texts: List[str], provider: str = None,
                            model: str = None) -> List[List[float]]:
        """分批计算 embedding，避免单次请求过大"""
        emb_provider = self._get_emb_provider(provider or self._provider)
        cfg = get_provider_config(emb_provider)
        model = model or cfg["default_embedding"]
        api_key = get_api_key(emb_provider)

        all_embeddings = []

        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            logger.info(f"Embedding batch {i // EMBEDDING_BATCH_SIZE + 1}/"
                        f"{(len(texts) - 1) // EMBEDDING_BATCH_SIZE + 1} "
                        f"({len(batch)} texts)")

            if emb_provider == "dashscope":
                resp = TextEmbedding.call(
                    model=model, input=batch, api_key=api_key
                )
                if resp.status_code != 200:
                    raise ValueError(
                        f"DashScope embedding error [{resp.status_code}]: {resp.message}"
                    )
                all_embeddings.extend(
                    item["embedding"] for item in resp.output["embeddings"]
                )
            else:
                from langchain_openai import OpenAIEmbeddings
                emb = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=api_key,
                    openai_api_base=cfg["base_url"],
                )
                all_embeddings.extend(emb.embed_documents(batch))

        return all_embeddings

    def _compute_query_embedding(self, text: str, provider: str = None,
                                  model: str = None) -> List[float]:
        emb_provider = self._get_emb_provider(provider or self._provider)
        cfg = get_provider_config(emb_provider)
        model = model or cfg["default_embedding"]
        api_key = get_api_key(emb_provider)

        if emb_provider == "dashscope":
            resp = TextEmbedding.call(
                model=model, input=[text], api_key=api_key
            )
            if resp.status_code != 200:
                raise ValueError(
                    f"DashScope embedding error [{resp.status_code}]: {resp.message}"
                )
            return resp.output["embeddings"][0]["embedding"]

        from langchain_openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            openai_api_base=cfg["base_url"],
        )
        return emb.embed_query(text)

    def from_documents(self, documents: List[Document], provider: str = None,
                       model: str = None):
        """从文档列表构建向量库（增量添加，基于内容 hash 去重）"""
        if provider:
            self._provider = provider
        if model:
            self._model = model

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        ids = [_content_hash(t) for t in texts]

        self._collection = self._client.get_or_create_collection(self._collection_name)

        # 去重：检查已存在的 ID
        existing = set()
        if self._collection.count() > 0:
            existing_results = self._collection.get(ids=ids)
            existing = set(existing_results["ids"])

        new_texts, new_metas, new_ids = [], [], []
        dup_count = 0
        for text, meta, hid in zip(texts, metadatas, ids):
            if hid in existing:
                dup_count += 1
            else:
                new_texts.append(text)
                new_metas.append(meta)
                new_ids.append(hid)

        if dup_count:
            logger.info(f"跳过 {dup_count} 个重复片段（已存在于知识库中）")

        if not new_texts:
            logger.info("所有文档片段均已存在，无需重新嵌入")
            return self

        embeddings = self._compute_embeddings(new_texts, provider, model)

        self._collection.add(
            embeddings=embeddings,
            documents=new_texts,
            metadatas=new_metas if any(new_metas) else None,
            ids=new_ids,
        )
        return self

    def similarity_search(self, query: str, k: int = 4,
                          search_type: str = "similarity") -> List[Document]:
        """检索最相似的 k 个文档

        Args:
            query: 查询文本
            k: 返回数量
            search_type: "similarity" 或 "mmr"（最大边际相关性去重）
        """
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(self._collection_name)

        if self._collection.count() == 0:
            return []

        query_embedding = self._compute_query_embedding(query)
        n = min(k * 3 if search_type == "mmr" else k, self._collection.count())

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
        )

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append(Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            ))

        if search_type == "mmr" and len(docs) > k:
            docs = self._mmr_rerank(query_embedding, docs, k)

        return docs[:k]

    def _mmr_rerank(self, query_emb: List[float], docs: List[Document],
                    k: int, lambda_mult: float = 0.5) -> List[Document]:
        """MMR（最大边际相关性）重排，平衡相关性和多样性"""
        import numpy as np

        # 重新计算文档 embedding 用于多样性比较（用内容 hash 缓存太复杂，直接算）
        texts = [d.page_content for d in docs]
        doc_embeddings = self._compute_embeddings(texts)

        q = np.array(query_emb)
        doc_embs = np.array(doc_embeddings)

        # 归一化
        q = q / (np.linalg.norm(q) + 1e-10)
        doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)

        selected = []
        candidates = list(range(len(docs)))

        for _ in range(min(k, len(candidates))):
            if not candidates:
                break
            best_score = -float("inf")
            best_idx = candidates[0]
            for idx in candidates:
                relevance = float(np.dot(q, doc_embs[idx]))
                diversity = max(
                    (float(np.dot(doc_embs[idx], doc_embs[s])) for s in selected),
                    default=0.0,
                )
                score = lambda_mult * relevance - (1 - lambda_mult) * diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            candidates.remove(best_idx)

        return [docs[i] for i in selected]

    def delete_collection(self):
        """删除整个知识库集合"""
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = None
        except Exception:
            pass

    def count(self) -> int:
        """返回当前集合中的文档数量"""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(self._collection_name)
        return self._collection.count()

    def as_retriever(self, search_kwargs: dict = None):
        """返回 LangChain 兼容的 Retriever 对象"""
        kwargs = search_kwargs or {}
        k = kwargs.get("k", 4)
        search_type = kwargs.get("search_type", "similarity")

        from langchain_core.retrievers import BaseRetriever

        class _StoreRetriever(BaseRetriever):
            store: "SmartRAGVectorStore" = None
            k: int = 4
            search_type: str = "similarity"

            def _get_relevant_documents(self, query: str) -> List[Document]:
                # LCEL 中 RunnableParallel 会将完整 dict 传入
                if isinstance(query, dict):
                    query = query.get("question") or query.get("query") or str(query)
                return self.store.similarity_search(
                    query, k=self.k, search_type=self.search_type
                )

        return _StoreRetriever(store=self, k=k, search_type=search_type)


def create_vectorstore(
    documents: List[Document],
    provider: str = "dashscope",
    model: str = None,
    persist_dir: str = "./chroma_db",
    collection_name: str = "smartrag",
) -> SmartRAGVectorStore:
    """创建向量库"""
    vs = SmartRAGVectorStore(persist_dir=persist_dir, collection_name=collection_name)
    vs.from_documents(documents, provider=provider, model=model)
    return vs


def load_vectorstore(persist_dir: str = "./chroma_db",
                     collection_name: str = "smartrag") -> Optional[SmartRAGVectorStore]:
    """加载已有向量库"""
    if not os.path.exists(persist_dir):
        return None
    vs = SmartRAGVectorStore(persist_dir=persist_dir, collection_name=collection_name)
    vs._collection = vs._client.get_or_create_collection(collection_name)
    return vs
