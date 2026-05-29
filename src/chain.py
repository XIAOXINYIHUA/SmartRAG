"""RAG Chain 组装 — 多平台 LLM 支持，流式输出 + 可选重排序"""

import logging
from operator import itemgetter
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .config import get_provider_config, get_api_key

logger = logging.getLogger(__name__)


def get_llm(provider: str = "dashscope", model: str = None,
            streaming: bool = False) -> ChatOpenAI:
    """获取 LLM — 统一走 OpenAI 兼容协议，支持所有平台"""
    cfg = get_provider_config(provider)
    model = model or cfg["default_llm"]

    return ChatOpenAI(
        model=model,
        temperature=0,
        streaming=streaming,
        openai_api_key=get_api_key(provider),
        openai_api_base=cfg["base_url"],
    )


def format_docs(docs) -> str:
    """将检索到的文档格式化为上下文字符串"""
    return "\n\n---\n\n".join(
        f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
        for d in docs
    )


RAG_SYSTEM_PROMPT = """你是一个专业的文档问答助手。根据以下检索到的上下文回答用户问题。

规则：
1. 只根据提供的上下文回答，不要编造信息
2. 如果上下文没有相关信息，明确说"根据已加载的文档，我无法回答这个问题"
3. 回答要准确、简洁，引用具体来源

上下文：
{context}"""


def rerank_documents(query: str, docs: list, top_k: int = 4,
                     provider: str = None) -> list:
    """使用 Cross-Encoder 重排序检索结果

    优先使用 DashScope rerank API，回退到本地 sentence-transformers。
    如果都不可用，返回原始顺序。
    """
    if not docs or len(docs) <= top_k:
        return docs

    try:
        # 尝试 DashScope Rerank API
        import dashscope
        from dashscope import TextReRank
        import os

        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if api_key:
            passages = [d.page_content for d in docs]
            resp = TextReRank.call(
                model="gte-rerank",
                query=query,
                documents=passages,
                top_n=top_k,
                api_key=api_key,
            )
            if resp.status_code == 200:
                reranked = []
                for item in resp.output.results:
                    idx = item.index
                    docs[idx].metadata["rerank_score"] = item.relevance_score
                    reranked.append(docs[idx])
                logger.info(f"DashScope rerank: {len(docs)} → {len(reranked)} docs")
                return reranked
    except Exception as e:
        logger.debug(f"DashScope rerank failed: {e}")

    try:
        # 回退到本地 cross-encoder
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        pairs = [[query, d.page_content] for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        result = [d for _, d in ranked[:top_k]]
        logger.info(f"Local rerank: {len(docs)} → {len(result)} docs")
        return result
    except Exception as e:
        logger.debug(f"Local rerank failed: {e}")

    # 无可用重排器，返回原始顺序
    return docs[:top_k]


def build_rag_chain(vectorstore, llm: ChatOpenAI = None, top_k: int = 4,
                    use_rerank: bool = False, search_type: str = "similarity"):
    """构建 RAG 检索问答链

    Args:
        vectorstore: 向量存储
        llm: LLM 实例
        top_k: 检索返回的文档数量
        use_rerank: 是否启用重排序
        search_type: 检索类型 ("similarity" 或 "mmr")
    """
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k, "search_type": search_type}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    if use_rerank:
        # 带重排序的链：先检索多一些，再重排精简
        def retrieve_and_rerank(inputs):
            docs = retriever.invoke(inputs)
            return rerank_documents(inputs["question"], docs, top_k=min(top_k, len(docs)))

        rag_chain = (
            {
                "context": retrieve_and_rerank | format_docs,
                "chat_history": itemgetter("chat_history"),
                "question": itemgetter("question"),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        rag_chain = (
            {
                "context": retriever | format_docs,
                "chat_history": itemgetter("chat_history"),
                "question": itemgetter("question"),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
    return rag_chain
