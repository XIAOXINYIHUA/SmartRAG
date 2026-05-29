"""SmartRAG - Streamlit Web UI（多平台支持）"""

import hashlib
import logging
import os
import tempfile
from operator import itemgetter
from typing import List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("smartrag")

# ═══════════════════════════════════════════════
#  以下内容原本在 src/ 各模块中，现在全部内联
# ═══════════════════════════════════════════════

# ─── config.py ─────────────────────────────────

PROVIDERS: dict[str, dict] = {
    "dashscope": {
        "name": "阿里云 DashScope（通义千问）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "llm_models": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-plus-latest"],
        "embedding_models": ["text-embedding-v3", "text-embedding-v2", "text-embedding-v1"],
        "default_llm": "qwen-plus",
        "default_embedding": "text-embedding-v3",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "llm_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "embedding_models": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        "default_llm": "gpt-4o-mini",
        "default_embedding": "text-embedding-3-small",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "llm_models": ["deepseek-chat", "deepseek-reasoner"],
        "embedding_models": [],
        "default_llm": "deepseek-chat",
        "default_embedding": None,
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "siliconflow": {
        "name": "硅基流动 (SiliconFlow)",
        "base_url": "https://api.siliconflow.cn/v1",
        "llm_models": [
            "Qwen/Qwen3-235B-A22B",
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen2.5-72B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Pro/ZhipuAI/GLM-4-Plus",
        ],
        "embedding_models": [
            "BAAI/bge-large-zh-v1.5",
            "BAAI/bge-m3",
            "netease-youdao/bce-embedding-base_v1",
        ],
        "default_llm": "Qwen/Qwen3-235B-A22B",
        "default_embedding": "BAAI/bge-large-zh-v1.5",
        "api_key_env": "SILICONFLOW_API_KEY",
    },
    "zhipu": {
        "name": "智谱 AI (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "llm_models": ["glm-4-flash", "glm-4-plus", "glm-4-long"],
        "embedding_models": ["embedding-2"],
        "default_llm": "glm-4-flash",
        "default_embedding": "embedding-2",
        "api_key_env": "ZHIPU_API_KEY",
    },
    "ollama": {
        "name": "Ollama（本地模型）",
        "base_url": "http://localhost:11434/v1",
        "llm_models": ["qwen3", "llama3", "deepseek-r1", "mistral"],
        "embedding_models": ["nomic-embed-text", "bge-m3"],
        "default_llm": "qwen3",
        "default_embedding": "nomic-embed-text",
        "api_key_env": None,
    },
}

DEFAULT_PROVIDER = "dashscope"
EMBEDDING_FALLBACK_PROVIDER = "dashscope"


def get_provider_config(provider_id: str) -> dict:
    if provider_id not in PROVIDERS:
        raise ValueError(f"不支持的平台: {provider_id}，可用: {list(PROVIDERS)}")
    return PROVIDERS[provider_id]


def get_api_key(provider_id: str) -> str:
    cfg = get_provider_config(provider_id)
    env_var = cfg.get("api_key_env")
    if env_var is None:
        return "ollama"
    key = os.getenv(env_var, "")
    if not key:
        raise ValueError(f"未设置 {env_var}，请在 .env 文件中配置或在侧边栏输入该平台的 API Key")
    return key


def get_embedding_provider(provider_id: str) -> str:
    cfg = get_provider_config(provider_id)
    if cfg["embedding_models"]:
        return provider_id
    return EMBEDDING_FALLBACK_PROVIDER


def list_providers() -> list[tuple[str, str]]:
    return [(k, v["name"]) for k, v in PROVIDERS.items()]


# ─── loader.py ─────────────────────────────────

def load_file(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".md":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式: {ext}（支持 PDF / TXT / MD）")
    return loader.load()


def load_url(url: str) -> List[Document]:
    loader = WebBaseLoader(url)
    return loader.load()


def load_files_safe(file_paths: List[str]) -> Tuple[List[Document], List[dict]]:
    all_docs: List[Document] = []
    errors: List[dict] = []
    for fpath in file_paths:
        try:
            docs = load_file(fpath)
            all_docs.extend(docs)
            logger.info(f"加载成功: {fpath} ({len(docs)} pages)")
        except Exception as e:
            logger.error(f"加载失败: {fpath} - {e}")
            errors.append({"file": os.path.basename(fpath), "error": str(e)})
    return all_docs, errors


# ─── splitter.py ───────────────────────────────

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(documents)


# ─── embedding.py ──────────────────────────────

EMBEDDING_BATCH_SIZE = 10


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class SmartRAGVectorStore:
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
                from dashscope import TextEmbedding
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
            from dashscope import TextEmbedding
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
        if provider:
            self._provider = provider
        if model:
            self._model = model

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        ids = [_content_hash(t) for t in texts]

        self._collection = self._client.get_or_create_collection(self._collection_name)

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
        import numpy as np

        texts = [d.page_content for d in docs]
        doc_embeddings = self._compute_embeddings(texts)

        q = np.array(query_emb)
        doc_embs = np.array(doc_embeddings)

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
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = None
        except Exception:
            pass

    def count(self) -> int:
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(self._collection_name)
        return self._collection.count()

    def as_retriever(self, search_kwargs: dict = None):
        kwargs = search_kwargs or {}
        k = kwargs.get("k", 4)
        search_type = kwargs.get("search_type", "similarity")
        store = self

        from langchain_core.runnables import RunnableLambda

        def _retrieve(query):
            if isinstance(query, dict):
                query = query.get("question") or query.get("query") or str(query)
            return store.similarity_search(query, k=k, search_type=search_type)

        return RunnableLambda(_retrieve)


def create_vectorstore(
    documents: List[Document],
    provider: str = "dashscope",
    model: str = None,
    persist_dir: str = "./chroma_db",
    collection_name: str = "smartrag",
) -> SmartRAGVectorStore:
    vs = SmartRAGVectorStore(persist_dir=persist_dir, collection_name=collection_name)
    vs.from_documents(documents, provider=provider, model=model)
    return vs


def load_vectorstore(persist_dir: str = "./chroma_db",
                     collection_name: str = "smartrag") -> Optional[SmartRAGVectorStore]:
    if not os.path.exists(persist_dir):
        return None
    vs = SmartRAGVectorStore(persist_dir=persist_dir, collection_name=collection_name)
    vs._collection = vs._client.get_or_create_collection(collection_name)
    return vs


# ─── chain.py ──────────────────────────────────

RAG_SYSTEM_PROMPT = """你是一个专业的文档问答助手。根据以下检索到的上下文回答用户问题。

规则：
1. 只根据提供的上下文回答，不要编造信息
2. 如果上下文没有相关信息，明确说"根据已加载的文档，我无法回答这个问题"
3. 回答要准确、简洁，引用具体来源

上下文：
{context}"""


def get_llm(provider: str = "dashscope", model: str = None,
            streaming: bool = False) -> ChatOpenAI:
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
    return "\n\n---\n\n".join(
        f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
        for d in docs
    )


def rerank_documents(query: str, docs: list, top_k: int = 4,
                     provider: str = None) -> list:
    if not docs or len(docs) <= top_k:
        return docs
    try:
        import dashscope
        from dashscope import TextReRank
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

    return docs[:top_k]


def build_rag_chain(vectorstore, llm: ChatOpenAI = None, top_k: int = 4,
                    use_rerank: bool = False, search_type: str = "similarity"):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k, "search_type": search_type}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    if use_rerank:
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


# ─── agent.py ──────────────────────────────────

RETRIEVE_KEYWORDS = [
    "文档", "文件", "报告", "论文", "内容", "资料", "书", "章节",
    "根据", "依据", "基于", "提到", "上面", "前文",
    "document", "file", "report", "paper", "mentioned", "above",
]

ROUTER_PROMPT = """根据以下用户问题，判断是否需要从文档库中检索信息来回答。

如果问题需要检索文档才能准确回答，回复 "retrieve"
如果问题可以用通用知识直接回答，回复 "direct"

问题：{question}

你的判断（只回复 retrieve 或 direct）："""


def _keyword_route(question: str, has_docs: bool) -> str:
    if not has_docs:
        return "direct"
    q = question.lower()
    for kw in RETRIEVE_KEYWORDS:
        if kw in q:
            return "retrieve"
    return "retrieve"


def build_direct_chain(llm=None):
    llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的 AI 助手，请直接回答用户问题。"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


def build_router(llm=None):
    llm = llm or get_llm()
    router_prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    return router_prompt | llm | StrOutputParser()


def build_agent_chain(rag_chain, llm=None, use_llm_router: bool = False,
                      has_docs: bool = True):
    llm = llm or get_llm()
    router = build_router(llm) if use_llm_router else None
    direct_chain = build_direct_chain(llm)

    def agent_invoke(inputs):
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        if use_llm_router and router:
            decision = router.invoke({"question": question})
            use_rag = "retrieve" in decision.lower()
            logger.info(f"LLM route: {decision.strip()} → {'RAG' if use_rag else 'direct'}")
        else:
            route = _keyword_route(question, has_docs)
            use_rag = route == "retrieve"
            logger.info(f"Keyword route: {route}")

        if use_rag:
            return {
                "answer": rag_chain.invoke({
                    "question": question,
                    "chat_history": chat_history,
                }),
                "source": "rag",
            }
        else:
            return {
                "answer": direct_chain.invoke({"question": question}),
                "source": "direct",
            }

    return agent_invoke


# ═══════════════════════════════════════════════
#  Streamlit UI
# ═══════════════════════════════════════════════

MAX_HISTORY_TURNS = 5

# ─── 页面配置 ───
st.set_page_config(page_title="SmartRAG", page_icon="📚", layout="wide")
st.title("📚 SmartRAG - 智能文档问答助手")

# ─── 初始化 session state ───
defaults = {
    "messages": [],
    "vectorstore": None,
    "agent": None,
    "doc_count": 0,
    "provider": DEFAULT_PROVIDER,
    "llm_model": PROVIDERS[DEFAULT_PROVIDER]["default_llm"],
    "embedding_model": PROVIDERS[DEFAULT_PROVIDER]["default_embedding"],
    "kb_name": "default",
    "use_rerank": False,
    "search_type": "similarity",
    "top_k": 4,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def build_agent():
    llm = get_llm(st.session_state.provider, st.session_state.llm_model)
    rag_chain = build_rag_chain(
        st.session_state.vectorstore, llm,
        top_k=st.session_state.top_k,
        use_rerank=st.session_state.use_rerank,
        search_type=st.session_state.search_type,
    )
    st.session_state.agent = build_agent_chain(
        rag_chain, llm,
        has_docs=st.session_state.doc_count > 0,
    )


def get_truncated_history(messages: list, max_turns: int = MAX_HISTORY_TURNS) -> list:
    pairs = []
    i = len(messages) - 1
    while i >= 1 and len(pairs) < max_turns:
        if messages[i]["role"] == "assistant" and messages[i - 1]["role"] == "user":
            pairs.insert(0, (messages[i - 1], messages[i]))
            i -= 2
        else:
            i -= 1
    history = []
    for user_msg, asst_msg in pairs:
        history.append(("user", user_msg["content"]))
        history.append(("assistant", asst_msg["content"]))
    return history


# ─── 侧边栏 ───
with st.sidebar:
    st.header("⚙️ 平台配置")

    provider_options = {v["name"]: k for k, v in PROVIDERS.items()}
    current_name = PROVIDERS[st.session_state.provider]["name"]
    selected_name = st.selectbox(
        "选择平台",
        list(provider_options.keys()),
        index=list(provider_options.values()).index(st.session_state.provider),
    )
    selected_provider = provider_options[selected_name]

    if selected_provider != st.session_state.provider:
        st.session_state.provider = selected_provider
        cfg = get_provider_config(selected_provider)
        st.session_state.llm_model = cfg["default_llm"]
        st.session_state.embedding_model = cfg.get("default_embedding")
        try:
            if st.session_state.vectorstore:
                build_agent()
        except ValueError as e:
            st.error(str(e))
        st.rerun()

    cfg = get_provider_config(st.session_state.provider)

    # API Key 输入
    key_env = cfg.get("api_key_env")
    if key_env:
        current_key = os.getenv(key_env, "")
        key_label = f"🔑 {key_env}"
        input_key = st.text_input(
            key_label,
            value=current_key if current_key else "",
            type="password",
            placeholder=f"输入你的 {key_env}（留空则用 .env 中的值）",
        )
        if input_key:
            os.environ[key_env] = input_key
        if not input_key and not current_key:
            st.warning(f"请设置 {key_env}")
    else:
        st.info("Ollama 无需 API Key")

    # LLM 模型选择
    st.divider()
    st.header("🤖 问答模型")
    if cfg["llm_models"]:
        llm_idx = (
            cfg["llm_models"].index(st.session_state.llm_model)
            if st.session_state.llm_model in cfg["llm_models"]
            else 0
        )
        selected_llm = st.selectbox(
            "LLM 模型",
            cfg["llm_models"],
            index=llm_idx,
            key="llm_model_select",
        )
        if selected_llm != st.session_state.llm_model:
            st.session_state.llm_model = selected_llm
            try:
                if st.session_state.vectorstore:
                    build_agent()
            except ValueError as e:
                st.error(str(e))
    else:
        st.text_input("LLM 模型", value=st.session_state.llm_model, key="llm_custom")

    # Embedding 模型选择
    emb_provider = get_provider_config(st.session_state.provider)
    if not emb_provider["embedding_models"]:
        emb_cfg = get_provider_config(EMBEDDING_FALLBACK_PROVIDER)
        st.caption(f"⚠️ {emb_provider['name']} 不支持 Embedding，文档向量化使用 {emb_cfg['name']}")
    else:
        emb_cfg = emb_provider

    if emb_cfg.get("embedding_models"):
        emb_model = st.session_state.embedding_model or emb_cfg.get("default_embedding")
        emb_idx = (
            emb_cfg["embedding_models"].index(emb_model)
            if emb_model in emb_cfg["embedding_models"]
            else 0
        )
        selected_emb = st.selectbox(
            "Embedding 模型",
            emb_cfg["embedding_models"],
            index=emb_idx,
            key="emb_model_select",
        )
        st.session_state.embedding_model = selected_emb

    # ─── 检索配置 ───
    st.divider()
    st.header("🔍 检索设置")

    st.session_state.top_k = st.slider(
        "检索文档数量 (top-k)", min_value=1, max_value=20, value=st.session_state.top_k
    )

    st.session_state.search_type = st.selectbox(
        "检索方式",
        ["similarity", "mmr"],
        index=0 if st.session_state.search_type == "similarity" else 1,
        format_func=lambda x: "相似度检索" if x == "similarity" else "MMR（多样性检索）",
    )

    st.session_state.use_rerank = st.checkbox(
        "启用重排序 (Reranker)", value=st.session_state.use_rerank,
        help="使用 Cross-Encoder 对检索结果重排，提升相关性（首次使用需下载模型）",
    )

    # ─── 分块配置 ───
    st.divider()
    st.header("✂️ 分块设置")
    chunk_size = st.slider("块大小 (chunk_size)", 200, 4000, 1000, step=100)
    chunk_overlap = st.slider("块重叠 (chunk_overlap)", 0, 500, 200, step=50)

    st.divider()

    # ─── 文档管理 ───
    st.header("📄 文档管理")

    kb_name = st.text_input("知识库名称", value=st.session_state.kb_name,
                             help="不同名称 = 独立知识库，切换后需重新加载文档")
    if kb_name != st.session_state.kb_name:
        st.session_state.kb_name = kb_name
        try:
            vs = load_vectorstore(collection_name=kb_name)
            if vs and vs.count() > 0:
                st.session_state.vectorstore = vs
                st.session_state.doc_count = vs.count()
                build_agent()
                st.success(f"已切换到知识库「{kb_name}」（{vs.count()} 个片段）")
            else:
                st.session_state.vectorstore = None
                st.session_state.doc_count = 0
                st.session_state.agent = None
        except Exception as e:
            st.warning(f"切换知识库失败: {e}")
        st.rerun()

    if st.session_state.doc_count > 0:
        st.warning("⚠️ 修改 Embedding 设置后需要重新加载文档才能生效。")

    uploaded_files = st.file_uploader(
        "上传文档（支持 PDF / TXT / MD）",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    url_input = st.text_input("或输入网页 URL")

    if st.button("📥 加载文档", use_container_width=True):
        try:
            get_api_key(st.session_state.provider)
        except ValueError as e:
            st.error(str(e))
        else:
            tmp_paths = []
            for uploaded_file in uploaded_files:
                ext = uploaded_file.name.split('.')[-1] if '.' in uploaded_file.name else 'txt'
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_paths.append(tmp.name)

            with st.spinner("加载中..."):
                all_docs, errors = load_files_safe(tmp_paths)

                for p in tmp_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

                if url_input.strip():
                    try:
                        url_docs = load_url(url_input.strip())
                        all_docs.extend(url_docs)
                    except Exception as e:
                        errors.append({"file": url_input.strip(), "error": str(e)})

            for err in errors:
                st.error(f"❌ {err['file']}: {err['error']}")

            if all_docs:
                with st.spinner("正在分块和向量化..."):
                    chunks = split_documents(all_docs, chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
                    st.session_state.vectorstore = create_vectorstore(
                        chunks,
                        provider=st.session_state.provider,
                        model=st.session_state.embedding_model,
                        collection_name=st.session_state.kb_name,
                    )
                    st.session_state.doc_count = st.session_state.vectorstore.count()

                    build_agent()

                st.success(f"✅ 已加载 {len(all_docs)} 个文档 → {len(chunks)} 个片段")
                st.rerun()
            elif not errors:
                st.warning("请先上传文档或输入 URL")

    # 状态显示
    st.divider()
    if st.session_state.doc_count > 0:
        st.info(f"📚 知识库「{st.session_state.kb_name}」: {st.session_state.doc_count} 个片段")
    else:
        st.info("请先上传文档或输入 URL")

    # Agent 开关
    use_agent = st.checkbox("启用 Agent 自动决策", value=True)

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.doc_count > 0:
        if st.button("🗑️ 清空知识库", use_container_width=True):
            if st.session_state.vectorstore:
                st.session_state.vectorstore.delete_collection()
            st.session_state.vectorstore = None
            st.session_state.doc_count = 0
            st.session_state.agent = None
            st.session_state.messages = []
            st.success("知识库已清空")
            st.rerun()

# ─── 主区域：对话 ───
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source") == "rag":
            st.caption("📖 回答基于文档检索")
        elif msg.get("source") == "direct":
            st.caption("💡 回答基于通用知识")

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.agent is None:
            answer = "请先在左侧上传文档或输入 URL，加载到知识库后再提问。"
            source = None
            st.markdown(answer)
        else:
            chat_history = get_truncated_history(st.session_state.messages)

            if use_agent:
                route = _keyword_route(
                    prompt,
                    has_docs=st.session_state.doc_count > 0,
                )
                source = "rag" if route == "retrieve" else "direct"

                streaming_llm = get_llm(
                    st.session_state.provider,
                    st.session_state.llm_model,
                    streaming=True,
                )

                if source == "rag":
                    streaming_chain = build_rag_chain(
                        st.session_state.vectorstore, streaming_llm,
                        top_k=st.session_state.top_k,
                        use_rerank=st.session_state.use_rerank,
                        search_type=st.session_state.search_type,
                    )
                else:
                    prompt_tpl = ChatPromptTemplate.from_messages([
                        ("system", "你是一个有帮助的 AI 助手，请直接回答用户问题。"),
                        ("human", "{question}"),
                    ])
                    streaming_chain = prompt_tpl | streaming_llm | StrOutputParser()

                answer = st.write_stream(
                    streaming_chain.stream({
                        "question": prompt,
                        "chat_history": chat_history,
                    })
                )

                if source == "rag":
                    st.caption("📖 回答基于文档检索")
                elif source == "direct":
                    st.caption("💡 回答基于通用知识")
            else:
                with st.spinner("思考中..."):
                    rag_chain = build_rag_chain(
                        st.session_state.vectorstore,
                        get_llm(st.session_state.provider, st.session_state.llm_model),
                        top_k=st.session_state.top_k,
                        use_rerank=st.session_state.use_rerank,
                        search_type=st.session_state.search_type,
                    )
                    answer = rag_chain.invoke({
                        "question": prompt,
                        "chat_history": chat_history,
                    })
                    source = "rag"
                    st.markdown(answer)
                    st.caption("📖 回答基于文档检索")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "source": source,
    })
