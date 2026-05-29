"""SmartRAG - Streamlit Web UI（多平台支持）"""

import logging
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("smartrag")

from src.loader import load_file, load_url, load_files_safe
from src.splitter import split_documents
from src.embedding import SmartRAGVectorStore, create_vectorstore, load_vectorstore
from src.chain import build_rag_chain, get_llm
from src.agent import build_agent_chain
from src.config import (
    PROVIDERS,
    DEFAULT_PROVIDER,
    get_provider_config,
    get_api_key,
    list_providers,
)

# ─── 常量 ───
MAX_HISTORY_TURNS = 5  # 最多保留最近 N 轮对话历史

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
    "kb_name": "default",  # 当前知识库名称
    "use_rerank": False,
    "search_type": "similarity",
    "top_k": 4,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def build_agent():
    """根据当前 provider/model 重建 Agent"""
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
    """截断对话历史，只保留最近 max_turns 轮"""
    # 按 user/assistant 配对算"轮"
    pairs = []
    i = len(messages) - 1
    while i >= 1 and len(pairs) < max_turns:
        if messages[i]["role"] == "assistant" and messages[i - 1]["role"] == "user":
            pairs.insert(0, (messages[i - 1], messages[i]))
            i -= 2
        else:
            i -= 1

    # 转换为 LangChain 格式
    history = []
    for user_msg, asst_msg in pairs:
        history.append(("user", user_msg["content"]))
        history.append(("assistant", asst_msg["content"]))
    return history


# ─── 侧边栏 ───
with st.sidebar:
    st.header("⚙️ 平台配置")

    # 平台选择
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
        from src.config import EMBEDDING_FALLBACK_PROVIDER
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

    # 知识库名称
    kb_name = st.text_input("知识库名称", value=st.session_state.kb_name,
                             help="不同名称 = 独立知识库，切换后需重新加载文档")
    if kb_name != st.session_state.kb_name:
        st.session_state.kb_name = kb_name
        # 切换知识库：尝试加载已有 collection
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
            # 保存上传文件到临时目录
            tmp_paths = []
            for uploaded_file in uploaded_files:
                ext = uploaded_file.name.split('.')[-1] if '.' in uploaded_file.name else 'txt'
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_paths.append(tmp.name)

            # 加载文档（容错：单文件失败不影响其他）
            with st.spinner("加载中..."):
                all_docs, errors = load_files_safe(tmp_paths)

                # 清理临时文件
                for p in tmp_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

                # 加载 URL
                if url_input.strip():
                    try:
                        url_docs = load_url(url_input.strip())
                        all_docs.extend(url_docs)
                    except Exception as e:
                        errors.append({"file": url_input.strip(), "error": str(e)})

            # 显示加载错误
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

    # 清空当前知识库
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
            # 截断历史
            chat_history = get_truncated_history(st.session_state.messages)

            if use_agent:
                # 流式输出：先判断路由，再流式生成
                from src.agent import _keyword_route
                route = _keyword_route(
                    prompt,
                    has_docs=st.session_state.doc_count > 0,
                )
                source = "rag" if route == "retrieve" else "direct"

                # 重新构建流式 LLM
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
                    from langchain_core.prompts import ChatPromptTemplate
                    from langchain_core.output_parsers import StrOutputParser
                    prompt_tpl = ChatPromptTemplate.from_messages([
                        ("system", "你是一个有帮助的 AI 助手，请直接回答用户问题。"),
                        ("human", "{question}"),
                    ])
                    streaming_chain = prompt_tpl | streaming_llm | StrOutputParser()

                # 流式输出
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
                # 非 Agent 模式：直接走 RAG
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
