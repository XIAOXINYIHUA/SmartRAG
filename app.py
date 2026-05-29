"""SmartRAG - Streamlit Web UI（多平台支持）"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.loader import load_file, load_url
from src.splitter import split_documents
from src.embedding import create_vectorstore, load_vectorstore, get_embeddings
from src.chain import build_rag_chain, get_llm
from src.agent import build_agent_chain
from src.config import (
    PROVIDERS,
    DEFAULT_PROVIDER,
    get_provider_config,
    get_api_key,
    get_embedding_provider,
    list_providers,
)

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
    "emb_provider_used": None,  # 创建向量库时实际用的 embedding 平台
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def build_agent():
    """根据当前 provider/model 重建 Agent"""
    llm = get_llm(st.session_state.provider, st.session_state.llm_model)
    rag_chain = build_rag_chain(st.session_state.vectorstore, llm)
    st.session_state.agent = build_agent_chain(rag_chain, llm)


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
    emb_provider = get_embedding_provider(st.session_state.provider)
    if emb_provider != st.session_state.provider:
        st.caption(f"⚠️ {cfg['name']} 不支持 Embedding，文档向量化使用 {PROVIDERS[emb_provider]['name']}")

    emb_cfg = get_provider_config(emb_provider)
    if emb_cfg["embedding_models"]:
        emb_model = st.session_state.embedding_model or emb_cfg["default_embedding"]
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

    st.divider()

    # ─── 文档管理 ───
    st.header("📄 文档管理")

    # 如果切换了 embedding 平台，提示
    if (
        st.session_state.emb_provider_used
        and emb_provider != st.session_state.emb_provider_used
        and st.session_state.doc_count > 0
    ):
        st.warning("⚠️ Embedding 平台已变更，已有检索可能不准。建议重新加载文档。")

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
            all_docs = []
            with st.spinner("加载中..."):
                for uploaded_file in uploaded_files:
                    ext = uploaded_file.name.split('.')[-1] if '.' in uploaded_file.name else 'txt'
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    all_docs.extend(load_file(tmp_path))
                    os.unlink(tmp_path)

                if url_input.strip():
                    try:
                        all_docs.extend(load_url(url_input.strip()))
                    except Exception as e:
                        st.error(f"URL 加载失败: {e}")

            if all_docs:
                chunks = split_documents(all_docs)
                embeddings = get_embeddings(
                    st.session_state.provider,
                    st.session_state.embedding_model,
                )
                st.session_state.vectorstore = create_vectorstore(chunks, embeddings)
                st.session_state.doc_count = len(chunks)
                st.session_state.emb_provider_used = emb_provider

                build_agent()
                st.success(f"已加载 {len(all_docs)} 个文档，分割为 {len(chunks)} 个片段")
                st.rerun()

    # 状态显示
    st.divider()
    if st.session_state.doc_count > 0:
        st.info(f"📚 知识库: {st.session_state.doc_count} 个片段")
    else:
        st.info("请先上传文档或输入 URL")

    # Agent 开关
    use_agent = st.checkbox("启用 Agent 自动决策", value=True)

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
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
            with st.spinner("思考中..."):
                chat_history = [
                    (m["role"], m["content"])
                    for m in st.session_state.messages[:-1]
                    if m["role"] in ("user", "assistant")
                ]
                result = st.session_state.agent({
                    "question": prompt,
                    "chat_history": chat_history,
                })
                answer = result["answer"]
                source = result["source"]
                st.markdown(answer)

                if source == "rag":
                    st.caption("📖 回答基于文档检索")
                elif source == "direct":
                    st.caption("💡 回答基于通用知识")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "source": source,
    })
