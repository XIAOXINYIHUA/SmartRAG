"""SmartRAG - Streamlit Web UI"""

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

# ─── 页面配置 ───
st.set_page_config(page_title="SmartRAG", page_icon="📚", layout="wide")
st.title("📚 SmartRAG - 智能文档问答助手")

# ─── 初始化 session state ───
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# ─── 侧边栏：文档管理 ───
with st.sidebar:
    st.header("📄 文档管理")

    # 文件上传
    uploaded_files = st.file_uploader(
        "上传文档（支持 PDF / TXT / MD）",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    # URL 输入
    url_input = st.text_input("或输入网页 URL")

    if st.button("📥 加载文档", use_container_width=True):
        all_docs = []
        with st.spinner("加载中..."):
            # 处理上传文件
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                all_docs.extend(load_file(tmp_path))
                os.unlink(tmp_path)

            # 处理 URL
            if url_input.strip():
                try:
                    all_docs.extend(load_url(url_input.strip()))
                except Exception as e:
                    st.error(f"URL 加载失败: {e}")

        if all_docs:
            # 分割文档
            chunks = split_documents(all_docs)
            st.session_state.vectorstore = create_vectorstore(chunks)
            st.session_state.doc_count = len(chunks)

            # 重建 agent
            llm = get_llm()
            rag_chain = build_rag_chain(st.session_state.vectorstore, llm)
            st.session_state.agent = build_agent_chain(rag_chain, llm)

            st.success(f"已加载 {len(all_docs)} 个文档，分割为 {len(chunks)} 个片段")

    # 显示状态
    st.divider()
    if st.session_state.doc_count > 0:
        st.info(f"知识库包含 {st.session_state.doc_count} 个片段")
    else:
        st.info("请先上传文档或输入 URL")

    # 模型设置
    st.divider()
    st.header("⚙️ 设置")
    st.text_input("模型名称", value="gpt-4o-mini", key="model_name", disabled=True)
    use_agent = st.checkbox("启用 Agent 自动决策", value=True)

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─── 主区域：对话 ───
# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source") == "rag":
            st.caption("📖 回答基于文档检索")
        elif msg.get("source") == "direct":
            st.caption("💡 回答基于通用知识")

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
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
