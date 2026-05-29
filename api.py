"""SmartRAG REST API — FastAPI 封装"""

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

load_dotenv()

from src.loader import load_file, load_url, load_files_safe
from src.splitter import split_documents
from src.embedding import create_vectorstore, load_vectorstore
from src.chain import build_rag_chain, get_llm
from src.config import get_api_key, list_providers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartrag-api")

app = FastAPI(
    title="SmartRAG API",
    description="智能文档问答助手 REST API",
    version="0.2.0",
)


# ─── 数据模型 ───

class ChatRequest(BaseModel):
    question: str
    provider: str = Field(default="dashscope", description="LLM 平台")
    model: Optional[str] = Field(default=None, description="LLM 模型名")
    kb_name: str = Field(default="smartrag", description="知识库名称")
    top_k: int = Field(default=4, ge=1, le=20, description="检索文档数量")
    search_type: str = Field(default="similarity", description="检索方式: similarity / mmr")
    use_rerank: bool = Field(default=False, description="是否启用重排序")
    max_history: int = Field(default=10, ge=0, description="最大历史轮数")


class ChatResponse(BaseModel):
    answer: str
    source: str = Field(description="rag 或 direct")
    provider: str
    model: str


class LoadRequest(BaseModel):
    provider: str = Field(default="dashscope")
    embedding_model: Optional[str] = None
    kb_name: str = Field(default="smartrag")
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    urls: List[str] = Field(default_factory=list)


class LoadResponse(BaseModel):
    message: str
    doc_count: int
    chunk_count: int
    errors: List[dict] = []


class KbInfo(BaseModel):
    name: str
    doc_count: int


# ─── 全局状态 ───
_vectorstores: dict = {}


def _get_vs(kb_name: str = "smartrag"):
    """获取或加载向量库"""
    if kb_name not in _vectorstores:
        vs = load_vectorstore(collection_name=kb_name)
        if vs:
            _vectorstores[kb_name] = vs
        else:
            raise HTTPException(404, f"知识库「{kb_name}」不存在，请先上传文档")
    return _vectorstores[kb_name]


# ─── API 路由 ───

@app.get("/", tags=["info"])
def root():
    return {"name": "SmartRAG API", "version": "0.2.0"}


@app.get("/providers", tags=["info"])
def get_providers():
    """获取支持的平台列表"""
    return [{"id": k, "name": v} for k, v in list_providers()]


@app.post("/chat", response_model=ChatResponse, tags=["qa"])
def chat(req: ChatRequest):
    """问答接口"""
    vs = _get_vs(req.kb_name)
    llm = get_llm(req.provider, req.model)
    rag_chain = build_rag_chain(
        vs, llm,
        top_k=req.top_k,
        use_rerank=req.use_rerank,
        search_type=req.search_type,
    )

    # 简单路由：有文档就走 RAG
    answer = rag_chain.invoke({
        "question": req.question,
        "chat_history": [],  # API 暂不维护会话状态
    })

    return ChatResponse(
        answer=answer,
        source="rag",
        provider=req.provider,
        model=req.model or "default",
    )


@app.post("/load", response_model=LoadResponse, tags=["documents"])
def load_documents(req: LoadRequest):
    """加载 URL 文档到知识库"""
    if not req.urls:
        raise HTTPException(400, "请提供至少一个 URL")

    all_docs = []
    errors = []
    for url in req.urls:
        try:
            docs = load_url(url)
            all_docs.extend(docs)
        except Exception as e:
            errors.append({"url": url, "error": str(e)})

    if not all_docs:
        raise HTTPException(400, "所有 URL 加载失败")

    chunks = split_documents(all_docs, chunk_size=req.chunk_size,
                              chunk_overlap=req.chunk_overlap)
    vs = create_vectorstore(
        chunks,
        provider=req.provider,
        model=req.embedding_model,
        collection_name=req.kb_name,
    )
    _vectorstores[req.kb_name] = vs

    return LoadResponse(
        message=f"成功加载 {len(all_docs)} 个文档",
        doc_count=len(all_docs),
        chunk_count=len(chunks),
        errors=errors,
    )


@app.post("/upload", response_model=LoadResponse, tags=["documents"])
async def upload_files(
    files: List[UploadFile] = File(...),
    provider: str = "dashscope",
    embedding_model: Optional[str] = None,
    kb_name: str = "smartrag",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """上传文件到知识库"""
    import tempfile

    tmp_paths = []
    for f in files:
        suffix = os.path.splitext(f.filename)[1] or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await f.read()
            tmp.write(content)
            tmp_paths.append(tmp.name)

    all_docs, errors = load_files_safe(tmp_paths)

    for p in tmp_paths:
        try:
            os.unlink(p)
        except OSError:
            pass

    if not all_docs:
        raise HTTPException(400, "所有文件加载失败")

    chunks = split_documents(all_docs, chunk_size=chunk_size,
                              chunk_overlap=chunk_overlap)
    vs = create_vectorstore(
        chunks,
        provider=provider,
        model=embedding_model,
        collection_name=kb_name,
    )
    _vectorstores[kb_name] = vs

    return LoadResponse(
        message=f"成功加载 {len(all_docs)} 个文件",
        doc_count=len(all_docs),
        chunk_count=len(chunks),
        errors=errors,
    )


@app.get("/kb", response_model=List[KbInfo], tags=["knowledge-base"])
def list_knowledge_bases():
    """列出所有知识库"""
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    collections = client.list_collections()
    return [
        KbInfo(name=c.name, doc_count=c.count())
        for c in collections
    ]


@app.delete("/kb/{kb_name}", tags=["knowledge-base"])
def delete_knowledge_base(kb_name: str):
    """删除知识库"""
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection(kb_name)
        _vectorstores.pop(kb_name, None)
        return {"message": f"知识库「{kb_name}」已删除"}
    except Exception as e:
        raise HTTPException(404, f"知识库不存在: {e}")
