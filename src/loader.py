"""文档加载模块 - 支持 PDF / TXT / Markdown / URL"""

import logging
import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)

logger = logging.getLogger(__name__)


def load_file(file_path: str) -> List[Document]:
    """根据文件类型加载单个文件，返回 Document 列表"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".md":
        # 用 TextLoader 替代 UnstructuredMarkdownLoader，减少 unstructured 依赖
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式: {ext}（支持 PDF / TXT / MD）")
    return loader.load()


def load_url(url: str) -> List[Document]:
    """加载网页内容"""
    loader = WebBaseLoader(url)
    return loader.load()


def load_files_safe(file_paths: List[str]) -> Tuple[List[Document], List[dict]]:
    """批量加载文件，单个失败不影响其他文件

    Returns:
        (成功加载的文档列表, 失败文件信息列表 [{"file": ..., "error": ...}])
    """
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


def load_directory(dir_path: str) -> List[Document]:
    """批量加载目录下的所有支持文件"""
    docs: List[Document] = []
    for fname in sorted(os.listdir(dir_path)):
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".pdf", ".txt", ".md"):
                try:
                    docs.extend(load_file(fpath))
                except Exception as e:
                    logger.warning(f"跳过 {fname}: {e}")
    return docs
