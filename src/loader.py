"""文档加载模块 - 支持 PDF / TXT / Markdown / URL"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)


def load_file(file_path: str) -> List[Document]:
    """根据文件类型加载单个文件，返回 Document 列表"""
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": lambda p: TextLoader(p, encoding="utf-8"),
        ".md": UnstructuredMarkdownLoader,
    }
    if ext not in loaders:
        raise ValueError(f"不支持的文件格式: {ext}（支持 PDF / TXT / MD）")
    loader = loaders[ext](file_path)
    return loader.load()


def load_url(url: str) -> List[Document]:
    """加载网页内容"""
    loader = WebBaseLoader(url)
    return loader.load()


def load_directory(dir_path: str) -> List[Document]:
    """批量加载目录下的所有支持文件"""
    docs: List[Document] = []
    for fname in sorted(os.listdir(dir_path)):
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".pdf", ".txt", ".md"):
                docs.extend(load_file(fpath))
    return docs
