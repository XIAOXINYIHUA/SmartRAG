"""文本分割模块 - 将文档切分为语义块"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """按递归字符分割，保留文档元数据

    Args:
        documents: 文档列表
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块之间的重叠字符数
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(documents)
