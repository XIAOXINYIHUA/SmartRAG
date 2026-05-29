"""SmartRAG 单元测试"""

import os
import sys
import tempfile
import pytest

# 确保 src 可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── splitter 测试 ───

class TestSplitter:
    def test_split_basic(self):
        from langchain_core.documents import Document
        from src.splitter import split_documents

        docs = [Document(page_content="A" * 3000, metadata={"source": "test.txt"})]
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk.page_content) <= 1000

    def test_split_preserves_metadata(self):
        from langchain_core.documents import Document
        from src.splitter import split_documents

        docs = [Document(page_content="Hello World. " * 200, metadata={"source": "test.md"})]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.md"

    def test_split_small_doc_no_split(self):
        from langchain_core.documents import Document
        from src.splitter import split_documents

        docs = [Document(page_content="Short document.", metadata={})]
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1


# ─── loader 测试 ───

class TestLoader:
    def test_load_txt(self):
        from src.loader import load_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                          encoding="utf-8") as f:
            f.write("Hello SmartRAG\n测试文档")
            f.flush()
            path = f.name

        try:
            docs = load_file(path)
            assert len(docs) == 1
            assert "SmartRAG" in docs[0].page_content
        finally:
            os.unlink(path)

    def test_load_md(self):
        from src.loader import load_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False,
                                          encoding="utf-8") as f:
            f.write("# Title\n\nContent here")
            f.flush()
            path = f.name

        try:
            docs = load_file(path)
            assert len(docs) >= 1
        finally:
            os.unlink(path)

    def test_load_unsupported_format(self):
        from src.loader import load_file

        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_file("/tmp/test.xyz")

    def test_load_files_safe_partial_failure(self):
        from src.loader import load_files_safe

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                          encoding="utf-8") as f:
            f.write("Good file")
            f.flush()
            good_path = f.name

        try:
            docs, errors = load_files_safe([good_path, "/nonexistent/file.txt"])
            assert len(docs) == 1
            assert len(errors) == 1
            assert "file.txt" in errors[0]["file"]
        finally:
            os.unlink(good_path)


# ─── config 测试 ───

class TestConfig:
    def test_list_providers(self):
        from src.config import list_providers
        providers = list_providers()
        assert len(providers) == 6
        ids = [p[0] for p in providers]
        assert "dashscope" in ids
        assert "openai" in ids

    def test_get_provider_config(self):
        from src.config import get_provider_config
        cfg = get_provider_config("dashscope")
        assert "base_url" in cfg
        assert "llm_models" in cfg

    def test_invalid_provider(self):
        from src.config import get_provider_config
        with pytest.raises(ValueError, match="不支持的平台"):
            get_provider_config("nonexistent")

    def test_embedding_fallback(self):
        from src.config import get_embedding_provider
        # DeepSeek 没有 embedding，应回退
        assert get_embedding_provider("deepseek") == "dashscope"
        # DashScope 自己有 embedding
        assert get_embedding_provider("dashscope") == "dashscope"


# ─── embedding 测试 ───

class TestEmbedding:
    def test_content_hash(self):
        from src.embedding import _content_hash
        h1 = _content_hash("hello")
        h2 = _content_hash("hello")
        h3 = _content_hash("world")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 32  # MD5 hex

    def test_vectorstore_init(self):
        from src.embedding import SmartRAGVectorStore
        tmpdir = tempfile.mkdtemp()
        try:
            vs = SmartRAGVectorStore(persist_dir=tmpdir, collection_name="test")
            assert vs.count() == 0
            # 释放 chromadb 文件锁
            del vs
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ─── agent 路由测试 ───

class TestAgent:
    def test_keyword_route_no_docs(self):
        from src.agent import _keyword_route
        assert _keyword_route("你好", has_docs=False) == "direct"

    def test_keyword_route_with_doc_keyword(self):
        from src.agent import _keyword_route
        assert _keyword_route("根据文档回答", has_docs=True) == "retrieve"

    def test_keyword_route_default_retrieve(self):
        from src.agent import _keyword_route
        # 有文档时，无明确关键词默认 retrieve
        assert _keyword_route("什么是AI", has_docs=True) == "retrieve"
