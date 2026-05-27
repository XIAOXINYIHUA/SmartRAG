"""RAG Chain 组装 - 核心问答链"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma


def get_llm(model: str = None):
    """获取 LLM，默认使用 gpt-4o-mini"""
    model = model or "gpt-4o-mini"
    return ChatOpenAI(model=model, temperature=0)


def format_docs(docs):
    """将检索到的文档格式化为字符串"""
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


def build_rag_chain(vectorstore: Chroma, llm=None):
    """构建 RAG 检索问答链"""
    llm = llm or get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "chat_history": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
