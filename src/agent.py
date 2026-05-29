"""Agent 决策层 - 轻量路由 + 流式支持"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .chain import get_llm

logger = logging.getLogger(__name__)

# 检索关键词（轻量路由用）
RETRIEVE_KEYWORDS = [
    "文档", "文件", "报告", "论文", "内容", "资料", "书", "章节",
    "根据", "依据", "基于", "提到", "上面", "前文",
    "document", "file", "report", "paper", "mentioned", "above",
]


def _keyword_route(question: str, has_docs: bool) -> str:
    """基于关键词的轻量路由，无需额外 LLM 调用"""
    if not has_docs:
        return "direct"
    q = question.lower()
    for kw in RETRIEVE_KEYWORDS:
        if kw in q:
            return "retrieve"
    # 默认走检索（宁可多查，不漏答案）
    return "retrieve"


ROUTER_PROMPT = """根据以下用户问题，判断是否需要从文档库中检索信息来回答。

如果问题需要检索文档才能准确回答，回复 "retrieve"
如果问题可以用通用知识直接回答，回复 "direct"

问题：{question}

你的判断（只回复 retrieve 或 direct）："""


def build_direct_chain(llm=None):
    """构建直接回答链（不需要检索时使用）"""
    llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的 AI 助手，请直接回答用户问题。"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


def build_router(llm=None):
    """构建路由器，决定走检索还是直接回答"""
    llm = llm or get_llm()
    router_prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    return router_prompt | llm | StrOutputParser()


def build_agent_chain(rag_chain, llm=None, use_llm_router: bool = False,
                      has_docs: bool = True):
    """组装带 Agent 决策的问答链

    工作流程：
    用户提问 → 路由判断 → retrieve: 走 RAG Chain / direct: 直接回答

    Args:
        rag_chain: RAG 检索链
        llm: LLM 实例
        use_llm_router: True 用 LLM 判断路由（更准但多一次调用），
                        False 用关键词路由（即时无延迟）
        has_docs: 知识库是否有文档
    """
    llm = llm or get_llm()
    router = build_router(llm) if use_llm_router else None
    direct_chain = build_direct_chain(llm)

    def agent_invoke(inputs):
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        # 路由决策
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
