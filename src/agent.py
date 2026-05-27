"""Agent 决策层 - 自动判断是否需要检索"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .chain import get_llm


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


def build_agent_chain(rag_chain, llm=None):
    """组装带 Agent 决策的问答链

    工作流程：
    用户提问 → Router 判断 → retrieve: 走 RAG Chain / direct: 直接回答
    """
    llm = llm or get_llm()
    router = build_router(llm)
    direct_chain = build_direct_chain(llm)

    def agent_invoke(inputs):
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        decision = router.invoke({"question": question})
        if "retrieve" in decision.lower():
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
