from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .tools import ALL_TOOLS
from .memory import ConversationMemoryManager

class ChatbotAgent:
    """LangGraph 기반 챗본 에이전트"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        redis_url: str = "redis://localhost:6379"
    ):
        # llm init
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )

        self.memory_manager = ConversationMemoryManager(redis_url, self.llm)

        # LangGraph Agent 생성 (create_react_agent 사용)
        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=ALL_TOOLS,
        )

    async def chat(
        self,
        message: str,
        session_id: str,
        stream: bool = False
    ):
        """챗봇과 대화 (stream에 따라 generator 또는 str 반환)"""
        if stream:
            return self._chat_stream(message, session_id)
        else:
            return await self._chat_normal(message, session_id)

    async def _chat_stream(
        self,
        message: str,
        session_id: str
    ):
        """스트리밍 모드 (async generator)"""
        chat_history = self.memory_manager.get_langchain_messages(
            session_id,
            limit=10
        )

        # LangGraph는 messages 형식 사용
        messages = chat_history + [HumanMessage(content=message)]

        response = ""
        async for event in self.agent_executor.astream(
            {"messages": messages},
            stream_mode="values"
        ):
            if "messages" in event and len(event["messages"]) > 0:
                last_message = event["messages"][-1]
                if hasattr(last_message, "content"):
                    response = last_message.content
                    yield response

        # streaming 끝난 후 대화 기록 저장
        self.memory_manager.save_message(session_id, "human", message)
        self.memory_manager.save_message(session_id, "ai", response)
        self.memory_manager.summarize_if_needed(session_id)

    async def _chat_normal(
        self,
        message: str,
        session_id: str
    ) -> str:
        """일반 모드 (str 반환)"""
        chat_history = self.memory_manager.get_langchain_messages(
            session_id,
            limit=10
        )

        # LangGraph는 messages 형식 사용
        messages = chat_history + [HumanMessage(content=message)]

        result = await self.agent_executor.ainvoke({"messages": messages})

        # 마지막 메시지에서 응답 추출
        response = result["messages"][-1].content

        # 대화 기록 저장
        self.memory_manager.save_message(session_id, "human", message)
        self.memory_manager.save_message(session_id, "ai", response)
        self.memory_manager.summarize_if_needed(session_id)

        return response

    def clear_history(self, session_id: str):
        """대화 기록 삭제"""
        self.memory_manager.clear_session(session_id)

    def get_stats(self, session_id: str):
        """통계 조회"""
        return self.memory_manager.get_conversation_stats(session_id)