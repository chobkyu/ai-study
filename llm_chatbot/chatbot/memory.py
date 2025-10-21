from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import redis
import json
from typing import List, Optional
from datetime import datetime, timedelta

class ConversationMemoryManager:
    """대화 메모리 관리자"""

    def __init__(self, redis_url: str = "redis://localhost:6379", llm: Optional[ChatOpenAI] = None):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature= 0)

    def get_session_key(self, session_id: str) -> str:
        """세션 키 생성"""
        return f"chat:session:{session_id}"

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str
    ):
        """메세지 저장"""
        key = self.get_session_key(session_id)

        message = {
            "role": role,
            "content" : content,
            "timestamp": datetime.now().isoformat()
        }

        # add to redis
        self.redis_client.rpush(key, json.dumps(message))

        self.redis_client.expire(key, 86400)

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[dict]:
        """저장된 메시지 가져오기"""
        key = self.get_session_key(session_id)

        messages = self.redis_client.lrange(key, 0 ,-1)

        parsed = [json.loads(msg) for msg in messages]

        if limit:
            parsed = parsed[-limit:]

        return parsed
    
    def get_langchain_messages(self, session_id: str, limit: int = 10):
        """LangChain 메시지 형식으로 변환"""
        messages = self.get_messages(session_id, limit=limit)

        langchain_messages = []
        for msg in messages:
            if msg["role"] == "human":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                langchain_messages.append(AIMessage(content=msg["content"]))
            
        return langchain_messages
    
    def clear_session(self, session_id: str):
        """세션 삭제"""
        key = self.get_session_key(session_id)
        self.redis_client.delete(key)

    def summarize_if_needed(self, session_id: str, max_messages: int = 20):
        """메세지가 많으면 요약"""
        messages = self.get_messages(session_id)

        if len(messages) > max_messages:
            old_messages = messages[:max_messages//2]

            # 대화 내용 구성
            conversation = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in old_messages
            ])

            from .prompts import SUMMARIZATION_PROMPT

            # LangChain 1.0에서는 invoke 사용
            summary = self.llm.invoke(
                SUMMARIZATION_PROMPT.format(conversation=conversation)
            ).content

            # 요약본 저장
            self.save_message(session_id, "system", f"[이전 대화 요약] {summary}")

            # 오래된 메세지 삭제
            key = self.get_session_key(session_id)
            for _ in range(len(old_messages)):
                self.redis_client.lpop(key)

            return summary
        
        return None
    
    def get_conversation_stats(self, session_id: str) -> dict:
        """대화 통계"""
        messages = self.get_messages(session_id)

        if not messages:
            return {"total_messages": 0}

        return {
            "total_messages": len(messages),
            "human_messages": sum(1 for m in messages if m["role"] == "human"),
            "ai_messages": sum(1 for m in messages if m["role"] == "ai"),
            "start_time": messages[0]["timestamp"],
            "last_time": messages[-1]["timestamp"],
        }