import asyncio
from chatbot.agent import ChatbotAgent
import uuid

async def test_basic_conversation():
    """기본 대화 테스트"""
    chatbot = ChatbotAgent()
    session_id = str(uuid.uuid4())
    
    print("=== 기본 대화 테스트 ===\n")
    
    # 대화 1
    response1 = await chatbot.chat("안녕하세요!", session_id)
    print(f"User: 안녕하세요!")
    print(f"Bot: {response1}\n")
    
    # 대화 2 (맥락 유지)
    response2 = await chatbot.chat("제 이름은 김철수예요", session_id)
    print(f"User: 제 이름은 김철수예요")
    print(f"Bot: {response2}\n")
    
    # 대화 3 (기억 테스트)
    response3 = await chatbot.chat("제 이름이 뭐라고 했죠?", session_id)
    print(f"User: 제 이름이 뭐라고 했죠?")
    print(f"Bot: {response3}\n")

async def test_tool_usage():
    """도구 사용 테스트"""
    chatbot = ChatbotAgent()
    session_id = str(uuid.uuid4())
    
    print("=== 도구 사용 테스트 ===\n")
    
    # 계산기
    response1 = await chatbot.chat("123 곱하기 456은 얼마인가요?", session_id)
    print(f"User: 123 곱하기 456은 얼마인가요?")
    print(f"Bot: {response1}\n")
    
    # 시간
    response2 = await chatbot.chat("지금 몇 시예요?", session_id)
    print(f"User: 지금 몇 시예요?")
    print(f"Bot: {response2}\n")
    
    # 날씨
    response3 = await chatbot.chat("서울 날씨 알려주세요", session_id)
    print(f"User: 서울 날씨 알려주세요")
    print(f"Bot: {response3}\n")

async def test_long_conversation():
    """긴 대화 테스트 (요약 기능)"""
    chatbot = ChatbotAgent()
    session_id = str(uuid.uuid4())
    
    print("=== 긴 대화 테스트 ===\n")
    
    topics = [
        "파이썬에 대해 알려주세요",
        "파이썬의 장점은 뭔가요?",
        "그럼 단점은요?",
        "자바와 비교하면 어떤가요?",
        "초보자가 배우기 쉬운가요?",
        "추천 학습 자료 있나요?",
        "프로젝트 예시를 들어주세요",
        "실무에서 많이 쓰나요?",
        "연봉은 어떤가요?",
        "취업 전망은 어때요?",
    ]
    
    for i, topic in enumerate(topics, 1):
        response = await chatbot.chat(topic, session_id)
        print(f"[{i}] User: {topic}")
        print(f"    Bot: {response[:100]}...\n")
    
    # 통계 확인
    stats = chatbot.get_stats(session_id)
    print(f"통계: {stats}")

async def test_multi_session():
    """멀티 세션 테스트"""
    chatbot = ChatbotAgent()
    
    print("=== 멀티 세션 테스트 ===\n")
    
    # 세션 1
    session1 = str(uuid.uuid4())
    await chatbot.chat("제 이름은 Alice입니다", session1)
    response1 = await chatbot.chat("제 이름이 뭐죠?", session1)
    print(f"Session 1: {response1}")
    
    # 세션 2
    session2 = str(uuid.uuid4())
    await chatbot.chat("제 이름은 Bob입니다", session2)
    response2 = await chatbot.chat("제 이름이 뭐죠?", session2)
    print(f"Session 2: {response2}")
    
    # 세션 1 다시 확인
    response3 = await chatbot.chat("다시 한번, 제 이름은?", session1)
    print(f"Session 1 again: {response3}")

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_basic_conversation())
    asyncio.run(test_tool_usage())
    # asyncio.run(test_long_conversation())  # 시간 오래 걸림
    # asyncio.run(test_multi_session())