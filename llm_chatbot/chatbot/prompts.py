from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 시스템 프롬프트
SYSTEM_PROMPT = """당신은 친절하고 지적인 한국어 AI 어시스턴트입니다.

당신의 역할:
- 사용자의 질문에 정확하고 상세하게 답변합니다
- 한국어로 자연스럽게 대화합니다
- 이전 대화 내용을 기억하고 맥락을 이해합니다
- 필요시 도구(tools)를 활용하여 정보를 검색하거나 계산합니다
- 확실하지 않은 내용은 추측하지 않고 모른다고 솔직히 말합니다

대화 스타일:
- 존댓말을 사용하되 친근하게 대화합니다
- 답변은 명확하고 구조화되어 있습니다
- 필요시 예시를 들어 설명합니다
- 장황하지 않고 핵심을 전달합니다

현재 날짜: {current_date}
"""

# 메인 프롬프트 템플릿
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 요약 프롬프트 (긴 대화 요약용)
SUMMARIZATION_PROMPT = """다음은 사용자와의 대화 기록입니다. 
이 대화의 핵심 내용을 3-5문장으로 요약해주세요.
중요한 정보, 맥락, 사용자의 관심사를 포함해주세요.

대화 내용:
{conversation}

요약:"""