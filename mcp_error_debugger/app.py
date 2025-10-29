"""
Error Debugger API with LangGraph
PHP 에러 → FastAPI → LangGraph Agent → 비즈니스 로직 분석
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Annotated, TypedDict
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# 환경 변수 로드
load_dotenv()

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# FastAPI 앱
app = FastAPI(
    title="Error Debugger API with LangGraph",
    description="LangGraph로 체계적인 에러 분석",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Pydantic Models ==========

class ErrorRequest(BaseModel):
    error_type: str = Field(..., description="에러 타입")
    error_message: str = Field(..., description="에러 메시지")
    stack_trace: str = Field(..., description="전체 스택 트레이스")
    input_params: Optional[str] = Field(None, description="입력 파라미터")
    server_base_path: str = Field(
        default="/Users/fanding/develop/legacy-php-api",
        description="서버 코드 기본 경로"
    )


# ========== LangChain Tools ==========

@tool
def read_file(file_path: str) -> str:
    """파일의 전체 내용을 읽습니다."""
    try:
        # /home/fanding → /Users/fanding 경로 변환 (macOS)
        if file_path.startswith('/home/fanding'):
            file_path = file_path.replace('/home/fanding', '/Users/fanding')

        if not os.path.isabs(file_path):
            possible_bases = [
                "/Users/fanding/develop/legacy-php-api",
                "/Users/fanding/develop/ppp",
                os.getcwd()
            ]
            for base in possible_bases:
                full_path = os.path.join(base, file_path)
                if os.path.exists(full_path):
                    file_path = full_path
                    break

        print(f"[DEBUG] read_file: {file_path}, exists={os.path.exists(file_path)}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"[DEBUG] read_file: 읽은 길이 = {len(content)} bytes")
            return content
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def search_files(directory: str, pattern: str = "*.php") -> str:
    """디렉토리에서 파일을 검색합니다."""
    try:
        import glob
        if not os.path.isabs(directory):
            directory = os.path.abspath(directory)
        search_pattern = os.path.join(directory, "**", pattern)
        files = glob.glob(search_pattern, recursive=True)
        return json.dumps(files[:30], ensure_ascii=False)
    except Exception as e:
        return json.dumps([f"ERROR: {str(e)}"], ensure_ascii=False)


@tool
def grep_code(file_path: str, search_term: str) -> str:
    """파일에서 특정 코드를 검색합니다."""
    try:
        # /home/fanding → /Users/fanding 경로 변환 (macOS)
        if file_path.startswith('/home/fanding'):
            file_path = file_path.replace('/home/fanding', '/Users/fanding')

        content = read_file.invoke({"file_path": file_path})
        if content.startswith("ERROR"):
            return content
        lines = content.split('\n')
        results = []
        for i, line in enumerate(lines, 1):
            if search_term.lower() in line.lower():
                results.append(f"Line {i}: {line.strip()}")
        return "\n".join(results[:15]) if results else f"'{search_term}' 없음"
    except Exception as e:
        return f"ERROR: {str(e)}"


# ========== LangGraph State ==========

from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]  # add operator로 메시지 누적
    error_info: dict
    analysis_result: Optional[str]


# ========== LangGraph Nodes ==========

def find_primary_error_file(stack_trace: str, base_paths: list) -> tuple:
    """스택 트레이스에서 핵심 에러 파일 찾기"""
    import re
    exclude = ['/system/', '/vendor/', '/core/', 'CodeIgniter.php', 'index.php', '/bootstrap/']

    for line in stack_trace.split('\n'):
        match = re.search(r'#\d+\s+([^(]+\.php)\((\d+)\)', line)
        if match:
            file_path, line_no = match.groups()
            if any(p in file_path for p in exclude):
                continue

            for base in base_paths:
                for sub in ['', 'application/controllers/rest', 'application/controllers', 'application/models']:
                    full = Path(base) / sub / Path(file_path).name
                    if full.exists():
                        return str(full), line_no
    return None, None


def analyze_node(state: AgentState):
    """에러 분석 노드"""
    print("\n🤖 AI 에이전트 분석 중...")

    messages = state["messages"]
    error_info = state["error_info"]

    # 디버깅: 현재 messages 상태 확인
    print(f"[DEBUG] Current messages count: {len(messages)}")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        print(f"  [{i}] {msg_type}, tool_calls={has_tool_calls}")

    # 🔧 경로 변환: /home/fanding → /Users/fanding (macOS 로컬 개발 환경)
    stack_trace = error_info['stack_trace'].replace('/home/fanding', '/Users/fanding')
    error_info['stack_trace'] = stack_trace

    # 🎯 핵심 에러 파일 먼저 읽기
    base_paths = [
        '/Users/fanding/develop/legacy-php-api',
        '/Users/fanding/develop/ppp',
        error_info.get('server_base_path', '')
    ]

    primary_file, error_line = find_primary_error_file(stack_trace, base_paths)

    context_code = ""
    if primary_file:
        print(f"📍 시작점: {primary_file}:{error_line}")
        try:
            with open(primary_file, 'r', encoding='utf-8') as f:
                context_code = f.read()[:5000]
        except Exception as e:
            print(f"⚠️  파일 읽기 실패: {e}")

    # 🔑 첫 번째 호출인지 확인
    is_first_call = len(messages) == 0

    # 시스템 프롬프트
    if is_first_call:
        # 첫 번째: 도구 사용 가능
        system_msg = SystemMessage(content=f"""당신은 숙련된 PHP 백엔드 에러 분석 전문가입니다.

**중요 정보:**
- 실제 파일 위치: /Users/fanding/develop/legacy-php-api
- 아래 에러 파일이 이미 제공되었습니다: {primary_file if primary_file else '없음'}
- 제공된 파일만으로 충분하면 즉시 분석하세요

**분석 방법:**
1. **에러가 발생한 정확한 라인과 변수 특정**
   - 어떤 변수/객체가 문제인가?
   - 왜 null이거나 예상과 다른 값인가?
   - 입력 파라미터 중 어떤 값이 잘못 들어왔는가?

2. **근본 원인 파악**
   - 호출 체인 분석 (어디서 넘어온 값인가?)
   - 비즈니스 로직상 왜 이런 상황이 발생했는가?
   - DB 쿼리 결과가 비어있는가? 조건문 체크가 누락됐는가?

3. **재발 방지를 위한 개선안 제시**
   - 즉시 해결: 에러가 안 나도록 수정
   - 장기 개선: 더 안전한 코드 구조 제안

**출력 형식:**
## 🔍 에러 위치
- 파일: [파일명]:[라인번호]
- 함수: [함수명]
- 문제 변수: [변수명]

## 💥 원인 분석
**즉시 원인:**
- [어떤 변수가 null/잘못된 값인지]
- [왜 그런 값이 들어왔는지]

**근본 원인:**
- [비즈니스 로직상 문제점]
- [호출 경로 추적]

## 🔧 해결 방법
**즉시 수정 (Hot Fix):**
```php
// 수정 전
[기존 코드]

// 수정 후
[수정된 코드 + 주석으로 설명]
```

**장기 개선안:**
- [더 안전한 코드 구조]
- [validation 추가 제안]
- [에러 핸들링 개선]

**규칙:**
- 제공된 에러 파일 코드를 우선 분석
- 정말 필요한 경우만 read_file로 추가 파일 조회 (최대 1-2개)
- 구체적인 변수명과 라인 번호 언급
- "파일이 없다"고 말하지 말고 제공된 코드를 분석
""")

        # 에러 정보
        content = f"""에러 분석:

타입: {error_info['error_type']}
메시지: {error_info['error_message']}

스택:
{error_info['stack_trace']}

파라미터: {error_info.get('input_params', '없음')}
"""

        if context_code:
            content += f"""

📄 에러 파일 ({primary_file}:{error_line}):
```php
{context_code}
```
"""

        error_msg = HumanMessage(content=content)

        llm_with_tools = llm.bind_tools([read_file, search_files, grep_code])
        response = llm_with_tools.invoke([system_msg, error_msg])

    else:
        # 두 번째 이후: 도구 결과를 바탕으로 최종 분석 (도구 없이)
        prompt_msg = HumanMessage(content="""도구 조회 결과를 바탕으로 최종 에러 분석을 작성하세요.

**출력 형식:**
## 🔍 에러 위치
- 파일: [파일명]:[라인번호]
- 함수: [함수명]
- 문제 변수: [변수명]

## 💥 원인 분석
**즉시 원인:**
- [어떤 변수가 null/잘못된 값인지]
- [왜 그런 값이 들어왔는지]

**근본 원인:**
- [비즈니스 로직상 문제점]
- [호출 경로 추적]

## 🔧 해결 방법
**즉시 수정 (Hot Fix):**
```php
// 수정 전
[기존 코드]

// 수정 후
[수정된 코드 + 주석으로 설명]
```

**장기 개선안:**
- [더 안전한 코드 구조]
- [validation 추가 제안]
- [에러 핸들링 개선]

**중요: 더 이상 도구를 호출하지 말고, 지금 바로 상세한 분석 결과를 작성하세요.**
""")

        # messages 순서 유지: [AI(tool_calls), ToolMessage, ...]
        response = llm.invoke(messages + [prompt_msg])

    return {"messages": messages + [response]}


def tool_node_wrapper(state: AgentState):
    """툴 실행 노드"""
    print(f"🔧 툴 실행 중...")

    messages = state["messages"]

    # 디버깅: 도구 실행 전 messages 확인
    print(f"[DEBUG] Before tool execution, messages count: {len(messages)}")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        print(f"  [{i}] {msg_type}, tool_calls={has_tool_calls}")

    tools = [read_file, search_files, grep_code]
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)

    # 디버깅: messages 구조 확인
    print(f"[DEBUG] After tool execution, result messages count: {len(result.get('messages', []))}")
    for i, msg in enumerate(result.get('messages', [])):
        msg_type = type(msg).__name__
        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        print(f"  [{i}] {msg_type}, tool_calls={has_tool_calls}")

    return result


def should_continue(state: AgentState):
    """계속할지 결정"""
    messages = state["messages"]
    last_message = messages[-1]

    # AI 메시지 카운트 (최대 5번만 반복)
    ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
    if ai_count >= 5:
        print(f"⚠️  최대 반복 횟수 도달 ({ai_count}회), 강제 종료")
        return "end"

    # 툴 호출이 있으면 계속
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"


def extract_result(state: AgentState):
    """최종 결과 추출"""
    messages = state["messages"]

    # 마지막 AI 메시지 찾기
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return {"analysis_result": msg.content}

    return {"analysis_result": "분석 실패"}


# ========== LangGraph 생성 ==========

workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("analyze", analyze_node)
workflow.add_node("tools", tool_node_wrapper)
workflow.add_node("extract", extract_result)

# 엣지 설정
workflow.set_entry_point("analyze")
workflow.add_conditional_edges(
    "analyze",
    should_continue,
    {
        "tools": "tools",
        "end": "extract"
    }
)
workflow.add_edge("tools", "analyze")
workflow.add_edge("extract", END)

# 컴파일
graph = workflow.compile()


# ========== Helper Functions ==========

def _extract_file_locations(stack_trace: str, base_path: str) -> List[dict]:
    """스택 트레이스에서 파일 위치 정보 추출"""
    locations = []

    # Python
    python_pattern = r'File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+(\w+))?'
    for match in re.findall(python_pattern, stack_trace):
        file_path, line_num, function = match
        if base_path in file_path or os.path.isabs(file_path):
            locations.append({
                'file': file_path,
                'line': int(line_num),
                'function': function or None,
                'language': 'python'
            })

    # PHP
    php_pattern = r'([/\w\-\.]+\.php)[\(:]+(\d+)\)?'
    for match in re.findall(php_pattern, stack_trace):
        file_path, line_num = match
        if base_path in file_path or os.path.isabs(file_path):
            locations.append({
                'file': file_path,
                'line': int(line_num),
                'function': None,
                'language': 'php'
            })

    return locations


# ========== FastAPI Endpoints ==========

@app.get("/health")
def health_check():
    """헬스 체크"""
    return {
        "status": "ok",
        "service": "error-debugger",
        "version": "4.0.0"
    }


@app.post("/analyze")
async def analyze_error(request: ErrorRequest):
    """LangGraph로 에러 분석"""
    try:
        print("\n" + "="*80)
        print("🚀 에러 분석 시작")
        print(f"타입: {request.error_type}")
        print(f"메시지: {request.error_message}")
        print("="*80)

        # 파일 위치 추출
        file_locations = _extract_file_locations(
            request.stack_trace,
            request.server_base_path
        )

        print(f"\n📍 파일 위치: {len(file_locations)}개")
        for loc in file_locations:
            print(f"  - {loc['file']}:{loc['line']}")

        if not file_locations:
            return {
                "success": False,
                "error": "스택 트레이스에서 파일 위치를 찾을 수 없음",
                "analysis": None
            }

        # LangGraph 실행
        initial_state = {
            "messages": [],
            "error_info": {
                "error_type": request.error_type,
                "error_message": request.error_message,
                "stack_trace": request.stack_trace,
                "input_params": request.input_params,
                "server_base_path": request.server_base_path
            },
            "analysis_result": None
        }

        # 그래프 실행
        final_state = None
        for state in graph.stream(initial_state, {"recursion_limit": 15}):
            final_state = state
            print(f"\n[DEBUG] State keys: {state.keys()}")

        print(f"\n[DEBUG] Final state: {final_state}")

        # 결과 추출
        if final_state and "extract" in final_state:
            analysis = final_state["extract"]["analysis_result"]
            print(f"[DEBUG] Got analysis from extract node: {analysis[:100]}...")
        elif final_state:
            # 마지막 상태에서 분석 결과 찾기
            last_state = list(final_state.values())[0]
            print(f"[DEBUG] Last state keys: {last_state.keys() if isinstance(last_state, dict) else 'not a dict'}")

            # messages에서 직접 추출 시도
            if "messages" in last_state:
                messages = last_state["messages"]
                print(f"[DEBUG] Messages count: {len(messages)}")
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        analysis = msg.content
                        print(f"[DEBUG] Found AI message: {analysis[:100]}...")
                        break
                else:
                    analysis = last_state.get("analysis_result", "분석 실패")
            else:
                analysis = last_state.get("analysis_result", "분석 실패")
        else:
            analysis = "분석 실패"

        print(f"\n✅ 분석 완료!")
        print(f"\n📝 결과:\n{analysis}")
        print("\n" + "="*80)

        return {
            "success": True,
            "file_locations": file_locations,
            "analysis": analysis
        }

    except Exception as e:
        print(f"\n❌ 에러: {str(e)}")
        print("="*80)
        raise HTTPException(
            status_code=500,
            detail=f"분석 실패: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Error Debugger API (LangGraph)")
    print("   - LangGraph State Machine")
    print("   - 간결한 분석 결과")
    print("   - Port: 9000")
    uvicorn.run(app, host="0.0.0.0", port=9000)
