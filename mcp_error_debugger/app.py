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
from dotenv import load_dotenv
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# MCP 관련 import
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio

# 환경 변수 로드
load_dotenv()

# LLM 초기화
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

# GitHub 저장소 정보
GITHUB_REPO_OWNER = os.getenv("GITHUB_REPO_OWNER", "fanding")
GITHUB_REPO_NAME = os.getenv("GITHUB_REPO_NAME", "legacy-php-api")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

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
    git_ref: Optional[str] = Field(
        default="master",
        description="Git 브랜치/태그/커밋 (예: master, develop, refs/heads/feature-branch)"
    )


# ========== GitHub MCP Tools ==========

# GitHub MCP 서버 파라미터를 전역으로 저장
github_mcp_server_params = None

async def initialize_github_mcp():
    """GitHub MCP 서버 파라미터를 초기화합니다."""
    global github_mcp_server_params

    if not GITHUB_TOKEN:
        print("⚠️  GITHUB_TOKEN이 설정되지 않아 GitHub MCP를 사용할 수 없습니다.")
        return False

    try:
        # GitHub MCP 서버 설정 (npx 사용)
        github_mcp_server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN
            }
        )

        print(f"🔌 GitHub MCP 서버 설정 완료 (repo: {GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME})")

        # 테스트 연결로 도구 목록 확인
        async with stdio_client(github_mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                print(f"✅ GitHub MCP 도구 {len(tools)}개 사용 가능")
                print(f"\n📋 사용 가능한 도구 목록:")
                for tool in tools[:26]:  # 처음 10개만 출력
                    print(f"  - {tool.name}: {tool.description[:80] if hasattr(tool, 'description') and tool.description else 'No description'}...")
               
                print()

        return True
    except Exception as e:
        print(f"❌ GitHub MCP 초기화 실패: {e}")
        return False


# ========== 로컬 파일 도구 제거 ==========
# GitHub MCP만 사용합니다.


# ========== LangGraph State ==========

from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]  # add operator로 메시지 누적
    error_info: dict
    error_line: int  # 에러 발생 라인 번호
    git_ref: str  # Git 브랜치/태그/커밋
    analysis_result: Optional[str]
    token_usage: dict  # 토큰 사용량 추적


# ========== LangGraph Nodes ==========


async def analyze_node(state: AgentState):
    """에러 분석 노드 (비동기)"""
    print("\n🤖 AI 에이전트 분석 중...")

    messages = state["messages"]
    error_info = state["error_info"]
    git_ref = state.get("git_ref", "enhance/ai-log-analysis")

    # 디버깅: 현재 messages 상태 확인
    print(f"[DEBUG] Current messages count: {len(messages)}")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        print(f"  [{i}] {msg_type}, tool_calls={has_tool_calls}")

    # 경로 정보만 출력 (로컬 파일 읽기 제거)
    stack_trace = error_info['stack_trace']
    print(f"📍 스택 트레이스 분석 중...")

    # 🔑 AI 메시지 카운트로 판단 (최대 3번까지 도구 사용 허용)
    ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
    should_use_tools = ai_count < 3  # 최대 3번까지 도구 사용

    # 시스템 프롬프트
    if should_use_tools:
        # 첫 번째: 도구 사용 가능
        system_msg = SystemMessage(content=f"""당신은 숙련된 PHP 백엔드 에러 분석 전문가입니다.

**중요 정보:**
- GitHub 저장소: {GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}
- Git 브랜치/커밋: {git_ref}
- 반드시 GitHub MCP 도구를 사용하여 저장소에서 파일을 읽어야 합니다

**핵심 규칙:**
⚠️ 전체 파일을 일반적으로 분석하지 마세요!
⚠️ 스택 트레이스의 **정확한 라인 번호**에 집중하세요!

**당신의 임무 (깊이 있는 분석):**
1. **에러 파일 읽기**
   - 스택 트레이스에서 파일 경로 추출
   - 예: `/home/fanding/application/controllers/rest/Post.php:851` → `application/controllers/rest/Post.php`
   - get_file_contents로 읽기 (owner={GITHUB_REPO_OWNER}, repo={GITHUB_REPO_NAME}, path=파일경로, ref={git_ref})

2. **에러 라인 정확히 분석**
   - 851번째 줄의 실제 코드 확인
   - 어떤 함수/클래스를 호출하는가?
   - 어떤 변수를 인자로 전달하는가?
   - 그 변수는 어디서 왔는가? (같은 함수 내에서 추적)

3. **관련 파일들 추가로 읽기 (중요!)**
   - 에러 라인에서 호출하는 클래스 파일 읽기
     예: `new Post_view_data($x)` → `repo/model_post/Post_view_data.php` 파일 읽기
   - 그 클래스의 __construct() 함수 확인 → 왜 int를 요구하는지?
   - 문제 변수가 다른 함수에서 왔다면, 그 함수도 추적
   - 필요하면 search_repository로 관련 파일 찾기

4. **함수 호출 흐름 추적**
   - 입력 파라미터 → 현재 함수 → 문제 변수 → 에러 발생
   - 각 단계에서 왜 타입이 변했는지 추적

5. **근본 원인 파악**
   - 비즈니스 로직상 왜 이런 상황이 발생했는가?
   - 앞단에서 validation이 빠졌는가?
   - 데이터베이스에서 잘못된 타입으로 가져왔는가?

**중요: 여러 파일을 읽으면서 깊이 파고드세요!**
- 한 파일만 읽고 끝내지 마세요
- 최소 2-3개 파일을 읽어야 근본 원인을 찾을 수 있습니다
- get_file_contents, search_repository 도구를 적극 활용하세요

**출력 형식 (간결하게!):**

## 🎯 원인 분석

**에러 위치:**
- 파일: Post.php:851
- 메서드: [메서드명]
- 코드: `[실제 코드 한 줄]`

**왜 에러가 났는가:**
1. [에러 라인에서 무엇을 했는지]
   예: `$badData = $this->model_post->getPostViewDataWithBadTypes()`
2. [그 메서드/함수가 무엇을 반환했는지]
   예: DB 쿼리 - `SELECT CONCAT('POST_', no) AS post_no ...`
3. [왜 타입이 안 맞는지]
   예: CONCAT은 문자열 반환 → 'POST_10738', 생성자는 int 요구

**해결:**
`(int)$badData['post_no']` 또는 쿼리 수정

**간결하게! 핵심만!**
""")

        # 첫 번째 호출인지 확인
        if ai_count == 0:
            # 첫 번째: 에러 파일 읽기 시작
            # 스택 트레이스에서 라인 번호 추출
            import re
            line_match = re.search(r'Post\.php.*?line (\d+)', error_info['error_message'])
            error_line = line_match.group(1) if line_match else "unknown"

            # 스택 트레이스에서 실제 전달된 인자 값 추출
            import re
            arg_match = re.search(r'__construct\((.*?)\)', error_info['stack_trace'])
            actual_args = arg_match.group(1) if arg_match else "확인 안 됨"

            content = f"""🚨 **에러 분석 시작** 🚨

**에러 정보:**
타입: {error_info['error_type']}
메시지: {error_info['error_message']}

**스택 트레이스:**
{error_info['stack_trace']}

**⚠️ 중요: 스택 트레이스에서 실제 전달된 인자 값:**
`__construct({actual_args})`
→ 이 값들을 분석에 활용하세요!

파라미터: {error_info.get('input_params', '없음')}

**첫 번째 작업: 에러가 발생한 파일을 읽으세요**
- get_file_contents 도구로 Post.php 파일 읽기
- owner: {GITHUB_REPO_OWNER}, repo: {GITHUB_REPO_NAME}, ref: {git_ref}
- path: application/controllers/rest/Post.php
- {error_line}번째 줄 주변을 중점적으로 확인
- 실제 전달된 값: {actual_args}
"""
        else:
            # 두 번째 이후: 더 깊이 파고들기
            content = f"""이전에 읽은 파일을 바탕으로 더 깊이 분석하세요.

**다음 작업:**
1. **에러 라인에서 호출하는 클래스/메서드를 추적하세요**
   - 851번 라인에서 어떤 메서드를 호출했는지 확인
   - 그 메서드가 정의된 파일을 찾아서 읽기
   - 예: `getPostViewDataWithBadTypes()` 같은 메서드 → model_post.php 읽기

2. **문제 변수의 출처를 추적하세요**
   - 스택 트레이스의 실제 값을 확인했으니, 왜 그런 값이 나왔는지 추적
   - DB 쿼리를 확인 (CONCAT, CAST 등 타입 변환 확인)
   - 예: `'POST_10738'` → CONCAT('POST_', no) 같은 쿼리 찾기

3. **클래스 파일을 찾으세요**
   - `Post_view_data` 클래스 파일 찾기
   - 경로: `application/objects/repo/model_post/Post_view_data.php`
   - search_repository나 get_file_contents 사용

**충분한 정보를 모았다면:**
- 구체적인 값과 메서드 이름을 언급하며 최종 분석 작성
"""

        error_msg = HumanMessage(content=content)

        # GitHub MCP 도구만 사용 - 매번 새로운 세션에서 도구 로드
        if not github_mcp_server_params:
            raise Exception("GitHub MCP가 초기화되지 않았습니다. GITHUB_TOKEN을 설정하고 서버를 재시작하세요.")

        # 새로운 세션 생성 및 도구 로드
        async with stdio_client(github_mcp_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)

                llm_with_tools = llm.bind_tools(tools)
                response = await llm_with_tools.ainvoke([system_msg, error_msg])

    else:
        # 3번 이후: 도구 사용 끝, 최종 분석 (도구 없이)
        print(f"🏁 최종 분석 단계 (AI 호출 {ai_count + 1}회차)")
        prompt_msg = HumanMessage(content="""지금까지 읽은 파일을 바탕으로 간결하게 분석하세요.

**형식:**
## 🎯 원인 분석
**에러 위치:** 파일:라인, 메서드, 코드
**왜 에러가 났는가:** 3단계로 (무엇을 했는지 → 무엇을 반환했는지 → 왜 타입 안 맞는지)
**해결:** 한 줄 수정

간결하게! 사족 없이!
""")

        # messages 순서 유지: [AI(tool_calls), ToolMessage, ...]
        response = await llm.ainvoke(messages + [prompt_msg])

    # 토큰 사용량 추적
    current_token_usage = state.get("token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    # 디버깅: response 구조 확인
    print(f"[DEBUG] Response type: {type(response)}")
    print(f"[DEBUG] Has usage_metadata: {hasattr(response, 'usage_metadata')}")
    print(f"[DEBUG] Has response_metadata: {hasattr(response, 'response_metadata')}")
    if hasattr(response, 'response_metadata'):
        print(f"[DEBUG] response_metadata keys: {response.response_metadata.keys() if response.response_metadata else 'None'}")

    # LangChain AIMessage의 usage_metadata 확인
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        current_token_usage["input_tokens"] += response.usage_metadata.get("input_tokens", 0)
        current_token_usage["output_tokens"] += response.usage_metadata.get("output_tokens", 0)
        current_token_usage["total_tokens"] += response.usage_metadata.get("total_tokens", 0)
        print(f"  [AI 호출 {ai_count + 1}] 입력: {response.usage_metadata.get('input_tokens', 0)}, 출력: {response.usage_metadata.get('output_tokens', 0)}")
    elif hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
        # 다른 형태의 메타데이터
        token_info = response.response_metadata['token_usage']
        current_token_usage["input_tokens"] += token_info.get("prompt_tokens", 0)
        current_token_usage["output_tokens"] += token_info.get("completion_tokens", 0)
        current_token_usage["total_tokens"] += token_info.get("total_tokens", 0)
        print(f"  [AI 호출 {ai_count + 1}] 입력: {token_info.get('prompt_tokens', 0)}, 출력: {token_info.get('completion_tokens', 0)}")

    return {
        "messages": messages + [response],
        "token_usage": current_token_usage
    }


async def tool_node_wrapper(state: AgentState):
    """툴 실행 노드 (비동기) - 매번 새로운 GitHub MCP 세션 생성"""
    print(f"🔧 GitHub MCP 툴 실행 중...")

    messages = state["messages"]

    # 디버깅: 도구 실행 전 messages 확인
    print(f"[DEBUG] Before tool execution, messages count: {len(messages)}")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        if has_tool_calls:
            tool_names = [tc.get('name') for tc in msg.tool_calls]
            tool_args = [tc.get('args') for tc in msg.tool_calls]
            print(f"  [{i}] {msg_type}, tool_calls={tool_names}")
            for j, (name, args) in enumerate(zip(tool_names, tool_args)):
                print(f"      Tool {j}: {name}({args})")
        else:
            print(f"  [{i}] {msg_type}, tool_calls=False")

    # GitHub MCP 세션 확인
    if not github_mcp_server_params:
        raise Exception("GitHub MCP가 초기화되지 않았습니다.")

    # 새로운 세션 생성 및 도구 실행
    async with stdio_client(github_mcp_server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            tool_node = ToolNode(tools)
            result = await tool_node.ainvoke(state)

            # GitHub 파일 내용 처리 및 에러 라인 추출
            import json

            # state에서 에러 라인 번호 가져오기
            error_line_num = state.get('error_line')
            if error_line_num:
                print(f"🎯 에러 라인 번호 사용: {error_line_num}")
            else:
                print("⚠️ 에러 라인 번호를 찾을 수 없음")

            messages = result.get('messages', [])
            for msg in messages:
                if type(msg).__name__ == 'ToolMessage':
                    try:
                        # JSON 파싱
                        parsed = json.loads(msg.content)

                        # content 필드가 있는 경우 (GitHub MCP는 이미 디코딩된 문자열을 반환함)
                        if 'content' in parsed:
                            file_content = parsed['content']
                            print(f"✅ GitHub 파일 내용 확인: {len(file_content)} chars")

                            # 에러 라인 주변 코드 추출 (±30줄)
                            if error_line_num:
                                lines = file_content.split('\n')
                                total_lines = len(lines)

                                # 에러 라인 주변만 추출 (전체 파일 대신)
                                context_range = 30
                                start = max(0, error_line_num - context_range - 1)  # 배열은 0-based
                                end = min(total_lines, error_line_num + context_range)

                                error_lines = []
                                for i in range(start, end):
                                    line_marker = ">>> 🔥 " if (i + 1) == error_line_num else "     "
                                    error_lines.append(f"{line_marker}{i+1:4d} | {lines[i]}")

                                error_context = "\n".join(error_lines)
                                print(f"✅ 에러 라인 컨텍스트 추출 완료: {error_line_num}번 라인 (±{context_range}줄)")

                                # 새로운 형식으로 변환 - 에러 라인 주변 코드만 제공
                                new_content = f"""📄 파일: {parsed.get('name', 'unknown')}
경로: {parsed.get('path', 'unknown')}
전체 크기: {parsed.get('size', 0)} bytes (총 {total_lines}줄)

🎯🎯🎯 에러 발생 라인 {error_line_num} 주변 코드 (±{context_range}줄) 🎯🎯🎯
{'='*80}
{error_context}
{'='*80}

⚠️ **중요: {error_line_num}번 라인 (🔥 표시)의 코드를 정확히 분석하세요!**
이 라인에서 Post_view_data::__construct()가 호출되고 있고,
첫 번째 인자로 int가 아닌 string이 전달되어 에러가 발생했습니다.
"""
                            else:
                                # 에러 라인을 못 찾은 경우에만 전체 파일 제공
                                print("⚠️ 에러 라인 번호를 찾을 수 없어 전체 파일 제공")
                                new_content = f"""📄 파일: {parsed.get('name', 'unknown')}
경로: {parsed.get('path', 'unknown')}
크기: {parsed.get('size', 0)} bytes

=== 전체 파일 내용 ===
{file_content}
=== 파일 내용 끝 ===
"""

                            # 메시지 내용 교체
                            msg.content = new_content
                            print(f"✅ GitHub 파일 포맷 변환 완료")
                    except Exception as e:
                        import traceback
                        print(f"⚠️ 파일 처리 실패: {e}")
                        print(f"📍 상세 에러:\n{traceback.format_exc()}")

            # 디버깅: messages 구조 확인
            print(f"[DEBUG] After tool execution, result messages count: {len(messages)}")
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                if msg_type == 'ToolMessage':
                    content_preview = str(msg.content)[:300]
                    print(f"  [{i}] {msg_type}, content_preview: {content_preview}...")
                else:
                    print(f"  [{i}] {msg_type}")

            return result


def should_continue(state: AgentState):
    """계속할지 결정"""
    messages = state["messages"]
    last_message = messages[-1]

    # AI 메시지 카운트 (최대 4번만 반복)
    ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
    if ai_count >= 4:
        print(f"✅ 분석 완료 ({ai_count}회 반복)")
        return "end"

    # 툴 호출이 있으면 계속
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"


async def extract_result(state: AgentState):
    """최종 결과 추출 (비동기)"""
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

        # 에러 라인 번호 추출 (스택 트레이스에서 첫 번째 발생 위치)
        import re
        error_line = None
        line_match = re.search(r'Post\.php.*?line (\d+)', request.error_message)
        if line_match:
            error_line = int(line_match.group(1))
            print(f"🎯 에러 라인 추출: {error_line}")

        # Git ref 정보 출력
        print(f"📌 Git ref: {request.git_ref}")

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
            "error_line": error_line,
            "git_ref": request.git_ref,
            "analysis_result": None,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }

        # 그래프 실행 (비동기) - 전체 상태 추적
        final_state = None
        accumulated_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        async for state in graph.astream(initial_state, {"recursion_limit": 15}):
            print(f"\n[DEBUG] State keys: {state.keys()}")

            # 각 노드의 token_usage 누적
            for node_name, node_state in state.items():
                if isinstance(node_state, dict) and "token_usage" in node_state:
                    accumulated_token_usage = node_state["token_usage"]
                    print(f"[DEBUG] {node_name} 노드의 토큰: {accumulated_token_usage}")

            final_state = state

        print(f"\n[DEBUG] Final state: {final_state}")

        # 토큰 사용량 출력
        if accumulated_token_usage["total_tokens"] > 0:
            print(f"\n{'='*60}")
            print(f"📊 토큰 사용량")
            print(f"{'='*60}")
            print(f"  입력 토큰  : {accumulated_token_usage['input_tokens']:>10,}")
            print(f"  출력 토큰  : {accumulated_token_usage['output_tokens']:>10,}")
            print(f"  {'─'*56}")
            print(f"  총 토큰    : {accumulated_token_usage['total_tokens']:>10,}")
            print(f"{'='*60}")
        else:
            print("\n⚠️ 토큰 사용량 정보를 찾을 수 없습니다.")

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


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 GitHub MCP 초기화"""
    print("\n" + "="*80)
    print("🚀 Error Debugger API (LangGraph) 시작")
    print("="*80)
    await initialize_github_mcp()
    print("="*80 + "\n")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Error Debugger API (LangGraph)")
    print("   - LangGraph State Machine")
    print("   - GitHub MCP 통합")
    print("   - Port: 9000")
    print("\n💡 GitHub MCP 사용을 위해 .env 파일에 다음을 설정하세요:")
    print("   GITHUB_TOKEN=your_github_personal_access_token")
    print("   GITHUB_REPO_OWNER=fanding")
    print("   GITHUB_REPO_NAME=legacy-php-api\n")
    uvicorn.run(app, host="0.0.0.0", port=9000)
