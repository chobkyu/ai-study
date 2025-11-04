"""
Error Debugger API with AI Agent + FastMCP
PHP 에러 → FastAPI → AI Agent (OpenAI + FastMCP tools) → 비즈니스 로직 분석
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import json

# 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI 앱
app = FastAPI(
    title="Error Debugger API (Local Files)",
    description="AI 에이전트가 로컬 파일을 읽으며 비즈니스 로직까지 분석",
    version="3.0.0-local"
)

# CORS 설정
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


# ========== Tools for AI Agent ==========
# OpenAI 에이전트가 사용할 툴들

def read_file(file_path: str, max_lines: int = 2000, error_line: int = None, context_range: int = 50) -> str:
    """
    파일 내용을 읽습니다. 에러 라인이 지정되면 주변 컨텍스트만 반환합니다.

    Args:
        file_path: 읽을 파일의 경로 (절대경로 또는 상대경로)
        max_lines: 최대 읽을 라인 수 (기본 2000줄)
        error_line: 에러 발생 라인 번호 (지정 시 주변만 반환)
        context_range: 에러 라인 주변 범위 (기본 50줄)

    Returns:
        파일 내용 또는 에러 라인 주변 컨텍스트
    """
    try:
        # 상대경로를 절대경로로 변환
        if not os.path.isabs(file_path):
            possible_bases = [
                "/Users/fanding/develop/legacy-php-api",
                "/Users/fanding/develop/ppp",
                os.getcwd()
            ]

            found = False
            for base in possible_bases:
                full_path = os.path.join(base, file_path)
                if os.path.exists(full_path):
                    file_path = full_path
                    found = True
                    break

            if not found:
                return f"ERROR: 파일을 찾을 수 없습니다: {file_path}"

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)

        # 에러 라인이 지정된 경우 주변 컨텍스트만 반환
        if error_line is not None and error_line > 0:
            start = max(0, error_line - context_range - 1)
            end = min(total_lines, error_line + context_range)

            context_lines = []
            for i in range(start, end):
                line_marker = ">>> 🔥 " if (i + 1) == error_line else "     "
                context_lines.append(f"{line_marker}{i+1:4d} | {lines[i].rstrip()}")

            context = "\n".join(context_lines)
            header = f"📄 파일: {os.path.basename(file_path)}\n"
            header += f"경로: {file_path}\n"
            header += f"전체 크기: {total_lines}줄\n\n"
            header += f"🎯 에러 발생 라인 {error_line} 주변 코드 (±{context_range}줄)\n"
            header += "="*80 + "\n"
            footer = "\n" + "="*80 + "\n"
            footer += f"\n⚠️ 에러는 {error_line}번 라인 (🔥 표시)에서 발생했습니다.\n"

            return header + context + footer

        # 에러 라인이 없으면 기존 방식대로
        # 파일이 너무 크면 잘라서 반환
        if total_lines > max_lines:
            content = ''.join(lines[:max_lines])
            warning = f"\n\n⚠️ 파일이 너무 커서 처음 {max_lines}줄만 표시합니다 (전체: {total_lines}줄)\n"
            warning += f"특정 부분이 필요하면 grep_code로 검색하세요.\n"
            return warning + "\n" + content
        else:
            return ''.join(lines)

    except Exception as e:
        return f"ERROR: 파일 읽기 실패: {str(e)}"


def search_files(directory: str, pattern: str = "*.php") -> str:
    """
    디렉토리에서 특정 패턴의 파일들을 검색합니다.

    Args:
        directory: 검색할 디렉토리 경로
        pattern: 파일 패턴 (예: *.php, *.py, UserController.php)

    Returns:
        검색된 파일 목록 (JSON 문자열)
    """
    try:
        import glob

        if not os.path.isabs(directory):
            directory = os.path.abspath(directory)

        search_pattern = os.path.join(directory, "**", pattern)
        files = glob.glob(search_pattern, recursive=True)

        # 최대 50개로 제한
        result = files[:50]
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps([f"ERROR: {str(e)}"], ensure_ascii=False)


def grep_code(file_path: str, search_term: str, max_results: int = 10) -> str:
    """
    파일에서 특정 코드나 함수를 검색합니다. (토큰 절약을 위해 최대 10개 결과만 반환)

    Args:
        file_path: 검색할 파일 경로
        search_term: 검색할 코드 (함수명, 클래스명, 변수명 등)
        max_results: 최대 결과 개수 (기본 10개)

    Returns:
        검색 결과 (라인 번호와 내용)
    """
    try:
        # 파일을 작게 읽기 (최대 1000줄)
        content = read_file(file_path, max_lines=1000)

        if content.startswith("ERROR"):
            return content

        lines = content.split('\n')
        results = []

        for i, line in enumerate(lines, 1):
            if search_term.lower() in line.lower():
                # 라인을 짧게 자르기 (최대 150자)
                trimmed = line.strip()[:150]
                results.append(f"Line {i}: {trimmed}")

                if len(results) >= max_results:
                    break

        if not results:
            return f"'{search_term}'을(를) 찾을 수 없습니다."

        return "\n".join(results)

    except Exception as e:
        return f"ERROR: {str(e)}"


def list_directory(directory: str) -> str:
    """
    디렉토리의 파일과 폴더 목록을 반환합니다.

    Args:
        directory: 조회할 디렉토리 경로

    Returns:
        파일과 폴더 목록 (JSON 문자열)
    """
    try:
        if not os.path.isabs(directory):
            directory = os.path.abspath(directory)

        if not os.path.exists(directory):
            return json.dumps({"error": f"디렉토리가 존재하지 않습니다: {directory}"})

        items = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            items.append({
                "name": item,
                "path": item_path,
                "is_dir": os.path.isdir(item_path)
            })

        return json.dumps(items, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# FastMCP 툴들을 OpenAI function calling 형식으로 변환
def get_openai_tools():
    """FastMCP 툴들을 OpenAI function calling 형식으로 변환"""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "파일의 내용을 읽습니다. error_line이 지정되면 해당 라인 주변만 반환하여 토큰을 절약합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "읽을 파일의 경로 (절대경로 또는 상대경로)"
                        },
                        "error_line": {
                            "type": "integer",
                            "description": "에러가 발생한 라인 번호 (지정 시 주변 ±50줄만 반환)"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "디렉토리에서 특정 패턴의 파일들을 검색합니다. 관련 파일을 찾을 때 사용하세요.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "검색할 디렉토리 경로"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "파일 패턴 (예: *.php, *.py, UserController.php)",
                            "default": "*.php"
                        }
                    },
                    "required": ["directory"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "grep_code",
                "description": "파일에서 특정 코드나 함수를 검색합니다. 함수 정의나 클래스를 찾을 때 사용하세요.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "검색할 파일 경로"
                        },
                        "search_term": {
                            "type": "string",
                            "description": "검색할 코드 (함수명, 클래스명, 변수명 등)"
                        }
                    },
                    "required": ["file_path", "search_term"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "디렉토리의 파일과 폴더 목록을 반환합니다. 프로젝트 구조를 파악할 때 사용하세요.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "조회할 디렉토리 경로"
                        }
                    },
                    "required": ["directory"]
                }
            }
        }
    ]


def execute_tool(tool_name: str, arguments: dict) -> str:
    """툴 실행"""
    if tool_name == "read_file":
        error_line = arguments.get("error_line")
        return read_file(arguments["file_path"], error_line=error_line)
    elif tool_name == "search_files":
        pattern = arguments.get("pattern", "*.php")
        return search_files(arguments["directory"], pattern)
    elif tool_name == "grep_code":
        return grep_code(arguments["file_path"], arguments["search_term"])
    elif tool_name == "list_directory":
        return list_directory(arguments["directory"])
    else:
        return f"ERROR: 알 수 없는 툴: {tool_name}"


# ========== Helper Functions ==========

def _extract_stack_trace_insights(stack_trace: str) -> dict:
    """
    스택 트레이스에서 중요한 정보를 추출합니다.
    - 실제 함수 호출 시 전달된 인자 값
    - 함수/메서드 이름
    - 타입 정보
    """
    insights = {
        "actual_arguments": [],
        "function_calls": [],
        "type_errors": []
    }

    # PHP 함수 호출에서 실제 인자 값 추출
    # 예: __construct('POST_10738', '1746', 'yes', 'invalid_price')
    arg_pattern = r'(\w+)\((.*?)\)'
    for match in re.finditer(arg_pattern, stack_trace):
        function_name = match.group(1)
        args_str = match.group(2)

        if args_str and args_str.strip():
            insights["function_calls"].append({
                "function": function_name,
                "arguments": args_str
            })

            # __construct나 주요 함수면 강조
            if function_name in ['__construct', 'new', 'call_user_func']:
                insights["actual_arguments"].append(f"{function_name}({args_str})")

    # 타입 에러 정보 추출
    # 예: "must be of the type int, string given"
    type_pattern = r'must be of the type (\w+), (\w+) given'
    type_match = re.search(type_pattern, stack_trace)
    if type_match:
        insights["type_errors"].append({
            "expected": type_match.group(1),
            "actual": type_match.group(2)
        })

    return insights

def _extract_file_locations(stack_trace: str, base_path: str) -> List[dict]:
    """스택 트레이스에서 파일 위치 정보 추출"""
    locations = []

    # Python 스타일
    python_pattern = r'File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+(\w+))?'
    python_matches = re.findall(python_pattern, stack_trace)

    for match in python_matches:
        file_path, line_num, function = match
        if base_path in file_path or os.path.isabs(file_path):
            locations.append({
                'file': file_path,
                'line': int(line_num),
                'function': function if function else None,
                'language': 'python'
            })

    # PHP 스타일
    php_pattern = r'([/\w\-\.]+\.php)[\(:]+(\d+)\)?'
    php_matches = re.findall(php_pattern, stack_trace)

    for match in php_matches:
        file_path, line_num = match
        if base_path in file_path or os.path.isabs(file_path):
            locations.append({
                'file': file_path,
                'line': int(line_num),
                'function': None,
                'language': 'php'
            })

    return locations


async def _analyze_with_ai_agent(
    error_type: str,
    error_message: str,
    stack_trace: str,
    file_locations: List[dict],
    error_line: int = None,
    input_params: Optional[str] = None,
    server_base_path: str = "/Users/fanding/develop/legacy-php-api"
) -> dict:
    """
    AI 에이전트가 FastMCP 툴을 사용하며 비즈니스 로직까지 분석합니다.
    """

    # 스택 트레이스 분석
    stack_insights = _extract_stack_trace_insights(stack_trace)

    # 실제 인자 값 강조
    actual_args_info = ""
    if stack_insights["actual_arguments"]:
        actual_args_info = f"""
🔥 **스택 트레이스에서 발견한 실제 인자 값:**
{chr(10).join(f"   - {arg}" for arg in stack_insights["actual_arguments"])}

⚠️ **이 값들이 핵심입니다!** 왜 이런 값이 전달되었는지 추적하세요!
"""

    # 타입 에러 정보
    type_error_info = ""
    if stack_insights["type_errors"]:
        te = stack_insights["type_errors"][0]
        type_error_info = f"**타입 불일치:** 예상={te['expected']}, 실제={te['actual']}\n"

    # 초기 컨텍스트 (간결하게!)
    initial_context = f"""PHP 에러 디버깅. 빠르고 간결하게!

**에러:** {error_type}
**메시지:** {error_message}
{type_error_info}**에러 라인:** {error_line if error_line else "확인 필요"}
{actual_args_info}
**임무 (3단계만):**
1. 에러 파일 읽기: `read_file(file_path="{server_base_path}/application/controllers/rest/Post.php", error_line={error_line})`
2. 호출된 메서드 파일 읽기 (1개만): 보통 model_post.php 같은 파일
   - **큰 파일이므로 read_file만 사용** (grep_code ❌)
3. **즉시 분석 완료** - 위 2개 파일만으로 충분!

**절대 금지:**
❌ grep_code 사용 금지 (파일이 너무 커서 비효율적)
❌ search_files 사용 금지
❌ 3개 이상 파일 읽기 금지

**목표:** read_file 2-3회만 호출하고 바로 최종 분석!

## 분석 결과 형식 (간결하게!)

### 🎯 원인 분석
**에러 위치:**
- 파일: Post.php:851
- 메서드: [메서드명]
- 코드: `[실제 코드 한 줄]`

**왜 에러가 났는가:**
1. [에러 라인에서 무엇을 했는지] (예: `getPostViewDataWithBadTypes()` 호출)
2. [그 메서드/함수가 무엇을 반환했는지] (예: DB 쿼리 결과 - `CONCAT('POST_', no)`)
3. [왜 타입이 안 맞는지] (예: CONCAT은 문자열 반환, 생성자는 int 요구)

**해결 방법:**
- 한 줄 수정: `[구체적인 코드 수정]` (예: `(int)$badData['post_no']`)

**간결하게! 사족 없이 핵심만!**
"""

    messages = [
        {"role": "system", "content": "전문 PHP 디버거. **절대 규칙: grep_code 사용 금지!** read_file만 사용하고 error_line 파라미터 필수. 2-3개 파일만 읽고 즉시 분석 완료."},
        {"role": "user", "content": initial_context}
    ]

    tool_calls_history = []
    max_iterations = 8  # 최대 8번 반복 (여유 있게)

    # 토큰 사용량 추적
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for iteration in range(max_iterations):
        try:
            # 4번째 반복부터는 툴 사용 중단하고 분석 요청
            if iteration >= 4:
                # 강제로 최종 분석 유도
                messages.append({
                    "role": "user",
                    "content": "충분한 파일을 읽었습니다. 이제 툴 호출 없이 **즉시 최종 분석**을 작성하세요!"
                })
                response = openai_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4000
                )
            else:
                # OpenAI API 호출
                response = openai_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    tools=get_openai_tools(),
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=4000
                )

            # 토큰 사용량 누적
            if hasattr(response, 'usage'):
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
                total_tokens += response.usage.total_tokens

            assistant_message = response.choices[0].message

            # 툴 호출이 없으면 최종 답변
            if not assistant_message.tool_calls:
                return {
                    "analysis": assistant_message.content,
                    "tool_calls": tool_calls_history,
                    "iterations": iteration + 1,
                    "token_usage": {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_tokens
                    }
                }

            # 툴 호출 실행
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # 툴 실행
                tool_result = execute_tool(function_name, function_args)

                # 히스토리 저장
                tool_calls_history.append({
                    "tool": function_name,
                    "arguments": function_args,
                    "result_preview": tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                })

                # 🔥 토큰 절약: 파일 내용이 너무 길면 요약해서 저장
                condensed_result = tool_result
                if function_name == "read_file" and len(tool_result) > 3000:
                    # 파일 내용이 3000자 넘으면 앞부분만 유지
                    condensed_result = tool_result[:3000] + f"\n\n... (나머지 {len(tool_result) - 3000}자 생략, 필요하면 다시 읽으세요)"

                # 결과를 메시지에 추가
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": condensed_result
                })

        except Exception as e:
            return {
                "analysis": f"AI 에이전트 분석 중 에러 발생: {str(e)}",
                "tool_calls": tool_calls_history,
                "iterations": iteration + 1,
                "error": str(e),
                "token_usage": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_tokens": total_tokens
                }
            }

    # 최대 반복 도달
    return {
        "analysis": "최대 반복 횟수에 도달했습니다. 부분 분석 결과를 확인하세요.",
        "tool_calls": tool_calls_history,
        "iterations": max_iterations,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens
        }
    }


# ========== FastAPI Endpoints ==========

@app.get("/health")
def health_check():
    """헬스 체크"""
    return {
        "status": "ok",
        "service": "error-debugger",
        "version": "3.0.0"
    }


@app.post("/analyze")
async def analyze_error(request: ErrorRequest):
    """
    AI 에이전트가 FastMCP 툴을 사용하며 비즈니스 로직까지 분석합니다.

    Flow:
    1. PHP 서버 → FastAPI (에러 정보)
    2. AI 에이전트 시작
    3. AI가 FastMCP read_file로 여러 파일 읽음
    4. AI가 FastMCP search_files로 관련 파일 찾음
    5. AI가 FastMCP grep_code로 함수/클래스 찾음
    6. AI가 FastMCP list_directory로 구조 파악
    7. AI가 비즈니스 로직 분석
    8. 최종 결과 반환
    """
    try:
        print("\n" + "="*80)
        print("🚀 에러 분석 요청 시작")
        print("="*80)
        print(f"에러 타입: {request.error_type}")
        print(f"에러 메시지: {request.error_message}")
        print(f"스택 트레이스:\n{request.stack_trace}")
        print(f"입력 파라미터: {request.input_params}")
        print(f"서버 경로: {request.server_base_path}")
        print("="*80)

        # 1. 스택 트레이스에서 파일 위치 추출
        file_locations = _extract_file_locations(
            request.stack_trace,
            request.server_base_path
        )

        print(f"\n📍 추출된 파일 위치: {len(file_locations)}개")
        for loc in file_locations:
            print(f"  - {loc['file']}:{loc['line']}")

        if not file_locations:
            print("❌ 스택 트레이스에서 파일 위치를 찾을 수 없습니다.")
            return {
                "success": False,
                "error": "스택 트레이스에서 파일 위치를 찾을 수 없습니다.",
                "analysis": None
            }

        # 에러 라인 번호 추출
        error_line = None
        if file_locations:
            error_line = file_locations[0].get('line')
            print(f"🎯 에러 라인 번호: {error_line}")

        # 2. AI 에이전트가 FastMCP 툴을 사용하며 분석
        print("\n🤖 AI 에이전트 분석 시작...")
        result = await _analyze_with_ai_agent(
            error_type=request.error_type,
            error_message=request.error_message,
            stack_trace=request.stack_trace,
            file_locations=file_locations,
            error_line=error_line,
            input_params=request.input_params,
            server_base_path=request.server_base_path
        )

        print(f"\n✅ 분석 완료!")
        print(f"  - 툴 호출 횟수: {len(result['tool_calls'])}회")
        print(f"  - 반복 횟수: {result['iterations']}회")

        # 토큰 사용량 출력
        if 'token_usage' in result:
            token_info = result['token_usage']
            print(f"\n{'='*60}")
            print(f"📊 토큰 사용량")
            print(f"{'='*60}")
            print(f"  입력 토큰  : {token_info['input_tokens']:>10,}")
            print(f"  출력 토큰  : {token_info['output_tokens']:>10,}")
            print(f"  {'─'*56}")
            print(f"  총 토큰    : {token_info['total_tokens']:>10,}")
            print(f"{'='*60}")
        print(f"\n📊 툴 호출 내역:")
        for i, tc in enumerate(result['tool_calls'], 1):
            args_str = str(tc['arguments'])
            # error_line 사용 여부 표시
            if tc['tool'] == 'read_file':
                if 'error_line' in tc['arguments'] and tc['arguments']['error_line']:
                    args_str += " ✅ (error_line 사용!)"
                else:
                    args_str += " ⚠️ (error_line 미사용 - 토큰 낭비!)"
            print(f"  {i}. {tc['tool']}({args_str})")

        print(f"\n📝 분석 결과:")
        print(result["analysis"][:1000] + "..." if len(result["analysis"]) > 1000 else result["analysis"])
        print("\n" + "="*80)

        return {
            "success": True,
            "file_locations": file_locations,
            "analysis": result["analysis"],
            "tool_calls": result["tool_calls"],
            "iterations": result["iterations"],
            "using_fastmcp": True
        }

    except Exception as e:
        print(f"\n❌ 에러 발생: {str(e)}")
        print("="*80)
        raise HTTPException(
            status_code=500,
            detail=f"에러 분석 중 문제 발생: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Error Debugger API (Local Files) 시작")
    print("   - AI 에이전트가 로컬 파일 직접 읽기")
    print("   - 여러 파일을 읽으며 비즈니스 로직 분석")
    print("   - OpenAI GPT-4o")
    print("   - Port: 9001 (로컬 파일 버전)")
    uvicorn.run(app, host="0.0.0.0", port=9001)
