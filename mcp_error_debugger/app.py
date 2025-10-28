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
    title="Error Debugger API with AI Agent",
    description="AI 에이전트가 FastMCP 툴로 여러 파일을 읽으며 비즈니스 로직까지 분석",
    version="3.0.0"
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

def read_file(file_path: str) -> str:
    """
    파일의 전체 내용을 읽습니다.

    Args:
        file_path: 읽을 파일의 경로 (절대경로 또는 상대경로)

    Returns:
        파일 내용
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
            content = f.read()

        return content

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


def grep_code(file_path: str, search_term: str) -> str:
    """
    파일에서 특정 코드나 함수를 검색합니다.

    Args:
        file_path: 검색할 파일 경로
        search_term: 검색할 코드 (함수명, 클래스명, 변수명 등)

    Returns:
        검색 결과 (라인 번호와 내용)
    """
    try:
        content = read_file(file_path)

        if content.startswith("ERROR"):
            return content

        lines = content.split('\n')
        results = []

        for i, line in enumerate(lines, 1):
            if search_term.lower() in line.lower():
                results.append(f"Line {i}: {line.strip()}")

        if not results:
            return f"'{search_term}'을(를) 찾을 수 없습니다."

        return "\n".join(results[:20])  # 최대 20개

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
                "description": "파일의 전체 내용을 읽습니다. 에러가 발생한 파일이나 관련된 다른 파일들을 읽을 때 사용하세요.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "읽을 파일의 경로 (절대경로 또는 상대경로)"
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
        return read_file(arguments["file_path"])
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
    input_params: Optional[str] = None,
    server_base_path: str = "/Users/fanding/develop/legacy-php-api"
) -> dict:
    """
    AI 에이전트가 FastMCP 툴을 사용하며 비즈니스 로직까지 분석합니다.
    """

    # 초기 컨텍스트
    initial_context = f"""당신은 전문 소프트웨어 디버거이자 코드 분석가입니다.

## 에러 정보
- **에러 타입**: {error_type}
- **에러 메시지**: {error_message}

## 스택 트레이스
```
{stack_trace}
```

## 입력 파라미터
{input_params if input_params else "없음"}

## 에러 발생 파일들
{json.dumps(file_locations, indent=2, ensure_ascii=False)}

## 서버 경로
{server_base_path}

## 당신의 임무
1. 에러가 발생한 파일을 read_file로 읽어서 분석하세요
2. 관련된 다른 파일들도 read_file로 읽어서 비즈니스 로직을 파악하세요
3. 함수 호출 흐름을 추적하세요
4. 필요하면 search_files로 관련 파일들을 찾으세요
5. 필요하면 grep_code로 특정 함수나 클래스를 찾으세요
6. 필요하면 list_directory로 프로젝트 구조를 파악하세요

## 분석 결과 형식
최종적으로 다음 형식으로 분석 결과를 제공하세요:

### 🔍 에러 분석

#### 1. 에러 발생 위치와 원인
- 정확히 어디서 왜 에러가 발생했는지

#### 2. 비즈니스 로직 분석
- 에러가 발생한 코드의 비즈니스 목적
- 어떤 흐름으로 이 코드가 실행되었는지
- 관련된 다른 파일/함수들의 역할

#### 3. 근본 원인 (Root Cause)
- 단순히 코드 에러가 아니라, 왜 이런 상황이 발생했는지

#### 4. 해결 방법
- 구체적인 수정 방법 (코드 예시 포함)
- 비즈니스 로직을 고려한 해결 방안

#### 5. 예방 방법
- 이런 에러를 미리 방지하는 방법

**중요**: 반드시 FastMCP 툴들(read_file, search_files, grep_code, list_directory)을 적극 활용하세요!
"""

    messages = [
        {"role": "system", "content": "당신은 FastMCP 툴을 사용할 수 있는 전문 디버거입니다. 여러 파일을 읽으며 비즈니스 로직까지 깊이 분석합니다."},
        {"role": "user", "content": initial_context}
    ]

    tool_calls_history = []
    max_iterations = 15  # 최대 15번의 툴 호출

    for iteration in range(max_iterations):
        try:
            # OpenAI API 호출
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=get_openai_tools(),
                tool_choice="auto",
                temperature=0.3,
                max_tokens=4000
            )

            assistant_message = response.choices[0].message

            # 툴 호출이 없으면 최종 답변
            if not assistant_message.tool_calls:
                return {
                    "analysis": assistant_message.content,
                    "tool_calls": tool_calls_history,
                    "iterations": iteration + 1
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

                # 결과를 메시지에 추가
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result
                })

        except Exception as e:
            return {
                "analysis": f"AI 에이전트 분석 중 에러 발생: {str(e)}",
                "tool_calls": tool_calls_history,
                "iterations": iteration + 1,
                "error": str(e)
            }

    # 최대 반복 도달
    return {
        "analysis": "최대 반복 횟수에 도달했습니다. 부분 분석 결과를 확인하세요.",
        "tool_calls": tool_calls_history,
        "iterations": max_iterations
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

        # 2. AI 에이전트가 FastMCP 툴을 사용하며 분석
        print("\n🤖 AI 에이전트 분석 시작...")
        result = await _analyze_with_ai_agent(
            error_type=request.error_type,
            error_message=request.error_message,
            stack_trace=request.stack_trace,
            file_locations=file_locations,
            input_params=request.input_params,
            server_base_path=request.server_base_path
        )

        print(f"\n✅ 분석 완료!")
        print(f"  - 툴 호출 횟수: {len(result['tool_calls'])}회")
        print(f"  - 반복 횟수: {result['iterations']}회")
        print(f"\n📊 툴 호출 내역:")
        for i, tc in enumerate(result['tool_calls'], 1):
            print(f"  {i}. {tc['tool']}({tc['arguments']})")

        print(f"\n📝 분석 결과:")
        print(result["analysis"][:500] + "..." if len(result["analysis"]) > 500 else result["analysis"])
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
    print("🚀 Error Debugger API with AI Agent + FastMCP 시작")
    print("   - AI 에이전트가 FastMCP 툴 사용")
    print("   - 여러 파일을 읽으며 비즈니스 로직 분석")
    print("   - OpenAI GPT-4o")
    print("   - Port: 9000")
    uvicorn.run(app, host="0.0.0.0", port=9000)
