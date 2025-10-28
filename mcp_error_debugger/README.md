# 🔍 Error Debugger API

서버에서 발생한 에러를 HTTP POST로 전송하면 **OpenAI GPT가 자동으로 분석**하고 원인과 해결 방법을 제안하는 FastAPI 서버입니다.

## 🤖 특징
- **GPT-4o 기반 분석** - 단순 패턴 매칭이 아닌 실제 코드와 컨텍스트를 이해하는 AI 분석
- **실제 코드 읽기** - 에러 발생 위치의 실제 코드를 읽어서 GPT에게 전달
- **상세한 해결 방법** - 구체적인 코드 예시와 함께 해결책 제시
- **입력 파라미터 분석** - 어떤 입력이 문제를 일으켰는지 파악
- **FastAPI + FastMCP** - HTTP API와 MCP 프로토콜 모두 지원

## ✨ 주요 기능

### 1. **POST /analyze** - 종합 에러 분석 ⭐
에러 객체를 JSON으로 받아서 종합적으로 분석합니다.

**Request Body:**
```json
{
  "error_type": "AttributeError",
  "error_message": "'CLIPVisionModel' object has no attribute 'vision_tower'",
  "stack_trace": "File \"/path/to/file.py\", line 84, in load_model\n...",
  "input_params": "{\"use_quantization\": false, \"use_mps\": true}",
  "server_base_path": "/Users/fanding/develop/ppp/image-analysis-api"
}
```

**Response:**
```json
{
  "success": true,
  "analysis": "🔍 에러 분석 결과\n...",
  "suggestions": [
    "객체의 dir() 또는 __dict__를 출력하여 실제 속성을 확인하세요",
    "오타가 있는지 확인하세요 (예: vision_tower vs vision_model)"
  ]
}
```

**분석 내용:**
- ✅ 에러 발생 위치 (파일:라인:함수)
- ✅ 입력 파라미터 분석 및 문제점
- ✅ 에러 원인 분석
- ✅ 관련 코드 컨텍스트 표시
- ✅ 해결 방법 제안

### 2. **POST /read-context** - 코드 컨텍스트 읽기
에러 발생 위치 주변 코드를 읽습니다.

**Request Body:**
```json
{
  "file_path": "/path/to/file.py",
  "line_number": 84,
  "context_lines": 10
}
```

### 3. **POST /search** - 에러 패턴 검색
프로젝트에서 유사한 에러나 관련 코드를 검색합니다.

**Request Body:**
```json
{
  "error_message": "AttributeError vision_tower",
  "search_path": "/Users/fanding/develop/ppp",
  "file_extensions": [".py", ".js"]
}
```

### 4. **POST /trace-variable** - 변수 흐름 추적
특정 변수의 선언, 할당, 사용을 추적합니다.

**Request Body:**
```json
{
  "file_path": "/path/to/file.py",
  "variable_name": "vision_tower",
  "start_line": 84
}
```

## 🚀 사용 방법

### 1. 환경 설정
```bash
cd mcp_error_debugger
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# .env 파일 생성
cp .env.example .env
# .env 파일에 OpenAI API 키 입력
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 2. 서버 시작

**옵션 1: FastAPI 서버** (HTTP API)
```bash
python app.py  # 포트 9000
```

**옵션 2: FastMCP 서버** (MCP 프로토콜)
```bash
python mcp_server.py
```

### 3. Swagger UI로 테스트
브라우저에서 접속:
- **Swagger UI**: http://localhost:9000/docs
- **ReDoc**: http://localhost:9000/redoc

### 3. Python 클라이언트로 테스트
```bash
python test_api.py
```

### 4. 실제 서버에서 사용
```python
import requests
import traceback
import json

def your_server_function(param1, param2):
    try:
        # 서버 로직
        result = some_operation(param1, param2)
        return result
    except Exception as e:
        # 에러 발생 시 디버거 API로 분석 요청
        error_data = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "stack_trace": traceback.format_exc(),
            "input_params": json.dumps({
                "param1": param1,
                "param2": param2
            }),
            "server_base_path": "/path/to/your/server"
        }

        # 디버거 API 호출
        response = requests.post(
            "http://localhost:9000/analyze",
            json=error_data,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("🔍 에러 분석 결과:")
            print(result['analysis'])

            # 로그에 기록
            logger.error(f"Error analysis: {result['analysis']}")

            # 슬랙/이메일 알림
            send_alert(result['analysis'])

        # 에러 재발생 또는 처리
        raise
```

## 🎯 실제 사용 예시

### 예시 1: FastAPI 서버에 미들웨어로 추가
```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import traceback
import json

app = FastAPI()

@app.middleware("http")
async def error_debugger_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # 에러 분석
        error_data = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "stack_trace": traceback.format_exc(),
            "input_params": json.dumps(dict(request.query_params)),
            "server_base_path": "/path/to/server"
        }

        # 디버거 API로 분석 (비동기로 백그라운드에서)
        try:
            debug_response = requests.post(
                "http://localhost:9000/analyze",
                json=error_data,
                timeout=5
            )
            if debug_response.status_code == 200:
                analysis = debug_response.json()
                # 로그에 상세 분석 기록
                print(f"🔍 에러 분석:\n{analysis['analysis']}")
        except:
            pass  # 디버거 실패해도 원본 에러는 처리

        # 원본 에러 반환
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
```

### 예시 2: 개발 환경에서 자동 디버깅
```python
import os
import requests
import traceback
import json

def auto_debug_on_error(func):
    """데코레이터: 함수 실행 중 에러 발생 시 자동 분석"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 개발 환경에서만 자동 디버깅
            if os.getenv("ENV") == "development":
                error_data = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "stack_trace": traceback.format_exc(),
                    "input_params": json.dumps({
                        "args": str(args),
                        "kwargs": str(kwargs)
                    })
                }

                response = requests.post(
                    "http://localhost:9000/analyze",
                    json=error_data,
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    print("\n" + "="*60)
                    print("🔍 자동 에러 분석")
                    print("="*60)
                    print(result['analysis'])
                    print("="*60 + "\n")

            raise  # 원본 에러 재발생

    return wrapper


@auto_debug_on_error
def load_model(model_path, use_mps=True):
    # 모델 로딩 로직
    model = load_from_disk(model_path)
    if use_mps:
        model = model.to("mps")
    return model
```

## 📊 지원하는 에러 타입

- ✅ **AttributeError** - 속성 접근 에러
- ✅ **TypeError** - 타입 관련 에러 (NoneType 연산 등)
- ✅ **ValueError** - 잘못된 값 에러
- ✅ **KeyError** - 딕셔너리 키 에러
- ✅ **IndexError** - 인덱스 범위 에러
- ✅ **기타** - 모든 Python 에러

## 🛠️ API 스펙

### Base URL
```
http://localhost:9000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | 서버 정보 |
| POST | `/analyze` | 종합 에러 분석 |
| POST | `/read-context` | 코드 컨텍스트 읽기 |
| POST | `/search` | 에러 패턴 검색 |
| POST | `/trace-variable` | 변수 흐름 추적 |

## 💡 고급 활용

### 1. 슬랙 알림 연동
```python
def send_error_to_slack(analysis_result):
    webhook_url = "YOUR_SLACK_WEBHOOK"
    message = {
        "text": "🚨 에러 발생!",
        "attachments": [{
            "color": "danger",
            "text": analysis_result['analysis']
        }]
    }
    requests.post(webhook_url, json=message)
```

### 2. 에러 히스토리 저장
```python
# 에러 분석 결과를 DB에 저장
def save_error_analysis(error_data, analysis_result):
    db.errors.insert_one({
        "timestamp": datetime.now(),
        "error_type": error_data['error_type'],
        "error_message": error_data['error_message'],
        "analysis": analysis_result['analysis'],
        "suggestions": analysis_result['suggestions'],
        "resolved": False
    })
```

### 3. CI/CD 파이프라인 통합
```yaml
# GitHub Actions 예시
- name: Run tests with error debugging
  run: |
    pytest --tb=long 2>&1 | tee test_output.txt || true
    python scripts/analyze_test_failures.py test_output.txt
```

## 🔐 보안 고려사항

⚠️ **주의**: 이 API는 로컬 개발 환경용입니다.

프로덕션에서 사용할 경우:
- [ ] 인증/인가 추가 (API Key, JWT 등)
- [ ] Rate limiting 적용
- [ ] 민감한 정보 필터링 (비밀번호, API Key 등)
- [ ] HTTPS 사용
- [ ] 파일 접근 권한 제한

## 📈 성능

- 평균 응답 시간: **100-300ms**
- 동시 요청 처리: **uvicorn workers로 확장 가능**
- 파일 검색: **프로젝트 크기에 비례**

## 🔗 연관 프로젝트

- [mcp_local_file](../mcp_local_file/) - 로컬 파일 시스템 접근 MCP
- [image-analysis-api](../image-analysis-api/) - LLaVA 이미지 분석 API

## 🤝 Contributing

개선 아이디어나 버그 리포트 환영합니다!

## 📝 라이선스

MIT License
