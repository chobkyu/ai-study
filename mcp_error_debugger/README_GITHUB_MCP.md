# GitHub MCP 통합 가이드

## 🎯 개요

이 에러 디버거는 GitHub MCP 서버를 통해 GitHub 저장소의 코드를 직접 읽을 수 있습니다.

## 📦 설치

```bash
cd /Users/fanding/develop/ppp/mcp_error_debugger
pip install -r requirements.txt
```

필수 라이브러리:
- `langchain-mcp-adapters` - LangChain과 MCP 연결
- `mcp` - Model Context Protocol 클라이언트
- `npx` - GitHub MCP 서버 실행 (Node.js 필요)

## ⚙️ 설정

### 1. GitHub Personal Access Token 생성

1. GitHub 설정으로 이동: https://github.com/settings/tokens
2. "Generate new token (classic)" 클릭
3. 권한 선택:
   - `repo` (전체 저장소 접근)
   - `read:org` (조직 저장소 읽기)
4. 토큰 복사

### 2. 환경 변수 설정

`.env` 파일에 다음 추가:

```bash
OPENAI_API_KEY=your-openai-api-key

# GitHub MCP 설정
GITHUB_TOKEN=ghp_your_token_here
GITHUB_REPO_OWNER=fanding
GITHUB_REPO_NAME=legacy-php-api
```

### 3. Node.js 설치 (npx 필요)

```bash
# macOS
brew install node

# 확인
npx --version
```

## 🚀 실행

```bash
python app.py
```

서버 시작 시 다음 로그를 확인:

```
================================================================================
🚀 Error Debugger API (LangGraph) 시작
================================================================================
🔌 GitHub MCP 서버 연결 중... (repo: fanding/legacy-php-api)
✅ GitHub MCP 도구 N개 로드됨
================================================================================
```

## 🔧 동작 방식

### GitHub MCP 도구 사용 가능한 경우:
1. 에이전트가 GitHub 저장소에서 직접 파일 읽기
2. 실시간 코드 분석
3. 로컬 파일 없이도 분석 가능

### GitHub MCP 미설정 시:
1. 로컬 파일 시스템 fallback
2. `/Users/fanding/develop/legacy-php-api` 경로에서 읽기
3. 경로 자동 변환 (`/home/fanding` → `/Users/fanding`)

## 📊 사용 가능한 GitHub MCP 도구

- `search_repositories` - 저장소 검색
- `create_or_update_file` - 파일 생성/수정
- `search_code` - 코드 검색
- `create_issue` - 이슈 생성
- `create_pull_request` - PR 생성
- 기타 GitHub API 기능

## 🐛 문제 해결

### GitHub MCP 연결 실패
```
❌ GitHub MCP 초기화 실패: ...
⚠️  GITHUB_TOKEN이 설정되지 않아 로컬 파일 시스템만 사용합니다.
```

**해결:**
1. `.env` 파일에 `GITHUB_TOKEN` 확인
2. Node.js 및 npx 설치 확인: `npx --version`
3. GitHub 토큰 권한 확인

### npx 없음
```bash
# macOS
brew install node

# Ubuntu/Debian
sudo apt install nodejs npm
```

## 📝 테스트

```bash
curl -X POST http://localhost:9000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "error_type": "TypeError",
    "error_message": "Call to member function on null",
    "stack_trace": "#0 /home/fanding/application/controllers/rest/Post.php(828)",
    "input_params": "user_id=123"
  }'
```

GitHub MCP가 활성화되면 로그에서 확인:
```
[DEBUG] Using GitHub MCP tool: read_repository_file
```

## 🎉 장점

- ✅ 로컬 파일 복사 불필요
- ✅ 여러 저장소 동시 접근
- ✅ 실시간 코드 읽기
- ✅ GitHub API 기능 활용
- ✅ Fallback으로 로컬 파일도 지원
