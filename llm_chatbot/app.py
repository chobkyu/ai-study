from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uuid
import os
from dotenv import load_dotenv
import logging

from chatbot.agent import ChatbotAgent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
     title="LLM Chatbot",
    description="LangGraph 기반 한국어 챗봇",
    version="1.0.0"
)

# 정적 파일
app.mount("/static", StaticFiles(directory="static"), name="static")

# 챗봇 초기화
chatbot = ChatbotAgent(
    model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    temperature=float(os.getenv("TEMPERATURE", "0.7")),
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
)


# 요청/응답 모델
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

# 세션 관리
active_sessions = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 API (일반 모드)"""
    try:
        # 세션 ID 생성 또는 사용
        session_id = request.session_id or str(uuid.uuid4())
        
        # 챗봇 응답
        response = await chatbot.chat(
            message=request.message,
            session_id=session_id,
            stream=False
        )
        
        return ChatResponse(
            response=response,
            session_id=session_id
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """채팅 API (스트리밍 모드)"""
    session_id = request.session_id or str(uuid.uuid4())
    
    async def generate():
        try:
            async for chunk in chatbot.chat(
                message=request.message,
                session_id=session_id,
                stream=True
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 채팅"""
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")
    
    try:
        while True:
            # 메시지 수신
            data = await websocket.receive_text()
            
            # 챗봇 응답
            response = await chatbot.chat(
                message=data,
                session_id=session_id,
                stream=False
            )
            
            # 응답 전송
            await websocket.send_text(response)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """세션 삭제"""
    chatbot.clear_history(session_id)
    return {"message": "Session cleared"}

@app.get("/session/{session_id}/stats")
async def get_session_stats(session_id: str):
    """세션 통계"""
    stats = chatbot.get_stats(session_id)
    return stats

@app.get("/health")
async def health():
    """헬스체크"""
    return {
        "status": "healthy",
        "model": os.getenv("MODEL_NAME"),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)