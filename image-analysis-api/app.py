from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import time
import logging
from typing import List, Optional

from model_manager import ModelManager
from schemas import (
    AnalyzeRequest, AnalyzeResponse,
    BatchAnalyzeResponse, HealthResponse,
    AnalysisCategory, RatingRequest, RatingResponse
)
from utils import (
    validate_image, get_image_hash,
    get_cached_result, set_cached_result,
    get_category_question, resize_image
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Image Analysis API",
    description="LLaVA 기반 이미지 분석 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 매니저 초기화
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로딩"""
    logger.info("Starting up...")
    try:
        # 맥북에서는 양자화 끄기 (MPS는 bitsandbytes 미지원)
        model_manager.load_model(use_quantization=False, use_mps=True)
        logger.info("Server ready!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/", response_model=dict)
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Image Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크"""
    import torch
    
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "unhealthy",
        model_loaded=model_manager.is_loaded,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(..., description="분석할 이미지 파일"),
    question: Optional[str] = Form(None, description="질문 (선택)"),
    category: Optional[str] = Form("general", description="분석 카테고리"),
    max_tokens: Optional[int] = Form(100, ge=10, le=500),
    temperature: Optional[float] = Form(0.7, ge=0.0, le=2.0),
    use_cache: Optional[bool] = Form(True, description="캐시 사용 여부")
):
    """
    단일 이미지 분석
    
    - **file**: 이미지 파일 (JPEG, PNG 등)
    - **question**: 분석할 질문 (없으면 카테고리 기본 질문 사용)
    - **category**: 분석 카테고리 (general, objects, colors 등)
    - **max_tokens**: 최대 생성 토큰 수 (10-500)
    - **temperature**: 생성 온도 (0.0-2.0)
    - **use_cache**: 캐시 사용 여부
    """
    start_time = time.time()
    
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 이미지 검증
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # 이미지 리사이징 (메모리 절약)
        image = resize_image(image)
        
        # 질문 결정
        if question is None:
            question = get_category_question(category)
        
        # 캐시 확인
        answer = None
        if use_cache:
            image_hash = get_image_hash(image)
            answer = get_cached_result(image_hash, question)
            if answer:
                logger.info("Cache hit!")
        
        # 캐시 미스 시 추론
        if answer is None:
            logger.info(f"Analyzing image with question: {question}")
            answer = model_manager.analyze_image(
                image=image,
                question=question,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 캐시 저장
            if use_cache:
                set_cached_result(image_hash, question, answer)
        
        processing_time = time.time() - start_time
        
        return AnalyzeResponse(
            question=question,
            answer=answer,
            category=category,
            processing_time=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def batch_analyze(
    files: List[UploadFile] = File(..., description="이미지 파일들 (최대 10개)"),
    questions: Optional[List[str]] = Form(None, description="질문 리스트"),
    max_tokens: Optional[int] = Form(100)
):
    """
    배치 이미지 분석
    
    - **files**: 이미지 파일 리스트 (최대 10개)
    - **questions**: 각 이미지에 대한 질문 리스트
    - **max_tokens**: 최대 생성 토큰 수
    """
    start_time = time.time()
    
    try:
        # 파일 개수 체크
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
        
        # 이미지 로드
        images = []
        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            is_valid, error_msg = validate_image(image)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)
            
            image = resize_image(image)
            images.append(image)
        
        # 질문 결정
        if questions is None:
            questions = ["Describe this image."] * len(images)
        elif len(questions) != len(images):
            raise HTTPException(
                status_code=400,
                detail=f"Number of questions ({len(questions)}) must match number of images ({len(images)})"
            )
        
        # 배치 분석
        logger.info(f"Batch analyzing {len(images)} images")
        answers = model_manager.batch_analyze(
            images=images,
            questions=questions,
            max_tokens=max_tokens
        )
        
        # 결과 구성
        results = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            results.append({
                "index": i,
                "filename": files[i].filename,
                "question": question,
                "answer": answer
            })
        
        total_time = time.time() - start_time
        
        return BatchAnalyzeResponse(
            results=results,
            total_processing_time=round(total_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/categories", response_model=dict)
async def analyze_categories(
    file: UploadFile = File(...),
    categories: List[str] = Form(
        default=["general", "objects", "colors"],
        description="분석할 카테고리 리스트"
    )
):
    """
    여러 카테고리로 한 이미지 분석
    
    - **file**: 이미지 파일
    - **categories**: 분석할 카테고리 리스트
    """
    start_time = time.time()
    
    try:
        # 이미지 로드
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        image = resize_image(image)
        
        # 각 카테고리별 분석
        results = {}
        for category in categories:
            question = get_category_question(category)
            answer = model_manager.analyze_image(
                image=image,
                question=question,
                max_tokens=100
            )
            results[category] = {
                "question": question,
                "answer": answer
            }
        
        total_time = time.time() - start_time
        
        return {
            "results": results,
            "processing_time": round(total_time, 2)
        }
    
    except Exception as e:
        logger.error(f"Error analyzing categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rate", response_model=RatingResponse)
async def rate_image(
    file: UploadFile = File(..., description="평가할 이미지 파일"),
    rating_type: Optional[str] = Form("attractiveness", description="평가 유형"),
    scale: Optional[int] = Form(10, ge=5, le=100, description="평가 척도"),
    detailed: Optional[bool] = Form(True, description="상세 평가 여부")
):
    """
    이미지 평가 API

    - **file**: 평가할 이미지 파일
    - **rating_type**: 평가 유형
      - attractiveness: 잘생김/예쁨
      - cuteness: 귀여움
      - coolness: 멋짐/쿨함
      - style: 스타일/패션
    - **scale**: 평가 척도 (5~100점)
    - **detailed**: 상세 평가 포함 여부
    """
    start_time = time.time()

    try:
        # 이미지 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 이미지 검증
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        image = resize_image(image)

        # 평가 유형에 따른 질문 생성
        rating_questions = {
            "attractiveness": f"Rate the attractiveness or beauty in this image on a scale of 1 to {scale}. Give a specific score and explain why.",
            "cuteness": f"Rate how cute the subject in this image is on a scale of 1 to {scale}. Provide a score and reasoning.",
            "coolness": f"Rate the coolness factor of this image on a scale of 1 to {scale}. Give a score and explain.",
            "style": f"Rate the style and fashion in this image on a scale of 1 to {scale}. Provide a score and detailed feedback."
        }

        question = rating_questions.get(
            rating_type,
            f"Rate this image on a scale of 1 to {scale}. Give a specific score."
        )

        if not detailed:
            question = question.split("Provide")[0].split("Give")[0] + f"Just give me a number from 1 to {scale}."

        # 평가 실행
        logger.info(f"Rating image: {rating_type} (scale: {scale})")
        answer = model_manager.analyze_image(
            image=image,
            question=question,
            max_tokens=150 if detailed else 50,
            temperature=0.3  # 평가는 일관성을 위해 낮은 temperature
        )

        processing_time = time.time() - start_time

        return RatingResponse(
            rating_type=rating_type,
            score=answer,
            scale=scale,
            detailed_feedback=answer if detailed else None,
            processing_time=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"Error rating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache/clear")
async def clear_cache():
    """캐시 전체 삭제"""
    from utils import cache
    cache.clear()
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)