from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class AnalysisCategory(str, Enum):
    """분석 카테고리"""
    GENERAL = "general"          # 일반 설명
    OBJECTS = "objects"          # 객체 탐지
    COLORS = "colors"            # 색상 분석
    TEXT = "text"                # 텍스트 추출
    SCENE = "scene"              # 장면 이해
    PEOPLE = "people"            # 사람 분석
    EMOTIONS = "emotions"        # 감정 분석
    RATING = "rating"            # 평가/점수 매기기

class AnalyzeRequest(BaseModel):
    """단일 이미지 분석 요청"""
    question: Optional[str] = Field(
        default="Describe this image in detail.",
        description="분석할 질문"
    )
    category: Optional[AnalysisCategory] = Field(
        default=AnalysisCategory.GENERAL,
        description="분석 카테고리"
    )
    max_tokens: Optional[int] = Field(
        default=100,
        ge=10,
        le=500,
        description="최대 생성 토큰 수"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="생성 온도"
    )

class AnalyzeResponse(BaseModel):
    """분석 결과"""
    question: str
    answer: str
    category: str
    processing_time: float

class BatchAnalyzeRequest(BaseModel):
    """배치 분석 요청"""
    questions: List[str] = Field(
        min_length=1,
        max_length=10,
        description="질문 리스트 (최대 10개)"
    )
    max_tokens: Optional[int] = Field(default=100)

class BatchAnalyzeResponse(BaseModel):
    """배치 분석 결과"""
    results: List[dict]
    total_processing_time: float

class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    model_loaded: bool
    device: str

class RatingRequest(BaseModel):
    """평가 요청"""
    rating_type: Optional[str] = Field(
        default="attractiveness",
        description="평가 유형: attractiveness(잘생김/예쁨), cuteness(귀여움), coolness(멋짐), style(스타일)"
    )
    scale: Optional[int] = Field(
        default=10,
        ge=5,
        le=100,
        description="평가 척도 (예: 10점 만점, 100점 만점)"
    )
    detailed: Optional[bool] = Field(
        default=True,
        description="상세 평가 포함 여부"
    )

class RatingResponse(BaseModel):
    """평가 결과"""
    rating_type: str
    score: str
    scale: int
    detailed_feedback: Optional[str] = None
    processing_time: float