from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from typing import Optional
import requests
from datetime import datetime
import math

# 계산기 도구
class CalculatorInput(BaseModel):
    expression: str = Field(description="계산할 수식 (예: 2+2, sqrt(16), sin(45))")

def calculator(expression: str) -> str:
    """수학 계산 도구"""
    try:
        # 안전한 eval
        allowed_names = {
            k: v for k, v in math.__dict__.items() 
            if not k.startswith("__")
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"

calculator_tool = StructuredTool.from_function(
    func=calculator,
    name="calculator",
    description="수학 계산을 수행합니다. 사칙연산, 삼각함수, 제곱근 등을 계산할 수 있습니다.",
    args_schema=CalculatorInput,
)

# 날씨 조회 도구 (더미)
class WeatherInput(BaseModel):
    location: str = Field(description="도시 이름 (예: 서울, 부산)")

def get_weather(location: str) -> str:
    """날씨 정보 조회 (더미 데이터)"""
    # 실제로는 API 호출
    # response = requests.get(f"https://api.weather.com/v1/location/{location}")
    
    # 더미 응답
    weather_data = {
        "서울": "맑음, 기온 15°C, 습도 60%",
        "부산": "흐림, 기온 18°C, 습도 70%",
        "제주": "비, 기온 12°C, 습도 85%",
    }
    
    return weather_data.get(
        location,
        f"{location}의 날씨 정보를 찾을 수 없습니다."
    )

weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="weather",
    description="특정 도시의 현재 날씨를 조회합니다.",
    args_schema=WeatherInput,
)

# 시간 조회 도구
@tool
def current_time() -> str:
    """현재 날짜와 시간을 조회합니다."""
    now = datetime.now()
    return now.strftime("%Y년 %m월 %d일 %H시 %M분 %S초")

time_tool = current_time

# 웹 검색 도구 (더미)
class SearchInput(BaseModel):
    query: str = Field(description="검색 쿼리")

def web_search(query: str) -> str:
    """웹 검색 (더미)"""
    # 실제로는 검색 API 호출
    return f"'{query}'에 대한 검색 결과: [여기에 실제 검색 결과가 표시됩니다]"

search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="웹에서 정보를 검색합니다. 최신 정보나 모르는 내용을 찾을 때 사용합니다.",
    args_schema=SearchInput,
)

# 모든 도구 리스트
ALL_TOOLS = [
    calculator_tool,
    weather_tool,
    time_tool,
    search_tool,
]