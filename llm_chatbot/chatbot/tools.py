from langchain_core.tools import tool
from typing import Annotated
from datetime import datetime
import math
import logging

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 계산기 도구
@tool
def calculator(expression: Annotated[str, "계산할 수식 (예: 2+2, sqrt(16), sin(45))"]) -> str:
    """수학 계산을 수행합니다. 사칙연산, 삼각함수, 제곱근 등을 계산할 수 있습니다."""
    logger.info(f"🧮 [CALCULATOR] 호출됨 - 수식: {expression}")
    try:
        # 안전한 eval
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("__")
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        logger.info(f"🧮 [CALCULATOR] 결과: {result}")
        return f"계산 결과: {result}"
    except Exception as e:
        logger.error(f"🧮 [CALCULATOR] 오류: {str(e)}")
        return f"계산 오류: {str(e)}"

calculator_tool = calculator

# 날씨 조회 도구 (더미)
@tool
def weather(location: Annotated[str, "도시 이름 (예: 서울, 부산)"]) -> str:
    """특정 도시의 현재 날씨를 조회합니다."""
    logger.info(f"🌤️  [WEATHER] 호출됨 - 위치: {location}")
    # 실제로는 API 호출
    # response = requests.get(f"https://api.weather.com/v1/location/{location}")

    # 더미 응답
    weather_data = {
        "서울": "맑음, 기온 15°C, 습도 60%",
        "부산": "흐림, 기온 18°C, 습도 70%",
        "제주": "비, 기온 12°C, 습도 85%",
    }

    result = weather_data.get(
        location,
        f"{location}의 날씨 정보를 찾을 수 없습니다."
    )
    logger.info(f"🌤️  [WEATHER] 결과: {result}")
    return result

weather_tool = weather

# 시간 조회 도구
@tool
def current_time() -> str:
    """현재 날짜와 시간을 조회합니다."""
    logger.info(f"⏰ [TIME] 호출됨")
    now = datetime.now()
    result = now.strftime("%Y년 %m월 %d일 %H시 %M분 %S초")
    logger.info(f"⏰ [TIME] 결과: {result}")
    return result

time_tool = current_time

# 웹 검색 도구 (더미)
@tool
def web_search(query: Annotated[str, "검색 쿼리"]) -> str:
    """웹에서 정보를 검색합니다. 최신 정보나 모르는 내용을 찾을 때 사용합니다."""
    logger.info(f"🔍 [SEARCH] 호출됨 - 쿼리: {query}")
    # 실제로는 검색 API 호출
    result = f"'{query}'에 대한 검색 결과: [여기에 실제 검색 결과가 표시됩니다]"
    logger.info(f"🔍 [SEARCH] 결과 반환 완료")
    return result

search_tool = web_search

# 모든 도구 리스트
ALL_TOOLS = [
    calculator_tool,
    weather_tool,
    time_tool,
    search_tool,
]