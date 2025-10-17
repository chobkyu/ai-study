import requests
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """헬스체크 테스트"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    assert response.status_code == 200

def test_analyze_single(image_path: str):
    """단일 이미지 분석 테스트"""
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {
            "question": "이 이미지에 무엇이 보이나요?",
            "category": "general",
            "max_tokens": 150
        }
        
        response = requests.post(
            f"{BASE_URL}/analyze",
            files=files,
            data=data
        )
    
    print("\n=== Single Image Analysis ===")
    result = response.json()
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Processing Time: {result['processing_time']}s")
    
    assert response.status_code == 200

def test_analyze_categories(image_path: str):
    """카테고리별 분석 테스트"""
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {
            "categories": ["general", "objects", "colors", "scene"]
        }
        
        response = requests.post(
            f"{BASE_URL}/analyze/categories",
            files=files,
            data=data
        )
    
    print("\n=== Multi-Category Analysis ===")
    result = response.json()
    for category, analysis in result['results'].items():
        print(f"\n[{category.upper()}]")
        print(f"Q: {analysis['question']}")
        print(f"A: {analysis['answer']}")
    print(f"\nTotal Time: {result['processing_time']}s")
    
    assert response.status_code == 200

def test_batch_analyze(image_paths: list):
    """배치 분석 테스트"""
    files = [("files", open(path, "rb")) for path in image_paths]
    questions = [
        "What is in this image?",
        "Describe the main subject.",
        "What colors do you see?"
    ]
    
    data = {
        "questions": questions,
        "max_tokens": 100
    }
    
    response = requests.post(
        f"{BASE_URL}/analyze/batch",
        files=files,
        data=data
    )
    
    print("\n=== Batch Analysis ===")
    result = response.json()
    for item in result['results']:
        print(f"\n[Image {item['index']}: {item['filename']}]")
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
    print(f"\nTotal Time: {result['total_processing_time']}s")
    
    # 파일 닫기
    for _, f in files:
        f.close()
    
    assert response.status_code == 200

def test_rate_image(image_path: str):
    """이미지 평가 테스트"""
    rating_types = ["attractiveness", "cuteness", "coolness", "style"]

    print("\n=== Image Rating Tests ===")

    for rating_type in rating_types:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {
                "rating_type": rating_type,
                "scale": 10,
                "detailed": True
            }

            response = requests.post(
                f"{BASE_URL}/rate",
                files=files,
                data=data
            )

        if response.status_code == 200:
            result = response.json()
            print(f"\n[{rating_type.upper()}] (Scale: {result['scale']})")
            print(f"Score: {result['score']}")
            if result['detailed_feedback']:
                print(f"Feedback: {result['detailed_feedback']}")
            print(f"Processing Time: {result['processing_time']}s")
        else:
            print(f"Error rating {rating_type}: {response.text}")

    assert response.status_code == 200

def test_rate_image_simple(image_path: str):
    """간단한 점수만 테스트"""
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {
            "rating_type": "cuteness",
            "scale": 100,
            "detailed": False
        }

        response = requests.post(
            f"{BASE_URL}/rate",
            files=files,
            data=data
        )

    print("\n=== Simple Rating (Score Only) ===")
    result = response.json()
    print(f"Cuteness Score (out of {result['scale']}): {result['score']}")
    print(f"Processing Time: {result['processing_time']}s")

    assert response.status_code == 200

if __name__ == "__main__":
    # 테스트 이미지 경로
    image_path = "cat.17.jpg"

    if not Path(image_path).exists():
        print(f"Error: {image_path} not found")
        sys.exit(1)

    print("Starting API tests...")

    # 테스트 실행
    test_health()
    test_analyze_single(image_path)
    test_analyze_categories(image_path)

    # 평가 테스트 추가!
    test_rate_image(image_path)
    test_rate_image_simple(image_path)

    # 배치 테스트 (이미지 3개 필요)
    # test_batch_analyze([image_path, image_path, image_path])

    print("\n✅ All tests passed!")