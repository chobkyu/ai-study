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
    
    # 배치 테스트 (이미지 3개 필요)
    # test_batch_analyze([image_path, image_path, image_path])
    
    print("\n✅ All tests passed!")