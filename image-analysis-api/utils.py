from PIL import Image
import hashlib
import io
from typing import Optional
import diskcache
import os

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)
cache = diskcache.Cache(CACHE_DIR)

def validate_image(image: Image.Image) -> tuple[bool, Optional[str]]:
    """이미지 검증"""
    if image.size[0] > 4096 or image.size[1] > 4096:
        return False, "Image dimensions too large (max 4096x4096)"
    
    # 최소 크기 체크
    if image.size[0] < 50 or image.size[1] < 50:
        return False, "Image too small (min 50x50)"
    
    return True, None

def get_image_hash(image: Image.Image) -> str:
    """이미지 해시 생성 (캐싱용)"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    return hashlib.md5(img_bytes).hexdigest()

def get_cached_result(image_hash: str, question: str) -> Optional[str]:
    """캐시에서 결과 가져오기"""
    cache_key = f"{image_hash}:{question}"
    return cache.get(cache_key)

def set_cached_result(image_hash: str, question: str, result: str, expire: int = 3600):
    """캐시에 결과 저장 (기본 1시간)"""
    cache_key = f"{image_hash}:{question}"
    cache.set(cache_key, result, expire=expire)

def get_category_question(category: str) -> str:
    """카테고리별 기본 질문"""
    questions = {
        "general": "Describe this image in detail.",
        "objects": "What objects can you see in this image? List them.",
        "colors": "What are the dominant colors in this image?",
        "text": "Is there any text in this image? If so, what does it say?",
        "scene": "Describe the scene and setting of this image.",
        "people": "Are there any people in this image? If so, describe them.",
        "emotions": "What emotions or mood does this image convey?"
    }
    return questions.get(category, questions["general"])

def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """이미지 리사이징 (메모리 절약)"""
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image