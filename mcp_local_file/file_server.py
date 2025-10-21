from fastmcp import FastMCP
import os
from pathlib import Path
from typing import List
from PIL import Image
import io

mcp = FastMCP("Local File Server")

@mcp.tool()
def read_file(file_path: str) -> str:
    """
    로컬 파일의 내용을 읽습니다.
    
    Args:
        file_path: 읽을 파일의 경로
    
    Returns:
        파일 내용 (텍스트)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"
    
@mcp.tool()
def list_directory(directory_path: str = ".") -> str:
    """
      디렉토리의 파일과 폴더 목록을 보여줍니다.
    
    Args:
        directory_path: 조회할 디렉토리 경로 (기본값: 현재 디렉토리)
    
    Returns:
        파일 및 폴더 목록
    """
    try:
        items = os.listdir(directory_path)
        result = []

        for item in items:
            full_path = os.path.join(directory_path, item)
            item_type = "📁" if os.path.isdir(full_path) else "📄"
            result.append(f"{item_type} {item}")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error listing directory: {str(e)}"
    
@mcp.tool()
def search_in_files(
    search_query: str, 
    directory: str = ".",
    file_extensions: List[str] = None
) -> str:
    """
    특정 디렉토리에서 파일 내용을 검색합니다.
    
    Args:
        search_query: 검색할 텍스트
        directory: 검색할 디렉토리 경로
        file_extensions: 검색할 파일 확장자 리스트 (예: ['.py', '.txt'])
    
    Returns:
        검색 결과 (파일 경로와 매칭된 줄)
    """
    if file_extensions is None:
        file_extensions = ['.txt', '.md', '.py', '.js', '.json']
    
    results =[]

    try: 
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for line_num, line in enumerate(lines, 1):
                                if search_query.lower() in line.lower():
                                    results.append(
                                        f"{file_path}:{line_num} - {line.strip()}"
                                    )
                    except:
                        continue
        
        if results:
            return "\n".join(results[:20])  # 상위 20개만
        else:
            return f"'{search_query}'에 대한 검색 결과가 없습니다."
        
    except Exception as e:
         return f"Error searching files: {str(e)}"
    
@mcp.tool()
def get_file_info(file_path: str) -> str:
    """
    파일의 메타데이터 정보를 가져옵니다.
    
    Args:
        file_path: 파일 경로
    
    Returns:
        파일 크기, 생성일, 수정일 등의 정보
    """

    try:
        stat = os.stat(file_path)
        size_kb = stat.st_size / 1024

        import datetime
        created = datetime.datetime.fromtimestamp(stat.st_ctime)
        modified = datetime.datetime.fromtimestamp(stat.st_mtime)

        info = f"""
파일 정보:
- 경로: {file_path}
- 크기: {size_kb:.2f} KB
- 생성일: {created.strftime('%Y-%m-%d %H:%M:%S')}
- 수정일: {modified.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return info
    except Exception as e:
        return f"Error getting file info: {str(e)}"
    
import base64
from pathlib import Path

@mcp.tool()
def analyze_image(image_path: str, question: str = "이 이미지에 무엇이 있나요?") -> str:
    """
    이미지를 분석합니다.
    
    Args:
        image_path: 이미지 파일 경로 (.jpg, .jpeg, .png, .gif, .webp)
        question: 이미지에 대해 물어볼 질문
    
    Returns:
        이미지 분석 결과
    """
    try:
        # 파일 확장자 확인
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }

        if ext not in mime_types:
            return f"지원하지 않는 이미지 형식입니다: {ext}"

        mime_type = mime_types[ext]

        # 이미지 리사이징 (토큰 절약)
        img = Image.open(image_path)

        # 최대 크기 제한 (512x512)
        max_size = 512
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # JPEG로 변환 (용량 감소)
        buffer = io.BytesIO()
        img.convert('RGB').save(buffer, format='JPEG', quality=85)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime_type = 'image/jpeg'

        # 이미지 정보와 질문을 반환 (Agent에서 GPT-4o Vision으로 처리)
        return f"IMAGE_DATA:{mime_type}:{image_data}|QUESTION:{question}"
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

@mcp.tool()
def compare_images(image_path1: str, image_path2: str, question: str = "이 두 이미지의 차이점은?") -> str:
    """
    두 이미지를 비교합니다.
    
    Args:
        image_path1: 첫 번째 이미지 경로
        image_path2: 두 번째 이미지 경로
        question: 비교에 대한 질문
    
    Returns:
        비교 결과
    """
    try:
        images_data = []
        for path in [image_path1, image_path2]:
            # 이미지 리사이징 (토큰 절약)
            img = Image.open(path)

            # 최대 크기 제한 (512x512)
            max_size = 512
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # JPEG로 변환
            buffer = io.BytesIO()
            img.convert('RGB').save(buffer, format='JPEG', quality=85)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            mime_type = 'image/jpeg'
            images_data.append(f"{mime_type}:{image_data}")

        return f"COMPARE_IMAGES:{images_data[0]}|{images_data[1]}|QUESTION:{question}"
        
    except Exception as e:
        return f"Error comparing images: {str(e)}"

@mcp.tool()
def extract_text_from_image(image_path: str) -> str:
    """
    이미지에서 텍스트를 추출합니다 (OCR).
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        추출된 텍스트
    """
    try:
        # 이미지 리사이징 (토큰 절약)
        img = Image.open(image_path)

        # 최대 크기 제한 (512x512)
        max_size = 512
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # JPEG로 변환
        buffer = io.BytesIO()
        img.convert('RGB').save(buffer, format='JPEG', quality=85)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime_type = 'image/jpeg'

        return f"EXTRACT_TEXT:{mime_type}:{image_data}"
        
    except Exception as e:
        return f"Error extracting text: {str(e)}"
    
    
# Resource 예시 (정적 데이터)
@mcp.resource("config://workspace")
def get_workspace_config() -> str:
    """현재 작업 공간 설정"""
    return f"Current workspace: {os.getcwd()}"

if __name__ == "__main__":
    # 서버 실행
    mcp.run()