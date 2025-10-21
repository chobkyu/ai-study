from fastmcp import FastMCP
import os
from pathlib import Path
from typing import List

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
    
# Resource 예시 (정적 데이터)
@mcp.resource("config://workspace")
def get_workspace_config() -> str:
    """현재 작업 공간 설정"""
    return f"Current workspace: {os.getcwd()}"

if __name__ == "__main__":
    # 서버 실행
    mcp.run()