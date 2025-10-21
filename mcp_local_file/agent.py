# agent.py
import os
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MCP 서버의 도구들을 OpenAI Function으로 변환
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "로컬 파일의 내용을 읽습니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "읽을 파일의 경로"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "디렉토리의 파일과 폴더 목록을 보여줍니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "조회할 디렉토리 경로"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "특정 디렉토리에서 파일 내용을 검색합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "검색할 텍스트"
                    },
                    "directory": {
                        "type": "string",
                        "description": "검색할 디렉토리 경로"
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "검색할 파일 확장자"
                    }
                },
                "required": ["search_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_info",
            "description": "파일의 메타데이터 정보를 가져옵니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "파일 경로"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "이미지를 분석합니다 (사물, 장면, 텍스트 등 인식)",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "이미지 파일 경로"
                    },
                    "question": {
                        "type": "string",
                        "description": "이미지에 대한 질문",
                        "default": "이 이미지에 무엇이 있나요?"
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_images",
            "description": "두 이미지를 비교합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path1": {"type": "string"},
                    "image_path2": {"type": "string"},
                    "question": {"type": "string"}
                },
                "required": ["image_path1", "image_path2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_text_from_image",
            "description": "이미지에서 텍스트를 추출합니다 (OCR)",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"}
                },
                "required": ["image_path"]
            }
        }
    }
]

class FileAgent:
    def __init__(self):
        self.conversation_history = []
        # MCP 서버 도구들을 실제 함수로 매핑
        from file_server import read_file, list_directory, search_in_files, get_file_info, analyze_image, compare_images, extract_text_from_image
        self.tool_functions = {
            "read_file": read_file,
            "list_directory": list_directory,
            "search_in_files": search_in_files,
            "get_file_info": get_file_info,
            "analyze_image": analyze_image,
            "compare_images": compare_images,
            "extract_text_from_image": extract_text_from_image
        }
    
    def _call_tool(self, function_name: str, function_args: dict) -> str:
        """도구 실행 (이미지 처리 포함)"""
        
        result = self.tool_functions[function_name](**function_args)
        
        # 이미지 분석 결과 처리
        if result.startswith("IMAGE_DATA:"):
            return self._handle_image_analysis(result)
        elif result.startswith("COMPARE_IMAGES:"):
            return self._handle_image_comparison(result)
        elif result.startswith("EXTRACT_TEXT:"):
            return self._handle_text_extraction(result)
        
        return result
    
    def _handle_image_analysis(self, result: str) -> str:
        """이미지 분석 처리"""
        try:
            # 파싱: IMAGE_DATA:mime_type:base64_data|QUESTION:question
            parts = result.split("|QUESTION:")
            image_part = parts[0].replace("IMAGE_DATA:", "")
            question = parts[1] if len(parts) > 1 else "이 이미지에 무엇이 있나요?"
            
            mime_type, image_data = image_part.split(":", 1)
            
            # GPT-4o Vision 호출
            response = client.chat.completions.create(
                model="gpt-4.1-mini",  # 또는 "gpt-4o-mini"
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"이미지 분석 실패: {str(e)}"
    
    def _handle_image_comparison(self, result: str) -> str:
        """이미지 비교 처리"""
        try:
            parts = result.split("|")
            image1_data = parts[0].replace("COMPARE_IMAGES:", "")
            image2_data = parts[1]
            question = parts[2].replace("QUESTION:", "") if len(parts) > 2 else "차이점은?"
            
            mime1, data1 = image1_data.split(":", 1)
            mime2, data2 = image2_data.split(":", 1)
            
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime1};base64,{data1}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime2};base64,{data2}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"이미지 비교 실패: {str(e)}"
    
    def _handle_text_extraction(self, result: str) -> str:
        """OCR 처리"""
        try:
            parts = result.replace("EXTRACT_TEXT:", "").split(":", 1)
            mime_type, image_data = parts
            
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "이 이미지에 있는 모든 텍스트를 정확하게 추출해주세요. 텍스트만 출력하세요."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"텍스트 추출 실패: {str(e)}"
        
    def chat(self, user_message: str) -> str:
        """사용자 메시지 처리"""

        # 대화 히스토리에 추가
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # 최근 5개만 유지 (토큰 절약)
        recent_history = self.conversation_history[-10:]  # 최근 10개 (user+assistant 쌍 5개)

        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # gpt-4 대신 gpt-4o-mini 사용 (더 저렴하고 빠름)
            messages=[
                {
                    "role": "system",
                    "content": """당신은 로컬 파일 시스템을 탐색하고 분석하는 AI 어시스턴트입니다.

사용자가 파일이나 디렉토리에 대해 질문하면:
1. 적절한 도구를 선택해서 사용하세요
2. 결과를 분석해서 사용자에게 유용한 답변을 제공하세요
3. 여러 도구를 조합해서 사용할 수도 있습니다

주의사항:
- 파일 경로는 정확해야 합니다
- 보안을 위해 시스템 파일은 읽지 마세요
- 사용자가 요청한 내용만 처리하세요"""
                }
            ] + recent_history,  # 전체 히스토리 대신 최근 것만!
            tools=TOOLS,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # 도구 호출이 필요한 경우
        if tool_calls:
            # 어시스턴트 응답을 히스토리에 추가
            self.conversation_history.append(response_message)
            
            # 각 도구 호출 실행
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"\n🔧 도구 사용: {function_name}")
                print(f"   인자: {function_args}")
                
                # 도구 실행
                # FunctionTool 객체는 .fn 속성으로 실제 함수에 접근
                tool = self.tool_functions[function_name]
                if hasattr(tool, 'fn'):
                    # FastMCP의 FunctionTool인 경우
                    function_response = tool.fn(**function_args)
                else:
                    # 일반 함수인 경우
                    function_response = tool(**function_args)
                
                # 도구 결과를 히스토리에 추가
                self.conversation_history.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })
            
            # 도구 결과를 포함해서 다시 GPT 호출
            second_response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "당신은 로컬 파일 시스템을 탐색하는 AI 어시스턴트입니다."}
                ] + self.conversation_history
            )
            
            final_message = second_response.choices[0].message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": final_message
            })
            
            return final_message
        
        else:
            # 도구 호출 없이 바로 답변
            self.conversation_history.append({
                "role": "assistant",
                "content": response_message.content
            })
            return response_message.content

def main():
    agent = FileAgent()
    
    print("=" * 60)
    print("🤖 로컬 파일 AI Agent")
    print("=" * 60)
    print("로컬 파일을 읽고 검색할 수 있는 AI 어시스턴트입니다.")
    print("'quit' 또는 'exit'를 입력하면 종료됩니다.\n")
    
    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("👋 안녕히 가세요!")
            break
        
        response = agent.chat(user_input)
        print(f"\n🤖 Agent: {response}")

if __name__ == "__main__":
    main()