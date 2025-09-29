from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os
from typing import List
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import openai

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. LLM 준비
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# 2. 로컬 파일 다루기
@app.get("/list_files")
def list_files(directory: str = "."):
    p = Path(directory)
    return [str(x) for x in p.iterdir()]
    
@app.get("/read_file")
def read_file(path: str):
    p = Path(path)
    if not p.exists():
        return {"error" : "파일 없음"}
    return {"content": p.read_text(encoding="utf-8")}


# 3. 텍스트 생성
class QueryRequest(BaseModel):
    question: str
    file_paths: List[str] # 참조할 파일 목록

class GenerateRequest(BaseModel):
    text: str

@app.post("/query_file")
def query_file(req: QueryRequest):
    collected_texts = []

    # 1. 로컬 파일 읽기
    for path in req.file_paths:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        with open(path, "r", encoding="latin1") as f:
            collected_texts.append(f.read())

    # 2. 모든 파일 내용을 하나의 문자열로 합치기
    context = "/n".join(collected_texts)

    # 3. OpenAI GPT에게 질문 + 문서 context 입력
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents."},
                {"role": "user", "content": f"Use the following documents to answer the question:\n{context}\n\nQuestion: {req.question}"}
            ],
            max_tokens=150,
        )

        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"answer": answer}


@app.post("/generate")
def generate_text(req: GenerateRequest):
    template = ChatPromptTemplate.from_template(
        "You are a financial phrase generator.\nUser input:\n{input}\nAnswer:"
    )
    chain = template | llm
    response = chain.invoke({"input": req.text})
    return {"generated": response.content}