from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# 1. 모델과 토크나이저 로드
model_dir = "./gptneo-finetuned/checkpoint-1019"  # 학습 완료된 모델 경로
base_model_name = "EleutherAI/gpt-neo-125M"  # 원본 모델명 (토크나이저용)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_dir)

# 디바이스 설정 (MPS/CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# 2. 요청 스키마 정의
class GenerateRequest(BaseModel):
    text: str
    max_length: int = 64
    temperature: float = 0.7

# 3. 텍스트 생성 엔드포인트
@app.post("/generate")
def generate(req: GenerateRequest):
    # 토크나이징
    inputs = tokenizer(req.text, return_tensors="pt").to(device)

    # 텍스트 생성
    outputs = model.generate(
        **inputs,
        max_length=req.max_length,
        temperature=req.temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # 결과 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated": generated_text}
