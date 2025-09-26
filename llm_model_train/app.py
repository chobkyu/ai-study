from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("./my_finetuned_model")
model = AutoModelForCausalLM.from_pretrained("./my_finetuned_model").to(device)
model.eval()

class Prompt(BaseModel):
    text: str

@app.post("/generate")
async def generate(prompt: Prompt):
    inputs = tokenizer(prompt.text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=150,         # 생성 길이
            temperature=1.0,        # 낮추면 더 안정적
            top_p=0.95,              # 상위 확률 토큰만 샘플링
            do_sample=False,         # 확률 샘플링 활성화
            repetition_penalty=1.1  # 같은 단어 반복 억제
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated": text}

@app.get("/")
def root(): 
    return {"message": "POST /generate with {'text': '...'}"}