from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# 모델과 프로세서 불러오기
model_name = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForQuestionAnswering.from_pretrained(model_name)

# 테스트 용 이미지와 질문
image_path = "cat.26.jpg"
question = "고양이가 존재하나요?"

# 이미지 로드
image = Image.open(image_path).convert("RGB")

# 모델 입력 준비
inputs = processor(image, question, return_tensors="pt")

# 모델 추론 (답변 생성)
out = model.generate(**inputs, max_new_tokens=20)
answer = processor.decode(out[0], skip_special_tokens=True)

print(f"❓ 질문: {question}")
print(f"💬 답변: {answer}")