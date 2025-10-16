import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

# 모델 ID
model_id = "llava-hf/llava-1.5-7b-hf"

# Processor & model load
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 이미지 여러 장 불러오기
image_files = ["cat.26.jpg","dog.996.jpg"]
images = []

for f in image_files:
    img = Image.open(f).convert("RGB")
    img = img.resize((224, 224))  # 가로 512, 세로 512로 축소
    images.append(img)

# LLaVA 모델용 프롬프트
prompt = """USER: <image><image>
이 이미지들에 어떤 동물들이 있나요? 동물의 종류를 알려주세요
ASSISTANT:"""

# Processor에 텍스트 + 이미지 입력
inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)

# 답변 생성
output = model.generate(**inputs, max_new_tokens=200)
answer = processor.decode(output[0], skip_special_tokens=True)

print(answer)