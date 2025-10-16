import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 이미지 파일명 수정 (cat.16.jpg -> cat.17.jpg)
image = Image.open("cat.17.jpg").convert("RGB")

# LLaVA 모델용 프롬프트 형식
prompt = "USER: <image>\n이 이미지에 고양이가 있나요? 있다면 무슨 색인가요?\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=100)
answer = processor.decode(output[0], skip_special_tokens=True)
print(answer)
