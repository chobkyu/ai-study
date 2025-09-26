from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import models, transforms
from PIL import Image
import torch

app = FastAPI(title="Image Classification API")

# 1. 사전 학습된 모델 로드 (resnet18)
model = models.resnet18(pretrained=True)
model.eval() # 추론 모드

# 2. 입력 이미지 전처리 파이프라인
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# 3. ImageNet 클래스 이름 가져오기
import requests
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = requests.get(labels_url).text.splitlines()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. 파일 -> PIL 이미지
        image = Image.open(file.file).convert("RGB")

        # 2. 전처리
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # 배치 차원 추가

        # 3. 추론
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # 4. 상위 5개 클래스 추출
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        results = []

        for i in range(top5_prob.size(0)):
            results.append({
                "class_name": imagenet_classes[top5_catid[i]],
                "probability" : float(top5_prob[i])
            })

        return JSONResponse(content={"predictions": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {"message": "Send POST /predict with an image file"}

        
