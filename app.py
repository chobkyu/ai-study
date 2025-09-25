from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import models, transforms
from torch import nn

app = FastAPI()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model= models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("cats_dogs.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        pred = torch.argmax(outputs, dim=1).item()
    label = "cat" if pred == 0 else "dog"
    print("hihi")
    return {"prediction": label}


@app.get("/")
def root():
    return {"message": "POST /predict with an image"}