from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import os
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="marvelmmy/sortify-resnet-model", filename="sortify_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 4)
)
model.load_state_dict(torch.load("sortify_model.pth", map_location=device))
model.to(device)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['hazardous', 'organic', 'others', 'recyclable']
tips = {
    "hazardous": "Dispose in a hazardous bin. Do not touch. Contact authorities if needed.",
    "organic": "Throw it into the organic bin or compost it if possible.",
    "others": "General waste. Throw it in the regular trash bin.",
    "recyclable": "Throw it into the recyclable bin or make it into something valuable."
}

color_map = {
    "organic": "success",       # Green
    "recyclable": "primary",    # Blue
    "hazardous": "danger",      # Red
    "others": "secondary"       # Gray
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename
        save_dir = os.path.join("static", "uploads")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        # Save image for preview
        with open(save_path, "wb") as f:
            f.write(contents)

        # Read & preprocess for model
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence_score, predicted = torch.max(probs, 1)

            predicted_label = class_names[predicted.item()]
            confidence = round(confidence_score.item() * 100, 2)
            tip = tips.get(predicted_label, "No tip available")
            color_class = color_map.get(predicted_label, "secondary")

    except Exception as e:
        print(f"Prediction error: {e}")
        predicted_label = "prediction failed"
        tip = "no tip available"
        confidence = 0.0
        color_class = "secondary"
        save_path = None

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": predicted_label,
        "confidence": f'{confidence:.2f}%',
        "tip": tip,
        "color_class": color_class,
        "filename": file.filename,
        "image_path": f"/static/uploads/{filename}" if save_path else None
    })
