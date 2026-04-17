import io
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms, models

MODEL_PATH = Path("models/best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["defect", "good"]
DEFECT_THRESHOLD = 0.40
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

app = FastAPI(title="Pharma Defect Classifier")


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    defect_prob = float(probs[CLASS_NAMES.index("defect")].item())
    good_prob = float(probs[CLASS_NAMES.index("good")].item())

    predicted_class = "defect" if defect_prob >= DEFECT_THRESHOLD else "good"
    confidence = defect_prob if predicted_class == "defect" else good_prob

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "defect_probability": round(defect_prob, 4),
        "good_probability": round(good_prob, 4),
        "threshold": DEFECT_THRESHOLD,
    }
