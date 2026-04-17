from pathlib import Path
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

IMAGE_PATH = "data/processed/test/defect/016.png"  # r
MODEL_PATH = "models/best_model.pth"
OUTPUT_PATH = "artifacts/gradcam_result.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["defect", "good"]
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def main():
    model = load_model()

    image = Image.open(IMAGE_PATH).convert("RGB")
    rgb = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    os.makedirs("artifacts", exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
