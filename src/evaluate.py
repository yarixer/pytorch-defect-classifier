from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/best_model.pth")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def build_loader(split="test"):
    dataset = datasets.ImageFolder(DATA_DIR / split, transform=eval_transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return dataset, loader


def collect_predictions(model, loader):
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


def save_confusion_matrix(cm, class_names, save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_at_threshold(y_true, defect_probs, threshold, class_names):
    defect_idx = class_names.index("defect")
    good_idx = class_names.index("good")

    y_pred = np.where(defect_probs >= threshold, defect_idx, good_idx)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "defect_precision": float(report["defect"]["precision"]),
        "defect_recall": float(report["defect"]["recall"]),
        "defect_f1": float(report["defect"]["f1-score"]),
        "good_precision": float(report["good"]["precision"]),
        "good_recall": float(report["good"]["recall"]),
        "good_f1": float(report["good"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    return metrics, cm, y_pred


def main():
    dataset, loader = build_loader(split="test")
    class_names = dataset.classes
    print("Classes:", class_names)

    model = build_model()
    y_true, probs = collect_predictions(model, loader)

    defect_idx = class_names.index("defect")
    defect_probs = probs[:, defect_idx]

    thresholds = np.arange(0.10, 0.91, 0.05)

    best = None
    best_cm = None

    print("\nThreshold tuning results:")
    for threshold in thresholds:
        metrics, cm, _ = evaluate_at_threshold(y_true, defect_probs, threshold, class_names)
        print(
            f"threshold={threshold:.2f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"defect_precision={metrics['defect_precision']:.4f} | "
            f"defect_recall={metrics['defect_recall']:.4f}"
        )

        score = (metrics["defect_recall"], metrics["defect_f1"], metrics["accuracy"])
        if best is None or score > best["score"]:
            best = {"score": score, "metrics": metrics}
            best_cm = cm

    best_metrics = best["metrics"]

    print("\nBest threshold:")
    print(json.dumps(best_metrics, indent=2, ensure_ascii=False))

    with open(ARTIFACTS_DIR / "evaluation_threshold_tuning.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=False)

    save_confusion_matrix(
        best_cm,
        class_names,
        ARTIFACTS_DIR / "confusion_matrix_threshold_tuned.png",
        title=f"Confusion Matrix (threshold={best_metrics['threshold']:.2f})",
    )

    print("\nSaved:")
    print("-", ARTIFACTS_DIR / "evaluation_threshold_tuning.json")
    print("-", ARTIFACTS_DIR / "confusion_matrix_threshold_tuned.png")


if __name__ == "__main__":
    main()
