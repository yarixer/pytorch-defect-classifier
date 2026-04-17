from pathlib import Path
import copy
import json
import random
from collections import Counter

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
MODEL_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")

MODEL_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
IMAGE_SIZE = 224
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders():
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transforms)
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def get_class_weights(train_dataset):
    class_counts = Counter(train_dataset.targets)
    total = sum(class_counts.values())
    num_classes = len(class_counts)

    weights = []
    for class_idx in range(num_classes):
        weight = total / (num_classes * class_counts[class_idx])
        weights.append(weight)

    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    return class_weights, class_counts


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc, all_labels, all_preds, all_probs


def save_confusion_matrix(cm, class_names, save_path: Path):
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
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def to_float_dict(metrics_dict):
    result = {}
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            result[key] = to_float_dict(value)
        elif isinstance(value, (np.floating, np.integer)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def train():
    set_seed(SEED)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_loaders()

    print("Classes:", train_dataset.classes)
    print("Class to idx:", train_dataset.class_to_idx)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    class_weights, class_counts = get_class_weights(train_dataset)
    print("Train class counts:", dict(class_counts))
    print("Class weights:", class_weights.detach().cpu().tolist())

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc, y_val_true, y_val_pred, _ = evaluate(model, val_loader, criterion)
        val_macro_f1 = f1_score(y_val_true, y_val_pred, average="macro")

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    best_model_path = MODEL_DIR / "best_model.pth"
    torch.save(best_model_state, best_model_path)
    print(f"\nBest model saved to {best_model_path} with val_loss={best_val_loss:.4f}")

    model.load_state_dict(best_model_state)

    test_loss, test_acc, y_true, y_pred, _ = evaluate(model, test_loader, criterion)

    report = classification_report(
        y_true,
        y_pred,
        target_names=train_dataset.classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "seed": SEED,
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history["train_loss"]),
        "learning_rate": LEARNING_RATE,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "test_macro_precision": macro_precision,
        "test_macro_recall": macro_recall,
        "class_names": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "train_class_counts": dict(class_counts),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    metrics = to_float_dict(metrics)

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    history_path = ARTIFACTS_DIR / "history.json"
    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    save_confusion_matrix(cm, train_dataset.classes, cm_path)

    print(f"\nTEST LOSS: {test_loss:.4f}")
    print(f"TEST ACC:  {test_acc:.4f}")
    print(f"TEST MACRO F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=train_dataset.classes, zero_division=0))
    print("\nConfusion matrix:")
    print(cm)

    print("\nSaved artifacts:")
    print(f"- {best_model_path}")
    print(f"- {metrics_path}")
    print(f"- {history_path}")
    print(f"- {cm_path}")


if __name__ == "__main__":
    train()