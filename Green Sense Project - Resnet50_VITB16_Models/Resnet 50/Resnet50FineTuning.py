# -*- coding: utf-8 -*-
"""
Fine-tune ResNet50 על תמונות Generated – 3 מחלקות:
Healthy, Dried, Contaminated

- משתמש באותו root כמו ה-ViT fine-tuning:
  C:\afeca academy\year 2\CVDLP\GreenSpaceQualityProject\GreenSenseGeneratedPicturesForFineTunning

- טוען את המודל הבסיסי:
  outputs_gsq_torch\model_resnet50.pt

- מקפיא את כל הרשת חוץ מה-fully connected (head)
- מבצע Fine-Tuning על התמונות המג'ונרטות (Train/Validate/Test)
- שומר מודל ומשתני מדידה בתיקייה:
  outputs_resnet_finetune_generated

שמור כ: ResNet50FineTuneGenerated3Classes.py
הרצה (בסביבת greenspace):
    python ResNet50FineTuneGenerated3Classes.py
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import cv2
import albumentations as A

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm


# =============================
# Config
# =============================
class Config:
    # root ל-generated (כמו בקובץ של ה-ViT)
    finetune_root_dir = (
        r"C:\afeca academy\year 2\CVDLP\GreenSpaceQualityProject\GreenSenseGeneratedPicturesForFineTunning"
    )

    # מודל ResNet הבסיסי (מאומן על OriginalPictures)
    base_model_path = r"outputs_gsq_torch\model_resnet50.pt"

    # היפר-פרמטרים ל-Fine-Tuning
    batch = 8
    img_size = 224
    num_workers = 0
    epochs = 5
    lr = 1e-4
    seed = 42

    # תיקיית פלט ל-Fine-Tuning
    out_dir = "outputs_resnet_finetune_generated"


args = Config()

# שמות המחלקות – כמו במודל המקורי
CLASS_NAMES = ["Healthy", "Dried", "Contaminated"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================
# Utilities
# =============================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(img_size: int = 224) -> A.Compose:
    """
    אוגמנטציה עדינה ל-Train (כמו ב-ViT fine-tuning).
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(img_size, img_size),
        ]
    )


def build_eval_transform(img_size: int = 224) -> A.Compose:
    """
    ל-Validate/Test – רק resize.
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
        ]
    )


def collect_generated_finetune_files(
    root_dir: Path, class_names: List[str]
) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
    """
    root_dir = GreenSenseGeneratedPicturesForFineTunning

    מצפה למבנה:
      root_dir / <ClassName> / Train
      root_dir / <ClassName> / Validate
      root_dir / <ClassName> / Test
    לדוגמה:
      root_dir/Healthy/Train
      root_dir/Dried/Validate
      root_dir/Contaminated/Test
    """
    splits = ["Train", "Validate", "Test"]
    files = {s: [] for s in splits}
    labels = {s: [] for s in splits}

    if not root_dir.exists():
        raise FileNotFoundError(f"Fine-tune root dir not found: {root_dir}")

    # נוודא שכל 3 המחלקות קיימות (כמו ב-ViT)
    for cls_idx, cls_name in enumerate(class_names):
        cls_root = root_dir / cls_name
        if not cls_root.exists():
            raise FileNotFoundError(
                f"Class folder not found under generated root: {cls_root}"
            )

        for split in splits:
            split_dir = cls_root / split
            if not split_dir.exists():
                raise FileNotFoundError(
                    f"Split folder not found: {split_dir} (expected Train/Validate/Test)"
                )

            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
                for p in split_dir.glob(ext):
                    files[split].append(str(p))
                    labels[split].append(cls_idx)

    return files, labels


# =============================
# Dataset
# =============================
class GeneratedFineTuneDataset(Dataset):
    """
    Dataset ל-Fine-Tuning על Healthy/Dried/Contaminated Generated.
    """

    def __init__(self, files: List[str], labels: List[int], transform: A.Compose):
        self.files = list(files)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        y = self.labels[idx]

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        x = torch.from_numpy(img)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# =============================
# Model
# =============================
def build_resnet_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    ResNet50 כמו במודל הבסיסי, עם head מותאם ל-3 מחלקות.
    """
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# =============================
# Train / Eval helpers
# =============================
def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Train (fine-tune)", leave=False)

    for xb, yb in pbar:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(yb.detach().cpu().numpy())

        cur_loss = running_loss / max(1, len(all_labels) * loader.batch_size)
        pbar.set_postfix(loss=f"{cur_loss:.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, acc


def eval_one_epoch(model, loader, device, criterion, desc: str = "Eval"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            running_loss += loss.item() * xb.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, acc, all_labels, all_preds, all_probs


def compute_metrics(y_true_idx, y_pred_idx, y_proba, class_names):
    acc = accuracy_score(y_true_idx, y_pred_idx)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_idx,
        y_pred_idx,
        average="macro",
        zero_division=0,
    )
    kappa = cohen_kappa_score(y_true_idx, y_pred_idx)

    num_classes = len(class_names)
    y_true_oh = np.eye(num_classes)[y_true_idx]

    try:
        rocauc = roc_auc_score(
            y_true_oh,
            y_proba,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        rocauc = float("nan")

    report = classification_report(
        y_true_idx,
        y_pred_idx,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_idx, y_pred_idx)

    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "kappa": kappa,
        "roc_auc_macro_ovr": rocauc,
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


# =============================
# main
# =============================
def main():
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) טעינת קבצים ל-Fine-Tune (Generated – 3 מחלקות)
    root = Path(args.finetune_root_dir)
    files, labels = collect_generated_finetune_files(root, CLASS_NAMES)

    print("Samples per split:")
    for split in ["Train", "Validate", "Test"]:
        print(f"  {split}: {len(files[split])} samples")

    # 2) בניית Datasets + DataLoaders
    train_tf = build_train_transform(args.img_size)
    eval_tf = build_eval_transform(args.img_size)

    train_ds = GeneratedFineTuneDataset(files["Train"], labels["Train"], transform=train_tf)
    val_ds = GeneratedFineTuneDataset(files["Validate"], labels["Validate"], transform=eval_tf)
    test_ds = GeneratedFineTuneDataset(files["Test"], labels["Test"], transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 3) בניית מודל + טעינת משקלים קיימים
    model = build_resnet_model(NUM_CLASSES, pretrained=False)

    assert os.path.exists(
        args.base_model_path
    ), f"Base model not found: {args.base_model_path}"

    state = torch.load(args.base_model_path, map_location=device)
    # strict=True כי ה-head גם כאן 3 מחלקות כמו במודל הבסיס
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    # 4) הקפאת כל הרשת חוץ מה-FC
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    print("\nTrainable parameters (after freezing backbone):")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print("  ", n)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # 5) לולאת Fine-Tuning
    best_val_acc = 0.0
    best_state = None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, criterion
        )
        val_loss, val_acc, _, _, _ = eval_one_epoch(
            model, val_loader, device, criterion, desc="Val (fine-tune generated)"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train - loss: {train_loss:.4f}, acc: {train_acc:.4f}\n"
            f"Val   - loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            print("  >>> Improvement on Val, saving best model in memory.")
        else:
            print("  No improvement on Val.")

    if best_state is not None:
        model.load_state_dict(best_state)

    # 6) הערכה סופית על Test (Generated – 3 מחלקות)
    test_loss, test_acc, y_true, y_pred, y_proba = eval_one_epoch(
        model, test_loader, device, criterion, desc="Test (Generated 3-classes)"
    )

    print("\n=== Fine-Tuned ResNet – Evaluation on Generated Test (3 classes) ===")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc:.4f}")

    metrics = compute_metrics(y_true, y_pred, y_proba, CLASS_NAMES)
    print("\nMetrics:")
    for k, v in metrics.items():
        if k in ["report", "confusion_matrix"]:
            continue
        print(f"{k}: {v}")

    print("\nClassification Report:\n", metrics["report"])
    print("\nConfusion Matrix:\n", np.array(metrics["confusion_matrix"]))

    # 7) שמירה
    model_path = Path(args.out_dir) / "model_resnet50_finetuned_generated_3classes.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved fine-tuned ResNet to: {model_path}")

    metrics_path = Path(args.out_dir) / "metrics_resnet_finetune_generated_3classes.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    hist_path = Path(args.out_dir) / "history_resnet_finetune_generated_3classes.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Saved training history to: {hist_path}")


if __name__ == "__main__":
    main()