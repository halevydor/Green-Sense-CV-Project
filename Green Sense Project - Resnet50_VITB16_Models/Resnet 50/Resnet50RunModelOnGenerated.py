# -*- coding: utf-8 -*-
"""
Evaluate trained ResNet50 model ONLY on generated images
Folder structure (same as original):
  root/
    Healthy/
    Dried/
    Contaminated/

Metrics:
- Accuracy
- Precision/Recall/F1 Macro
- Cohen's Kappa
- ROC-AUC Macro OVR
- Confusion Matrix
- Classification Report

Outputs:
- outputs_generated_eval/metrics_generated.json
- outputs_generated_eval/predictions_generated.csv
"""

import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import albumentations as A

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
class Config:
    # ✅ generated images root
    generated_root_dir = r"C:\afeca academy\year 2\CVDLP\GreenSpaceQualityProject\GreenSenseHandedPictures"

    # ✅ where your trained weights were saved from training script
    # example: outputs_gsq_torch/model_resnet50.pt
    #trained_weights_path = r"outputs_gsq_torch\model_resnet50.pt"
    trained_weights_path = r"outputs_resnet_finetune_generated\model_resnet50_finetuned_generated_3classes.pt"

    arch = "resnet50"
    batch = 32
    img_size = 224
    num_workers = 0
    out_dir = "outputs_generated_eval"


args = Config()

CLASS_NAMES = ["Healthy", "Dried", "Contaminated"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -----------------------------
# Utils
# -----------------------------
def build_eval_transform(img_size=224):
    return A.Compose([A.Resize(img_size, img_size)])


def make_filelist(root_dir: Path, class_names: List[str]) -> Tuple[List[str], List[int]]:
    files, labels = [], []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")

    for idx, cname in enumerate(class_names):
        cdir = root_dir / cname
        if not cdir.exists():
            raise FileNotFoundError(f"Missing class folder: {cdir}")

        for e in exts:
            for p in cdir.glob(e):
                files.append(str(p))
                labels.append(idx)

    return files, labels


class GreenSpaceEvalDataset(Dataset):
    """
    Eval-only dataset: deterministic resize + ImageNet normalization.
    """
    def __init__(self, files: List[str], labels: List[int], transform=None):
        self.files = list(files)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        y = self.labels[idx]

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        # uint8 HWC -> float32 HWC in [0,1]
        img = img.astype(np.float32) / 255.0
        # normalize ImageNet
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        x = torch.from_numpy(img)             # (C,H,W)
        y = torch.tensor(y, dtype=torch.long)
        return x, y, path


def build_model_resnet50(num_classes: int, pretrained: bool = False) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_y_true = []
    all_y_pred = []
    all_probs = []
    all_paths = []

    for xb, yb, paths in tqdm(loader, desc="Eval Generated", leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_y_true.append(yb.cpu().numpy())
        all_y_pred.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_paths.extend(list(paths))

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_proba = np.concatenate(all_probs)
    return y_true, y_pred, y_proba, all_paths


def compute_metrics(y_true_idx, y_pred_idx, y_proba, class_names):
    num_classes = len(class_names)
    y_true_oh = np.eye(num_classes)[y_true_idx]

    acc = accuracy_score(y_true_idx, y_pred_idx)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, average="macro", zero_division=0
    )
    kappa = cohen_kappa_score(y_true_idx, y_pred_idx)

    try:
        rocauc = roc_auc_score(y_true_oh, y_proba, multi_class="ovr", average="macro")
    except Exception:
        rocauc = float("nan")

    report = classification_report(
        y_true_idx, y_pred_idx, target_names=class_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true_idx, y_pred_idx)

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "kappa": float(kappa),
        "roc_auc_macro_ovr": float(rocauc) if np.isfinite(rocauc) else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }


def save_predictions_csv(out_csv_path, y_true, y_pred, y_proba, paths, class_names):
    import csv
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    header = ["path", "y_true", "y_pred"] + [f"prob_{c}" for c in class_names]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for p, yt, yp, pr in zip(paths, y_true, y_pred, y_proba):
            w.writerow([p, int(yt), int(yp)] + [float(x) for x in pr])


def main():
    generated_root = Path(args.generated_root_dir)
    assert generated_root.exists(), f"Generated root dir not found: {generated_root}"

    weights_path = Path(args.trained_weights_path)
    assert weights_path.exists(), f"Trained weights not found: {weights_path}"

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) collect generated files
    files, labels = make_filelist(generated_root, CLASS_NAMES)
    print(f"Generated set: {len(files)} images")
    uniq, counts = np.unique(labels, return_counts=True)
    for i, c in zip(uniq, counts):
        print(f"  {CLASS_NAMES[i]}: {c}")

    # 2) loader
    eval_tf = build_eval_transform(args.img_size)
    ds = GreenSpaceEvalDataset(files, labels, transform=eval_tf)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 3) load model
    model = build_model_resnet50(NUM_CLASSES, pretrained=False)
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).float()

    # 4) eval
    y_true, y_pred, y_proba, paths = eval_model(model, loader, device)

    # 5) metrics
    metrics = compute_metrics(y_true, y_pred, y_proba, CLASS_NAMES)

    print("\n=== Generated Images Evaluation ===")
    for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "kappa", "roc_auc_macro_ovr"]:
        print(f"{k}: {metrics[k]}")
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification Report:\n")
    print(metrics["classification_report"])

    # 6) save outputs
    metrics_path = Path(args.out_dir) / "metrics_generated.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    preds_csv = Path(args.out_dir) / "predictions_generated.csv"
    save_predictions_csv(str(preds_csv), y_true, y_pred, y_proba, paths, CLASS_NAMES)
    print(f"Saved predictions: {preds_csv}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
