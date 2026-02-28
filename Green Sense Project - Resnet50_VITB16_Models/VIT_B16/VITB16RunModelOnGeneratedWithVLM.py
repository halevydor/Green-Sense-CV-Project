# -*- coding: utf-8 -*-
"""
Evaluate trained ViT-B/16 model ONLY on generated images (Healthy/Dried/Contaminated)
AND run VLM-based semantic explanations for misclassified images.

Outputs:
- outputs_generated_eval_vit/metrics_generated_vit.json
- outputs_generated_eval_vit/predictions_generated_vit.csv
- outputs_generated_eval_vit/per_class_*.csv
- outputs_generated_eval_vit/vlm_semantic_explanations.csv
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

# --- for VLM semantic explanations ---
import pandas as pd
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# -----------------------------
# Config
# -----------------------------
class Config:
    # ✅ generated images root
    generated_root_dir = r"C:\afeca academy\year 2\CVDLP\GreenSpaceQualityProject\GreenSenseGeneratedPictures"

    # ✅ ViT trained weights (from your ViT training script)
    # example: outputs_gsq_vit/model_vit_b_16.pt
    trained_weights_path = r"outputs_gsq_vit\model_vit_b_16.pt"

    arch = "vit_b_16"
    batch = 32
    img_size = 224
    num_workers = 0
    out_dir = "outputs_generated_eval_vit"

    # =========================
    # VLM semantic analysis
    # =========================
    # קובץ התחזיות שהסקריפט הזה יוצר
    predictions_csv = r"outputs_generated_eval_vit\predictions_generated_vit.csv"

    # קובץ פלט עם ההסברים הסמנטיים
    vlm_out_csv = r"outputs_generated_eval_vit\vlm_semantic_explanations.csv"

    # מודל ה-VLM
    vlm_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_max_new_tokens = 150

    # אם תרצה אפשר לכבות את זה בקלות
    enable_vlm = True


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
    Returns (x, y, path) for saving per-image predictions.
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

        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # CHW

        x = torch.from_numpy(img)
        y = torch.tensor(y, dtype=torch.long)
        return x, y, path


def build_vit_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    ViT-B/16 from torchvision + replace head to NUM_CLASSES.
    """
    model = models.vit_b_16(
        weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    )
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_y_true, all_y_pred, all_probs = [], [], []
    all_paths = []

    for xb, yb, paths in tqdm(loader, desc="Eval Generated (ViT)", leave=False):
        xb = xb.to(device, non_blocking=True, dtype=torch.float32)
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

    header = (
        ["path", "true_idx", "pred_idx", "true_name", "pred_name", "correct"]
        + [f"prob_{c}" for c in class_names]
    )

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for p, yt, yp, pr in zip(paths, y_true, y_pred, y_proba):
            true_name = class_names[int(yt)]
            pred_name = class_names[int(yp)]
            correct = int(yt == yp)
            w.writerow(
                [p, int(yt), int(yp), true_name, pred_name, correct]
                + [float(x) for x in pr]
            )


def save_per_class_lists(out_dir, y_true, y_pred, paths, class_names):
    """
    Creates one CSV per TRUE class:
    - per_class_Healthy.csv
    - per_class_Dried.csv
    - per_class_Contaminated.csv
    Each contains: path, pred_name, correct(0/1)
    """
    import csv
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cls_idx, cls_name in enumerate(class_names):
        out_path = out_dir / f"per_class_{cls_name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "true_name", "pred_name", "correct"])

            for p, yt, yp in zip(paths, y_true, y_pred):
                if int(yt) != cls_idx:
                    continue
                pred_name = class_names[int(yp)]
                correct = int(yt == yp)
                w.writerow([p, cls_name, pred_name, correct])


# -----------------------------
# VLM: prompt + explanation
# -----------------------------
def build_vlm_prompt(true_label: str) -> str:
    """
    בונה prompt למודל ה-VLM:
    לא מבקש סיווג מחדש, אלא הערכה סמנטית של ההתאמה לקטגוריה האמיתית.
    """
    return f"""
The true class of this image is "{true_label}".
Explain whether the visual appearance of the image clearly matches this category.
If not, explain which visual features make it ambiguous or closer to another category
(Healthy, Dried, or Contaminated).
Be concise, objective, and focus only on visual characteristics.
""".strip()


@torch.no_grad()
def explain_image_with_vlm(image_path: str, true_label: str, model, processor) -> str:
    image = Image.open(image_path).convert("RGB")

    prompt_text = build_vlm_prompt(true_label)

    # ✅ חשוב: להכניס את התמונה לתוך messages כדי שייווצרו image tokens
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ],
        }
    ]

    # ✅ תבנית הצ'אט מוסיפה את טוקני התמונה נכון
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=args.vlm_max_new_tokens
    )

    return processor.decode(output[0], skip_special_tokens=True).strip()


def run_vlm_semantic_explanations(batch_size: int = 4):
    """
    גרסה batched:
    - קוראת את predictions_generated_vit.csv
    - מאתרת רק טעויות (correct == 0)
    - טוענת את Qwen2.5-VL על CUDA (float16) אם אפשר, אחרת CPU
    - מריצה VLM על טעויות ב-batches של גודל batch_size
    - שומרת vlm_semantic_explanations.csv
    """
    # אפשר לכבות VLM דרך config
    if not getattr(args, "enable_vlm", True):
        print("\n[VLM] Skipping semantic explanations (enable_vlm=False).")
        return

    if not os.path.exists(args.predictions_csv):
        print(f"\n[VLM] predictions_csv not found, skipping: {args.predictions_csv}")
        return

    df = pd.read_csv(args.predictions_csv)

    required_cols = {"path", "correct", "true_name", "pred_name"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"\n[VLM] predictions CSV missing required columns: {missing}. Skipping.")
        return

    df_err = df[df["correct"] == 0].copy()
    print(f"\n[VLM] Found {len(df_err)} misclassified images (correct == 0)")

    if df_err.empty:
        print("[VLM] No misclassifications to explain.")
        return

    # ------------------------------
    # טעינת מודל ה-VLM על GPU אם אפשר
    # ------------------------------
    print(f"[VLM] Loading VLM model: {args.vlm_model_id}")
    use_cuda = torch.cuda.is_available()
    print("[VLM] torch.cuda.is_available():", use_cuda)

    if use_cuda:
        try:
            print("[VLM] Trying to load model on CUDA (float16)...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.vlm_model_id,
                dtype=torch.float16,  # עדיף מה- torch_dtype המיושן
            ).to("cuda")
            device = torch.device("cuda")
            print("[VLM] Model loaded on CUDA (float16).")
            print("[VLM] CUDA device:", torch.cuda.get_device_name(0))
        except RuntimeError as e:
            print("[VLM] Failed to load on CUDA, falling back to CPU. Error:")
            print("      ", e)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.vlm_model_id,
                dtype=torch.float32,
            )
            device = torch.device("cpu")
            print("[VLM] Model loaded on CPU (float32).")
    else:
        print("[VLM] CUDA not available, loading model on CPU (float32).")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.vlm_model_id,
            dtype=torch.float32,
        )
        device = torch.device("cpu")
        print("[VLM] Model loaded on CPU (float32).")

    processor = AutoProcessor.from_pretrained(args.vlm_model_id)

    first_param = next(model.parameters())
    print("[VLM] first param device:", first_param.device, "dtype:", first_param.dtype)

    model.eval()
    torch.set_grad_enabled(False)

    # ------------------------------
    # Batch-loop על הטעויות
    # ------------------------------
    rows = []
    n = len(df_err)
    indices = list(df_err.index)

    for start in tqdm(range(0, n, batch_size), desc="VLM semantic analysis (batched)"):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        batch_df = df_err.loc[batch_idx]

        images = []
        prompts = []
        paths_list = []
        true_labels = []
        pred_labels = []

        # הכנת תמונות ופרומפטים לכל דוגמה בבאץ'
        for _, r in batch_df.iterrows():
            path = r["path"]
            true_label = r["true_name"]
            pred_label = r["pred_name"]

            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                # אם התמונה לא נקראת – נסמן כשגיאה ונמשיך
                rows.append({
                    "path": path,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "semantic_explanation": f"ERROR: cannot open image ({e})"
                })
                continue

            prompt_text = build_vlm_prompt(true_label)

            images.append(img)
            prompts.append(prompt_text)
            paths_list.append(path)
            true_labels.append(true_label)
            pred_labels.append(pred_label)

        if not images:
            continue  # כל התמונות בבאץ' נכשלו בפתיחה

        # בניית messages ו-chat template לכל דוגמה בבאץ'
        messages_list = []
        for p_text in prompts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": p_text}
                    ],
                }
            ]
            chat_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            messages_list.append(chat_text)

        try:
            # המרה לטנסורים בבת אחת
            inputs = processor(
                text=messages_list,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=args.vlm_max_new_tokens
            )

            # outputs: [batch, seq_len] – מפענחים כל שורה בנפרד
            for i, out in enumerate(outputs):
                explanation = processor.decode(out, skip_special_tokens=True).strip()

                rows.append({
                    "path": paths_list[i],
                    "true_label": true_labels[i],
                    "pred_label": pred_labels[i],
                    "semantic_explanation": explanation
                })

        except Exception as e:
            # במקרה של שגיאה בבאץ' – נסמן לכל הדוגמאות באותו באץ'
            err_msg = f"ERROR during batched explanation: {e}"
            print("[VLM] Batch error:", err_msg)
            for i in range(len(paths_list)):
                rows.append({
                    "path": paths_list[i],
                    "true_label": true_labels[i],
                    "pred_label": pred_labels[i],
                    "semantic_explanation": err_msg
                })

    # ------------------------------
    # שמירת הפלט ל-CSV
    # ------------------------------
    out_path = Path(args.vlm_out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n[VLM] Saved batched semantic explanations to:")
    print(str(out_path))




# -----------------------------
# main
# -----------------------------
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
    model = build_vit_model(NUM_CLASSES, pretrained=False)
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).float()

    # 4) eval
    y_true, y_pred, y_proba, paths = eval_model(model, loader, device)

    # 5) metrics
    metrics = compute_metrics(y_true, y_pred, y_proba, CLASS_NAMES)

    print("\n=== Generated Images Evaluation (ViT) ===")
    for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "kappa", "roc_auc_macro_ovr"]:
        print(f"{k}: {metrics[k]}")
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification Report:\n")
    print(metrics["classification_report"])

    # 6) save outputs
    metrics_path = Path(args.out_dir) / "metrics_generated_vit.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    preds_csv = Path(args.predictions_csv)
    preds_csv.parent.mkdir(parents=True, exist_ok=True)
    save_predictions_csv(str(preds_csv), y_true, y_pred, y_proba, paths, CLASS_NAMES)
    print(f"Saved predictions: {preds_csv}")

    save_per_class_lists(args.out_dir, y_true, y_pred, paths, CLASS_NAMES)
    print("Saved per-class lists (path + correct) into:", args.out_dir)

    # 7) VLM semantic explanations for misclassified samples
    run_vlm_semantic_explanations()

    print("\nAll done.")


if __name__ == "__main__":
    main()
