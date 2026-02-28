# Green-Sense-CV-Project
Green Sense is a computer vision pipeline for automated urban vegetation quality monitoring from street-level municipal camera imagery. The project classifies green spaces into Healthy, Dried, and Contaminated conditions using synthetic data generation (FIBO), Vision Transformers (ViT-B16) and a hybrid CLIP–DINOv2 architecture.

# 🌿 Green Sense 🌿  
### Urban Green Space Quality Monitoring from Street-Level Cameras

## Overview

Green Sense is an end-to-end computer vision pipeline designed to monitor the quality of urban green spaces using street-level municipal camera imagery.

The system classifies vegetation into three operational categories:
- 🟢 **Healthy** – Well-maintained, green vegetation with no visible litter  
- 🟡 **Dried** – Vegetation showing signs of water stress or decay  
- 🔴 **Contaminated** – Vegetation affected by visible litter or pollution  

The project addresses a key municipal challenge: scalable monitoring of green space health without costly manual inspections.

To overcome the lack of labeled street-camera data, Green Sense constructs a **synthetic dataset** using the FIBO generative AI model and evaluates multiple model families under domain shift conditions.
---

## 🚀 Key Contributions

- ✅ Creation of a **synthetic street-camera dataset** using controlled generative AI  
- ✅ Validation using **human screening + NIQE filtering**
- ✅ Domain shift evaluation (close-up → street camera viewpoint)
- ✅ Comparison of three model families under identical protocols
- ✅ Hybrid architecture combining:
  - CLIP (vision-language)
  - DINOv2 (self-supervised vision)
  - Random Forest classifier
- ✅ Semantic auditing using Qwen2.5-VL for explainability
---

## 🧠 Models Evaluated

### 1️⃣ ResNet50 Baseline
- ImageNet-pretrained
- Fine-tuned end-to-end
- Strong on close-up images
- Significant performance drop under street-camera domain shift

### 2️⃣ Vision Transformer (ViT-B16)
- Global self-attention over image patches
- Improved robustness to scale and viewpoint changes
- Significantly outperforms ResNet50 in synthetic street-view setting

### 3️⃣ CLIP–DINOv2 Hybrid (Final Model)
- CLIP semantic embeddings
- DINOv2 visual embeddings
- Vegetation color & texture statistics
- Random Forest classifier
- Ensemble of semantic scores + classifier probabilities

This model achieves the strongest results.

---

## 📊 Results

### Synthetic Street Camera Test Set

| Model                  | Accuracy | F1   | Cohen Kappa | ROC AUC |
|------------------------|----------|------|-------------|---------|
| ResNet50               | 0.581    | 0.557| 0.375       | 0.796   |
| ViT-B16                | 0.905    | 0.902| 0.856       | 0.972   |
| **CLIP–DINOv2 Hybrid** | **0.972**|**0.972**|**0.958** | **0.998** |

### Real-World Image Evaluation (33 images)

CLIP–DINOv2 Hybrid:
- Accuracy: 0.849  
- ROC-AUC: 0.899  
- Demonstrates meaningful transfer without retraining

---

## 🏗 Project Architecture
