# 🌳 VLM Sense — Greenspace Quality Classifier

A Computer Vision pipeline that classifies park and greenspace images into **Healthy**, **Dried**, or **Contaminated** categories using multi-model feature fusion (CLIP + DINOv2) and an ensemble prediction strategy.

---

## 📁 Project Structure

```
vlm_sense/
├── app.py                   # Streamlit web interface (single image + batch evaluation)
├── config.py                # Central configuration (prompts, model names, paths)
├── train.py                 # Training pipeline (feature extraction + Random Forest)
├── dataset.py               # Dataset loading and CLIP preprocessing
├── scene_features.py        # CLIP + DINOv2 scene-level feature extraction
├── dino_features.py         # DINOv2 model wrapper and embedding extraction
├── vegetation_detector.py   # HSV color-based vegetation detection + road sign masking
├── vegetation_features.py   # Vegetation crop embeddings + color/texture analysis
├── models/                  # Trained model artifacts
│   ├── best_classifier_dino.pkl   # Random Forest (CLIP+DINO features, 1795-dim)
│   ├── scaler_dino.pkl            # StandardScaler for feature normalization
│   ├── best_classifier.pkl        # Fallback RF (CLIP-only, 1027-dim)
│   ├── scaler.pkl                 # Fallback scaler
│   ├── confusion_matrix.png       # Evaluation confusion matrix
│   ├── feature_importance.png     # Top feature importances
│   └── metrics.json               # Accuracy, F1, etc.
├── requirements.txt         # Python dependencies
├── packages.txt             # System dependencies (Streamlit Cloud)
├── Data/                    # Training/validation/test images (not included in deployment)
└── System_Architecture_Documentation.html
```

---

## 🧠 Architecture

### Feature Extraction (1,795-dimensional vector)

| Component | Dimensions | Description |
|---|---|---|
| CLIP Scene Embedding | 512 | Global scene understanding via ViT-B/32 |
| CLIP Vegetation Embedding | 512 | Mean-pooled crops from detected vegetation regions |
| DINOv2 Scene Embedding | 384 | Fine-grained visual texture via ViT-S/14 |
| DINOv2 Vegetation Embedding | 384 | Crop-level texture features |
| Color/Texture Features | 3 | Green ratio, edge density, vegetation coverage |

### Prediction Pipeline (Ensemble Strategy)

1. **Road Sign Masking** — HSV-based detection of red/blue/white rectangular objects, inpainted before scene analysis so signs aren't misread as contamination
2. **Random Forest** — Trained on 1,795-dim feature vectors
3. **CLIP Prompt Voting** — 30 text prompts (10 per class) scored against the image
4. **Ensemble** — 60% RF + 40% CLIP, with boosts:
   - Top-1 prompt class: +5%
   - 2/3 top-3 majority: +7%
   - 3/3 top-3 consensus: full override (98% confidence)

### Decision Hierarchy

| Priority | Condition | Action |
|---|---|---|
| 1 | 3/3 top prompts agree | Override → 98% confidence |
| 2 | 2/3 top prompts agree | +7% boost to majority class |
| 3 | RF confidence < 65% | Use ensemble (RF+CLIP) |
| 4 | RF confident ≥ 65% | Trust RF alone |

---

## 🚀 How to Run Locally

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501` with two tabs:

- **Single Image Analysis** — Upload one image, see detailed results with confidence breakdown
- **Batch Evaluation** — Upload labeled images per class, get accuracy, F1, confusion matrix, ROC curves

---

## ☁️ Deploy to Streamlit Cloud

1. Push this folder to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository, select `app.py`
4. Deploy — the app auto-downloads CLIP and DINOv2 on first run

---

## 🔧 Key Design Decisions

- **Multi-model fusion** — CLIP captures semantic meaning ("healthy park"), DINOv2 captures visual texture (leaf patterns)
- **HSV vegetation detection** — Lightweight fallback instead of GroundingDINO for cloud deployment (< 1GB RAM)
- **Road sign masking** — Prevents misclassification of urban signage as contamination
- **Ensemble voting** — Combines learned features (RF) with zero-shot reasoning (CLIP prompts) for robustness

---

## ⚙️ System Requirements

| Environment | RAM | Notes |
|---|---|---|
| Cloud (Streamlit) | 1 GB | Auto-resize, CPU-only, single-threaded |
| Local | 4 GB+ | Faster processing, optional GPU support |
| Python | 3.8+ | Tested on 3.10–3.13 |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| App won't start | Check Python ≥ 3.8, reinstall: `pip install -r requirements.txt --force-reinstall` |
| Out of memory (cloud) | Images are auto-resized to 1024px; reduce `max_size` in `app.py` if needed |
| "Models not found" | Ensure `models/` folder contains `.pkl` files |
| Wrong predictions | Verify the trained model matches the feature dimension (1795 for DINO-enhanced) |

---

**Version:** February 2026  
**Status:** Production-Ready ✅
