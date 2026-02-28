# NIQE Image Quality Analyzer

A Computer Vision project for **no-reference image quality assessment** using two complementary metrics:

1. **NIQE (Naturalness Image Quality Evaluator)** — measures perceptual quality without needing a reference image.
2. **Depth Consistency** — evaluates structural coherence using monocular depth estimation (MiDaS).

## Project Structure

```
niqe code/
├── niqe.py                  # Core NIQE algorithm implementation
├── niqe2.py                 # Batch CLI script for scoring folders of images
├── niqe_app.py              # Streamlit web dashboard for NIQE analysis
├── depth_consistency.py     # Depth consistency evaluator (MiDaS-based)
├── depth_app.py             # Streamlit web dashboard for depth analysis
├── run_desktop.py           # Desktop launcher (PyInstaller entry point)
├── NIQE_Analyzer.spec       # PyInstaller build configuration
├── data/
│   └── niqe_image_params.mat  # Pretrained pristine image statistics
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
├── requirements.txt         # Python dependencies
└── README.md
```

## How It Works

### NIQE Score
- Converts the image to grayscale
- Computes **MSCN (Mean Subtracted Contrast Normalized)** coefficients
- Extracts **36-dimensional NSS feature vectors** from image patches at 2 scales
- Compares the features to pretrained pristine statistics using a **Mahalanobis-like distance**
- **Lower score = better (more natural) quality**

### Depth Consistency Score
- Estimates depth using the **MiDaS** model (`Intel/dpt-hybrid-midas`)
- Evaluates three sub-metrics:
  - **Smoothness (30%)** — stability of depth gradients
  - **Edge Alignment (40%)** — overlap between RGB and depth edges (IoU)
  - **Distribution (30%)** — entropy of the depth histogram
- **Higher score (0–100) = better consistency**

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web Dashboard (NIQE)
```bash
streamlit run niqe_app.py
```
Upload images via the sidebar and view NIQE scores, statistics, and a visual grid.

### Web Dashboard (Depth Consistency)
```bash
streamlit run depth_app.py
```
Upload images to see depth maps, edge alignment, and consistency scores.

### Batch CLI (NIQE)
```bash
python niqe2.py
```
Scores all images in `./test_imgs/contaminated_Gen_3` and prints per-folder statistics.

### Single-Image CLI (NIQE)
```bash
python niqe.py
```
Runs NIQE on hardcoded test images defined in the `__main__` block.

## Dependencies

- Python 3.8+
- NumPy, SciPy, Pillow — core computation
- Streamlit, Pandas — web dashboard
- OpenCV (`opencv-python`) — edge detection for depth metrics
- PyTorch, Transformers — MiDaS depth estimation
- Matplotlib — depth map visualization
