# Green Sense -- Conda Environment

This file defines the full Conda environment required to reproduce the
Green Sense experiments, including model training, evaluation, and
inference.

## Environment Name

`greenspace`

## Python Version

Python 3.10

## Core Frameworks

-   PyTorch 2.5.1 (CUDA 12.1)
-   Torchvision 0.20.1
-   Transformers 4.57.6
-   Scikit-learn 1.7.2
-   OpenCV 4.12
-   NumPy 2.1
-   Pandas 2.3
-   Matplotlib 3.10

The environment supports: - Vision Transformer (ViT-B16) - CLIP & DINOv2
hybrid pipeline - Synthetic dataset preprocessing - NIQE quality
filtering - Training and evaluation workflows

------------------------------------------------------------------------

## Setup Instructions

### 1. Create the environment

``` bash
conda env create -f greenspace_full.yaml
```

### 2. Activate the environment

``` bash
conda activate greenspace
```

### 3. Verify GPU support (optional)

``` python
import torch
print(torch.cuda.is_available())
```

Expected output: `True` if CUDA is properly configured.

------------------------------------------------------------------------

## Notes

-   The environment includes CUDA 12.1 builds of PyTorch.
-   Ensure your GPU drivers are compatible.
-   If running on CPU only, replace torch with a CPU build.
-   The file includes JupyterLab for interactive experimentation.

------------------------------------------------------------------------

## Reproducibility

This environment was used for:

-   Synthetic street-camera dataset training
-   Vision Transformer fine-tuning
-   CLIP--DINOv2 hybrid evaluation
-   Real-world transfer experiments
