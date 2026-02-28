[README_Repository.md](https://github.com/user-attachments/files/25621409/README_Repository.md)
# Green Sense Full Project Repository 🌿

Green Sense is a modular Computer Vision research project focused on
urban greenspace quality analysis.

The repository combines multiple independently built, trained, and
evaluated components that address different stages of the pipeline:

-   Synthetic image generation
-   Image quality validation
-   CNN baseline modeling
-   Vision Transformer modeling
-   Hybrid Vision-Language modeling (CLIP + DINOv2)

Each module was developed, trained, and tested separately to enable
controlled experimentation and clear benchmarking.

------------------------------------------------------------------------

## Project Objective

To classify urban greenspace imagery into three categories:

-   Healthy
-   Dried
-   Contaminated

Under both synthetic and street-camera viewpoints, while evaluating
model robustness under domain shift.

------------------------------------------------------------------------

# Repository Components

## 1️⃣ Synthetic Image Generation (FIBO)

Synthetic vegetation images are generated using a FIBO-based pipeline.

Purpose: - Create class-balanced datasets - Simulate vegetation states -
Enable controlled experiments

Includes: - Class-structured dataset generation - Evaluation using FID,
LPIPS, SSIM, PSNR - Experimental workflow for downstream training

This module feeds training data into the classification pipelines.

------------------------------------------------------------------------

## 2️⃣ Image Quality Filtering (NIQE + Depth Consistency)

Before training, generated images can be evaluated using:

-   NIQE (No-Reference Image Quality Evaluator)
-   Depth Consistency scoring using MiDaS

Purpose: - Remove low-quality synthetic samples - Improve dataset
realism - Measure structural coherence

Includes: - CLI batch scoring - Streamlit dashboard - Depth alignment
evaluation

------------------------------------------------------------------------

## 3️⃣ ResNet50 Baseline Module

CNN-based baseline model trained on synthetic dataset.

Characteristics: - ImageNet pretrained ResNet50 - Backbone frozen -
Fine-tuned classification head - 3-class Cross Entropy training - Train
/ Validate / Test split

Purpose: - Establish CNN baseline - Measure synthetic dataset
effectiveness - Compare against transformer architectures

Outputs: - Trained model weights - Metrics JSON - Confusion matrix -
Classification reports

------------------------------------------------------------------------

## 4️⃣ Vision Transformer (ViT-B16)

Transformer-based classifier trained on synthetic street-camera dataset.

Characteristics: - ViT-B16 (ImageNet pretrained) - 224x224 input
resolution - Data augmentation (flip, rotation, CLAHE,
brightness/contrast) - Early stopping on validation accuracy

Purpose: - Evaluate transformer robustness - Benchmark against
ResNet50 - Test performance under viewpoint domain shift

Reported synthetic test performance: - Accuracy: 0.905 - Macro F1:
0.902 - Cohen Kappa: 0.856 - ROC AUC: 0.972

------------------------------------------------------------------------

## 5️⃣ VLM Sense CLIP + DINOv2 Hybrid Pipeline

Feature-fusion classifier combining:

-   CLIP scene embeddings
-   CLIP vegetation crop embeddings
-   DINOv2 scene features
-   DINOv2 vegetation features
-   Color & texture statistics
-   Random Forest classifier
-   Prompt-based ensemble voting

Feature dimension: 1,795-dimensional vector

Decision strategy: - Random Forest confidence - CLIP prompt voting -
Majority and consensus overrides

Purpose: - Improve robustness beyond supervised-only models - Combine
semantic and texture reasoning - Provide explainable ensemble behavior

Includes: - Streamlit interface - Batch evaluation - Feature importance
visualization - Confusion matrix export

------------------------------------------------------------------------

# Experimental Design Philosophy

Each component was:

-   Implemented independently
-   Trained independently
-   Evaluated independently
-   Benchmarked against other modules

This allows: - Controlled ablation studies - Clear comparison between
CNN, Transformer, and Hybrid approaches - Isolated evaluation of
synthetic data impact - Modular reproducibility

------------------------------------------------------------------------

#  High-Level Workflow

1.  Generate synthetic dataset (FIBO)
2.  Optionally filter using NIQE + Depth metrics
3.  Train:
    -   ResNet50 baseline
    -   ViT-B16 transformer
    -   CLIP+DINO hybrid classifier
4.  Evaluate synthetic test performance
5.  Compare model robustness under viewpoint shift

------------------------------------------------------------------------

#  Environment

The project includes a full Conda environment specification:

Environment name: greenspace\
Python: 3.10\
Core frameworks: - PyTorch 2.5.1 (CUDA 12.1) - Torchvision 0.20.1 -
Transformers 4.57.6 - Scikit-learn 1.7.2 - OpenCV 4.12

Used for: - Synthetic dataset training - Transformer fine-tuning -
CLIP+DINO hybrid evaluation - Real-world transfer experiments

------------------------------------------------------------------------

#  Modular Structure

The repository is organized into independent submodules:

-   Synthetic Generation
-   NIQE Quality Analyzer
-   ResNet50 Module
-   ViT-B16 Module
-   VLM Sense Hybrid Pipeline

Each subfolder contains its own README with detailed instructions.

------------------------------------------------------------------------

# Key Characteristics

-   Modular design
-   Synthetic-to-real domain shift experimentation
-   CNN vs Transformer vs Vision-Language comparison
-   Explicit metric reporting
-   Reproducible Conda environment
-   Independent evaluation per component

------------------------------------------------------------------------

# Research Focus

This repository explores:

-   Synthetic data effectiveness
-   Domain adaptation challenges
-   Model robustness across viewpoints
-   Hybrid semantic + texture reasoning

