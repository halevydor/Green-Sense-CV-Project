# 🌿 Green Sense Project --- FIBO Image Generation

## Overview

This notebook presents the image generation component of the Green Sense
project.\
The goal is to generate synthetic environmental images representing
different vegetation states using a generative model (FIBO), and
evaluate their quality for downstream computer vision tasks.

The generated images are used to augment datasets for classification of
greenspace quality into:

-   Healthy\
-   Dried\
-   Contaminated

------------------------------------------------------------------------

## Project Goals

1.  Generate high-quality synthetic vegetation images using FIBO.
2.  Maintain structural and color consistency with real-world
    environmental data.
3.  Evaluate realism and diversity of generated samples.
4.  Use synthetic data for improving classifier robustness.

------------------------------------------------------------------------

## Methodology

### Image Generation (FIBO)

The notebook uses a generative pipeline based on FIBO to:

-   Preserve structural consistency of vegetation
-   Control semantic attributes (healthy / dried / contaminated)
-   Maintain photorealistic textures
-   Generate diverse environmental conditions

Generation parameters include: - Prompt conditioning - Controlled
randomness - Seed reproducibility - Resolution handling

------------------------------------------------------------------------

## Dataset Structure

Generated images follow the class-based structure:

Data/ ├── Healthy/ ├── Dried/ └── Contaminated/

Each class contains synthetic images aligned with real-world
distributions.

------------------------------------------------------------------------

## Evaluation Metrics

The notebook supports evaluation using perceptual and distributional
metrics such as:

-   FID (Fréchet Inception Distance)
-   LPIPS
-   SSIM / MS-SSIM
-   PSNR

Lower FID and LPIPS indicate better realism.\
Higher SSIM and PSNR indicate better structural preservation.

------------------------------------------------------------------------

## Experimental Workflow

1.  Generate synthetic samples per class
2.  Compare synthetic vs real distribution
3.  Analyze feature-space similarity
4.  Prepare images for classifier training

------------------------------------------------------------------------

## How to Run

1.  Open the notebook:

Green Sense Project - FIBO Image Generation.ipynb

2.  Install required dependencies if needed:

pip install -r requirements.txt

3.  Run cells sequentially to:
    -   Generate images
    -   Save outputs
    -   Compute evaluation metrics

------------------------------------------------------------------------

## Outputs

The notebook produces:

-   Generated synthetic images
-   Evaluation metric reports
-   Optional comparison visualizations

------------------------------------------------------------------------

## Integration with Green Sense Pipeline

This notebook connects to the broader Green Sense CV framework:

-   Synthetic data feeds into the classification pipeline
-   Improves model generalization
-   Enables controlled experiments across vegetation states

------------------------------------------------------------------------

## Notes

-   Ensure GPU acceleration if generating large batches.
-   Use consistent seeds for reproducible experiments.
-   Validate distribution shift before large-scale training.

------------------------------------------------------------------------

## Author

Green Sense Research Project\
Environmental AI & Computer Vision
