# ResNet50 -- Training & Evaluation on Generated Dataset

Green Sense Project

This module contains the full ResNet50 workflow for the Green Sense
project, including:

-   Fine-tuning on generated synthetic images (3 classes)
-   Evaluation on generated dataset
-   Metric reporting and output export
-   Notebook-based experimentation

------------------------------------------------------------------------

## Included Files

### 1. Resnet50FineTuning.py

Fine-tunes a pre-trained ResNet50 model on the generated dataset.

Key characteristics:

-   Loads a base ResNet50 model
-   Freezes backbone layers
-   Trains only the fully connected classification head
-   Works on 3 classes:
    -   Healthy
    -   Dried
    -   Contaminated
-   Uses Train / Validate / Test split structure
-   Saves:
    -   Fine-tuned model
    -   Metrics JSON
    -   Training history JSON

Output directory: outputs_resnet_finetune_generated/

------------------------------------------------------------------------

### 2. Resnet50RunModelOnGenerated.py

Evaluates a trained ResNet50 model exclusively on generated images.

Expected dataset structure:

root/ Healthy/ Dried/ Contaminated/

Computed Metrics:

-   Accuracy
-   Macro Precision
-   Macro Recall
-   Macro F1
-   Cohen's Kappa
-   ROC-AUC (Macro OVR)
-   Confusion Matrix
-   Classification Report

Outputs: outputs_generated_eval/ metrics_generated.json
predictions_generated.csv

------------------------------------------------------------------------

### 3. GreenSpaceQualityModel.ipynb

Interactive notebook version of the Green Sense classification pipeline.

Used for:

-   Exploratory experimentation
-   Visual inspection
-   Training monitoring
-   Quick inference testing
-   Debugging transformations and augmentations

Recommended for experimentation before production script execution.

------------------------------------------------------------------------

## Model Architecture

Base architecture: ResNet50\
Pretraining: ImageNet\
Head: Fully connected layer → 3 output classes\
Loss: Cross Entropy\
Optimizer: Adam\
Learning Rate: 1e-4\
Input size: 224x224 RGB

ImageNet normalization applied.

------------------------------------------------------------------------

## Dataset Structure for Fine-Tuning

Expected directory structure:

GreenSenseGeneratedPicturesForFineTunning/ Healthy/ Train/ Validate/
Test/ Dried/ Train/ Validate/ Test/ Contaminated/ Train/ Validate/ Test/

------------------------------------------------------------------------

## How to Run

Activate environment:

conda activate greenspace

Fine-tune model:

python Resnet50FineTuning.py

Evaluate trained model:

python Resnet50RunModelOnGenerated.py

------------------------------------------------------------------------

## Purpose in the Green Sense Project

The ResNet50 pipeline serves as:

-   A CNN baseline for comparison
-   A domain shift benchmark (close-up → street camera)
-   A reference model against Vision Transformers
-   A controlled experiment to measure synthetic data effectiveness

------------------------------------------------------------------------

## Notes

-   Designed specifically for generated dataset evaluation.
-   Real-world transfer performance is lower than the CLIP--DINOv2
    hybrid configuration.
-   Backbone freezing ensures controlled fine-tuning rather than full
    retraining.

------------------------------------------------------------------------

## Related Components

-   ViT-B16 fine-tuning module
-   CLIP--DINOv2 hybrid architecture
-   Synthetic dataset generation pipeline
-   NIQE-based quality filtering

See main project README for full experimental context.
