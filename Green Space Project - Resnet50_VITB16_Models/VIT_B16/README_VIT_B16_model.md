# Green Sense -- ViT-B16 Model

This folder contains the fine-tuned Vision Transformer (ViT-B16) model
used in the Green Sense project for three-class vegetation quality
classification:

-   Healthy
-   Dried
-   Contaminated

## Model Overview

Architecture: ViT-B16\
Input Size: 224x224 RGB\
Pretraining: ImageNet\
Fine-tuning: Synthetic street-camera dataset\
Framework: PyTorch

The model is trained to classify urban green spaces from municipal-style
street camera viewpoints.

------------------------------------------------------------------------

## Training Setup

-   Loss: Cross Entropy
-   Optimizer: Adam
-   Learning Rate: 1e-4
-   Early stopping based on validation accuracy
-   Data augmentation:
    -   Horizontal flip
    -   90° rotation
    -   Brightness & contrast adjustments
    -   CLAHE histogram equalization

Dataset split: - 70% Training - 10% Validation - 20% Test

------------------------------------------------------------------------

## Performance (Synthetic Test Set)

Accuracy: 0.905\
Macro F1: 0.902\
Cohen Kappa: 0.856\
ROC AUC: 0.972

The model significantly outperforms the ResNet50 baseline under
viewpoint domain shift.

------------------------------------------------------------------------

## Inference Example

``` python
import torch
from torchvision import models, transforms
from PIL import Image

model = models.vit_b_16()
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

img = Image.open("example.jpg")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    preds = model(x)
    print(preds.softmax(dim=1))
```

------------------------------------------------------------------------

## Notes

-   Performs strongly on synthetic street-camera imagery.
-   Real-world performance improves further with the CLIP--DINOv2 hybrid
    configuration.
-   Recommended for benchmarking transformer robustness under domain
    shift.
