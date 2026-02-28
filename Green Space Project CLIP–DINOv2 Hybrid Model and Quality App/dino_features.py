"""
Greenspace Quality Feature Pipeline - DINOv2 Feature Extraction

DINOv2 is a self-supervised vision transformer from Meta AI that provides
powerful visual features without requiring text descriptions (unlike CLIP).
It excels at capturing fine-grained visual details and object representations.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from PIL import Image
import numpy as np
from torchvision import transforms


class DinoFeatureExtractor:
    """Extract features using DINOv2 vision transformer."""
    
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: str = "cuda",
        image_size: int = 518
    ):
        """
        Initialize DINOv2 feature extractor.
        
        Args:
            model_name: DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14')
            device: Device to run model on ('cuda' or 'cpu')
            image_size: Input image size (DINOv2 works best with 518x518)
        """
        self.device = device
        self.image_size = image_size
        self.model_name = model_name
        
        # Load DINOv2 model from torch hub
        print(f"Loading DINOv2 model: {model_name}...")
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
            self.model = self.model.to(device)
            self.model.eval()
            
            # Get embedding dimension based on model
            if 'vits' in model_name:
                self.embedding_dim = 384
            elif 'vitb' in model_name:
                self.embedding_dim = 768
            elif 'vitl' in model_name:
                self.embedding_dim = 1024
            elif 'vitg' in model_name:
                self.embedding_dim = 1536
            else:
                self.embedding_dim = 384  # Default to small
                
            print(f"DINOv2 loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"Error loading DINOv2: {e}")
            print("Falling back to CPU mode...")
            self.device = "cpu"
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
            self.model = self.model.to("cpu")
            self.model.eval()
        
        # Image preprocessing for DINOv2
        # DINOv2 expects ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract_scene_features(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract scene-level features from a batch of images.
        
        Args:
            images: Batch of preprocessed images (B, C, H, W)
                   Should be normalized with ImageNet mean/std
        
        Returns:
            scene_embeddings: L2-normalized CLS token embeddings (B, D)
        """
        images = images.to(self.device)
        
        # Get CLS token embeddings (global image representation)
        features = self.model(images)
        
        # L2 normalize for consistency with CLIP
        scene_embeddings = F.normalize(features, dim=-1)
        
        return scene_embeddings
    

    @torch.no_grad()
    def extract_batch_crops(
        self,
        image: Image.Image,
        boxes: List[Tuple[int, int, int, int]],
        batch_size: int = 8
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extract DINOv2 embeddings for vegetation crops with batching (faster).
        
        Args:
            image: PIL Image (RGB)
            boxes: List of (x_min, y_min, x_max, y_max) bounding boxes
            batch_size: Number of crops to process simultaneously
        
        Returns:
            crop_embeddings: List of embeddings per crop
            mean_embedding: Mean-pooled embedding across all crops
        """
        if not boxes:
            zero_emb = torch.zeros(self.embedding_dim, device=self.device)
            return [], zero_emb
        
        all_crops = []
        
        # Prepare all crops
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            crop = image.crop((x_min, y_min, x_max, y_max))
            crop_tensor = self.preprocess(crop)
            all_crops.append(crop_tensor)
        
        # Process in batches
        crop_embeddings = []
        for i in range(0, len(all_crops), batch_size):
            batch = all_crops[i:i + batch_size]
            batch_tensor = torch.stack(batch).to(self.device)  # (B, C, H, W)
            
            # Extract features
            features = self.model(batch_tensor)  # (B, D)
            features = F.normalize(features, dim=-1)
            
            crop_embeddings.extend(features.unbind(0))
        
        # Mean pooling
        if crop_embeddings:
            mean_embedding = torch.stack(crop_embeddings).mean(dim=0)
            mean_embedding = F.normalize(mean_embedding, dim=-1)
        else:
            mean_embedding = torch.zeros(self.embedding_dim, device=self.device)
        
        return crop_embeddings, mean_embedding
    
    def get_zero_embedding(self) -> torch.Tensor:
        """Return a zero embedding for images with no vegetation."""
        return torch.zeros(self.embedding_dim, device=self.device)
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a PIL image for DINOv2.
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            Preprocessed tensor (C, H, W)
        """
        return self.preprocess(image)
