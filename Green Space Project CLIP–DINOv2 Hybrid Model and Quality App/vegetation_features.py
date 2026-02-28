"""
Greenspace Quality Feature Pipeline - Vegetation Feature Extraction
Extracts CLIP embeddings and hand-crafted color/texture features from vegetation regions.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import cv2
from torchvision import transforms


class VegetationFeatureExtractor:
    """Extract CLIP embeddings from vegetation crops."""
    
    def __init__(
        self,
        scene_extractor,  # Reuse SceneFeatureExtractor
        image_size: int = 224  # CLIP ViT-B/32 expects 224x224
    ):
        """
        Args:
            scene_extractor: SceneFeatureExtractor instance to reuse CLIP model
            image_size: Target size for vegetation crops
        """
        self.scene_extractor = scene_extractor
        self.image_size = image_size
        self.device = scene_extractor.device
        
        # CLIP preprocessing for crops
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        
        self.crop_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])
    
    def extract_crop_embeddings(
        self,
        image: Image.Image,
        boxes: List[Tuple[int, int, int, int]],
        masks: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Extract CLIP embeddings for vegetation crops.
        
        Args:
            image: PIL Image (RGB)
            boxes: List of (x_min, y_min, x_max, y_max) bounding boxes
            masks: Optional list of binary masks for tighter crops
        
        Returns:
            veg_embeddings: List of embeddings per region
            veg_embedding_mean: Mean embedding across regions (D,)
            veg_embedding_max: Element-wise max across regions (D,)
        """
        if len(boxes) == 0:
            zero_emb = self.scene_extractor.get_zero_embedding()
            return [], zero_emb, zero_emb
        
        crops = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # If mask available, compute tight bounding box
            if masks is not None and i < len(masks) and masks[i] is not None:
                mask = masks[i]
                # Find bounding box of non-zero mask pixels
                ys, xs = np.where(mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x1 = max(0, int(xs.min()))
                    y1 = max(0, int(ys.min()))
                    x2 = min(image.width, int(xs.max()) + 1)
                    y2 = min(image.height, int(ys.max()) + 1)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop region
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
        
        if len(crops) == 0:
            zero_emb = self.scene_extractor.get_zero_embedding()
            return [], zero_emb, zero_emb
        
        # Preprocess crops
        crop_tensors = torch.stack([self.crop_transform(c) for c in crops])
        
        # Get CLIP embeddings
        embeddings = self.scene_extractor.encode_images(crop_tensors)
        
        # Convert to list for per-region storage
        veg_embeddings = [embeddings[i] for i in range(len(embeddings))]
        
        # Aggregate
        veg_embedding_mean = embeddings.mean(dim=0)
        veg_embedding_max = embeddings.max(dim=0)[0]
        
        return veg_embeddings, veg_embedding_mean, veg_embedding_max
    

    
    def extract_dino_crop_embeddings(
        self,
        image: Image.Image,
        boxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extract DINOv2 embeddings for vegetation crops (if DINO is enabled).
        
        Args:
            image: PIL Image (RGB)
            boxes: List of (x_min, y_min, x_max, y_max) bounding boxes
        
        Returns:
            dino_embeddings: List of embeddings per crop
            dino_embedding_mean: Mean-pooled embedding across all crops
        """
        if not self.scene_extractor.use_dino or self.scene_extractor.dino_extractor is None:
            # Return None if DINO not enabled
            return [], None
        
        # Use DINO extractor's batch method for efficiency
        crop_embs, mean_emb = self.scene_extractor.dino_extractor.extract_batch_crops(
            image, boxes, batch_size=8
        )
        
        return crop_embs, mean_emb



class ColorTextureAnalyzer:
    """Analyze color and texture features of vegetation regions."""
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def analyze_region(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Analyze color and texture features for a single region.
        
        Args:
            image: PIL Image (RGB)
            box: (x_min, y_min, x_max, y_max) bounding box
            mask: Optional binary mask for the region
        
        Returns:
            Dictionary of feature values
        """
        x1, y1, x2, y2 = box
        image_np = np.array(image)
        
        # Get pixels in region
        if mask is not None:
            # Use mask to select pixels
            region_mask = mask[y1:y2, x1:x2]
            region = image_np[y1:y2, x1:x2]
            
            # Get masked pixels
            if region_mask.sum() > 0:
                pixels = region[region_mask > 0]
            else:
                pixels = region.reshape(-1, 3)
        else:
            # Use all pixels in box
            region = image_np[y1:y2, x1:x2]
            pixels = region.reshape(-1, 3)
        
        if len(pixels) == 0:
            return self._empty_features()
        
        # Convert to float
        pixels = pixels.astype(np.float32)
        
        # Extract features
        features = {}
        
        # Normalize to 0-1 range
        pixels = pixels / 255.0
        
        # Mean RGB values (normalized)
        features['mean_R'] = float(np.mean(pixels[:, 0]))
        features['mean_G'] = float(np.mean(pixels[:, 1]))
        features['mean_B'] = float(np.mean(pixels[:, 2]))
        
        # Green ratio: G / (R + B + epsilon) - now properly scaled
        R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        green_ratios = G / (R + B + 1e-6)
        features['green_ratio'] = float(np.clip(np.mean(green_ratios), 0, 2))
        

        
        # Intensity variance (texture roughness)
        intensity = 0.299 * R + 0.587 * G + 0.114 * B
        features['intensity_variance'] = float(np.var(intensity))
        
        # Edge density using Canny
        features['edge_density'] = self._compute_edge_density(image_np, box, mask)
        
        return features
    
    def _compute_edge_density(
        self,
        image_np: np.ndarray,
        box: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None
    ) -> float:
        """Compute edge density using Canny edge detector."""
        x1, y1, x2, y2 = box
        
        # Get region
        region = image_np[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Compute density
        if mask is not None:
            region_mask = mask[y1:y2, x1:x2]
            if region_mask.sum() > 0:
                edge_pixels = np.sum((edges > 0) & (region_mask > 0))
                total_pixels = np.sum(region_mask > 0)
                return float(edge_pixels / total_pixels) if total_pixels > 0 else 0.0
        
        # Without mask, use full region
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        return float(edge_pixels / total_pixels) if total_pixels > 0 else 0.0
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dictionary."""
        return {
            'mean_R': 0.0,
            'mean_G': 0.0,
            'mean_B': 0.0,
            'green_ratio': 0.0,
            'intensity_variance': 0.0,
            'edge_density': 0.0,
        }
    
    def aggregate_stats(
        self,
        image: Image.Image,
        boxes: List[Tuple[int, int, int, int]],
        masks: Optional[List[np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Aggregate color/texture statistics across all vegetation regions.
        
        Args:
            image: PIL Image (RGB)
            boxes: List of bounding boxes
            masks: Optional list of masks
        
        Returns:
            Dictionary with per-image aggregated statistics
        """
        if len(boxes) == 0:
            # Return empty stats
            return {
                'mean_G': 0.0, 'std_G': 0.0,
                'mean_green_ratio': 0.0, 'std_green_ratio': 0.0,

                'mean_edge_density': 0.0, 'std_edge_density': 0.0,
                'mean_intensity_variance': 0.0, 'std_intensity_variance': 0.0,
                'num_vegetation_regions': 0,
                'vegetation_coverage': 0.0,
            }
        
        # Calculate Vegetation Coverage % (Pixel-wise Precision)
        # Use HSV color masking to get exact vegetation pixels, not just bounding boxes
        img_np = np.array(image)
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # 1. Green Range (Hue 30-90)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
        
        # 2. Dried/Yellow Range (Hue 10-30)
        lower_yellow = np.array([10, 40, 40])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        mask_combined = cv2.bitwise_or(mask_green, mask_yellow)
        
        # Calculate ratio of vegetation pixels
        coverage_pct = float(np.sum(mask_combined > 0) / mask_combined.size)
        
        # Analyze each region
        region_features = []
        for i, box in enumerate(boxes):
            mask = masks[i] if masks is not None and i < len(masks) else None
            features = self.analyze_region(image, box, mask)
            region_features.append(features)
        
        # Aggregate
        aggregated = {}
        
        # Number of regions

        aggregated['num_vegetation_regions'] = len(region_features)
        aggregated['vegetation_coverage'] = coverage_pct
        
        # Compute mean and std for each feature
        feature_keys = ['mean_G', 'green_ratio', 'edge_density', 'intensity_variance']
        
        for key in feature_keys:
            values = [f[key] for f in region_features]
            aggregated[f'mean_{key}'] = float(np.mean(values))
            aggregated[f'std_{key}'] = float(np.std(values)) if len(values) > 1 else 0.0
        
        return aggregated
