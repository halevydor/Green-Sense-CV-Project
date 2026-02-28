"""
Greenspace Quality Feature Pipeline - Dataset and DataLoaders
"""
import os
from typing import Tuple, Optional, List
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class GreenspaceDataset(Dataset):
    """Dataset for greenspace quality classification images."""
    
    def __init__(
        self,
        data_root: str,
        split: str,
        transform: Optional[transforms.Compose] = None,
        config = None
    ):
        """
        Args:
            data_root: Path to Data folder containing train/val/test
            split: One of 'train', 'val', 'test'
            transform: Torchvision transforms to apply
            config: Config object with class mapping
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.config = config
        
        # Collect all image paths and labels
        self.samples: List[Tuple[str, int]] = []
        self._load_samples()
    
    def _load_samples(self):
        """Walk the folder tree and collect (image_path, label) pairs."""
        split_dir = self.data_root / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory does not exist: {split_dir}")
        
        # Valid image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # Walk through class folders
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Get class index (handles typos like haelthy, healty, healthy)
            try:
                if self.config:
                    label = self.config.get_class_index(class_name)
                else:
                    label = self._default_class_index(class_name)
            except ValueError:
                print(f"Warning: Skipping unknown class folder: {class_name}")
                continue
            
            # Collect all images in this class folder
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), label))
        
        print(f"Loaded {len(self.samples)} images from {self.split} split")
    
    def _default_class_index(self, class_name: str) -> int:
        """Default class mapping if no config provided."""
        name_lower = class_name.lower()
        if name_lower in ['healthy', 'healty', 'haelthy']:
            return 0
        elif name_lower == 'dried':
            return 1
        elif name_lower == 'contaminated':
            return 2
        else:
            raise ValueError(f"Unknown class: {class_name}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            image_tensor: Preprocessed image (C, H, W)
            label: Integer class label (0, 1, or 2)
            image_path: Original image path for tracking
        """
        image_path, label = self.samples[index]
        
        # Load image as RGB
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        return image_tensor, label, image_path


def get_clip_preprocess(image_size: int = 448) -> transforms.Compose:
    """Get CLIP-compatible preprocessing transforms."""
    # CLIP normalization values
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])



