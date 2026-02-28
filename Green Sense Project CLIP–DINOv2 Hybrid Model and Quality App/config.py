"""
Greenspace Quality Feature Pipeline - Configuration
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class Config:
    """Global configuration for the feature extraction pipeline."""
    
    # Data paths
    data_root: str = os.path.join(os.path.dirname(__file__), "Data")
    output_dir: str = os.path.join(os.path.dirname(__file__), "features")
    
    # Image processing
    image_size: int = 448  # For detection models (GroundingDINO, SAM)
    clip_image_size: int = 224  # CLIP ViT-B/32 requires 224x224
    batch_size: int = 32
    num_workers: int = 4
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # CLIP model
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    
    # DINOv2 model
    use_dino: bool = True  # Feature flag for DINO integration
    dino_model_name: str = "dinov2_vits14"  # 384-dim, lighter than base/large
    dino_image_size: int = 518  # DINOv2 optimal size (divisible by 14)
    
    # Scene prompts — 10 per class (30 total)
    # Indices: 
    # 0-9   = HEALTHY  (5 detailed + 5 short)
    # 10-19 = DRIED    (5 detailed + 5 short)
    # 20-29 = CONTAMINATED (5 detailed + 5 short)
    scene_prompts: List[str] = field(default_factory=lambda: [
        # --- HEALTHY (Indices 0-9) ---
        # Detailed
        "a pristine park with lush vibrant green grass and healthy trees",
        "a well-maintained urban garden with fresh green vegetation and no trash",
        "a photo of a healthy lawn with rich green color and dense texture",
        "an image of vigorous plant growth in a clean outdoor greenspace",
        "a beautiful green park scene, perfectly clean and well-watered",
        # Short
        "vibrant green grass",
        "healthy lush vegetation",
        "clean fresh park",
        "well watered green turf",
        "dense green canopy",
        
        # --- DRIED (Indices 10-19) ---
        # Detailed
        "a dried-out park with dead brown grass and yellowed vegetation",
        "an image of a drought-stricken greenspace with withered plants",
        "a photo of dead dry grass patches and brown leaves on the ground",
        "a neglected park with dying vegetation due to lack of water",
        "an outdoor area with drought-stressed, dried vegetation and little green color",
        # Short
        "dead brown grass",
        "withered yellow vegetation",
        "dried out plants",
        "scorched yellow field",
        "cracked dry soil with dead grass",
        
        # --- CONTAMINATED (Indices 20-29) ---
        # Detailed
        "a photo of a polluted urban greenspace with visible trash and litter on the ground",
        "a dirty park with scattered plastic bottles, bags and other waste among the plants",
        "an image of plastic bottles, bags, and debris polluting a park",
        "a neglected area with piles of rubbish and discarded man-made items",
        "a contaminated outdoor scene full of refuse and waste materials",
        # Short
        "trash and garbage",
        "litter on grass",
        "plastic waste pollution",
        "dirty trash filled park",
        "rubbish on ground",
    ])
    
    # Vegetation detection settings
    vegetation_queries: List[str] = field(default_factory=lambda: [
        "vegetation", "grass", "trees", "bushes", "plants"
    ])
    box_threshold: float = 0.35
    nms_iou_threshold: float = 0.5
    
    # Class mapping
    class_names: List[str] = field(default_factory=lambda: [
        "healthy", "dried", "contaminated"
    ])
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    
    def get_class_index(self, folder_name: str) -> int:
        """Get class index from folder name."""
        name = folder_name.lower()
        if name in self.class_names:
            return self.class_names.index(name)
        # Fallback for old typos
        if name in ["healty", "haelthy"]:
            return 0
        raise ValueError(f"Unknown class folder: {name}")
