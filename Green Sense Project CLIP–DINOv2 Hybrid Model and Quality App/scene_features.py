"""
Greenspace Quality Feature Pipeline - CLIP Scene Understanding
"""
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
import open_clip


class SceneFeatureExtractor:
    """Extract scene-level features using CLIP and optionally DINOv2."""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
        scene_prompts: Optional[List[str]] = None,
        use_dino: bool = False,
        dino_model_name: str = "dinov2_vits14",
        dino_image_size: int = 518
    ):
        """
        Args:
            model_name: CLIP model architecture name
            pretrained: Pretrained weights identifier
            device: Device to run model on
            scene_prompts: List of text prompts for scene classification
            use_dino: Whether to also load DINOv2 extractor
            dino_model_name: DINOv2 model name
            dino_image_size: DINOv2 input image size
        """
        self.device = device
        self.use_dino = use_dino
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name} ({pretrained})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Default scene prompts
        if scene_prompts is None:
            scene_prompts = [
                "a clean, well maintained park with healthy green vegetation",
                "a park with dry, yellow or brown vegetation",
                "a park with visible trash, litter or contamination on the ground",
                "an urban street with little vegetation",
                "an empty field with dried grass",
            ]
        
        self.scene_prompts = scene_prompts
        self.prompt_embeddings = self._encode_prompts(scene_prompts)
        
        # Get embedding dimension
        self.embedding_dim = self.prompt_embeddings.shape[-1]
        print(f"CLIP loaded. Embedding dimension: {self.embedding_dim}")
        
        # Optionally load DINOv2
        self.dino_extractor = None
        if use_dino:
            from dino_features import DinoFeatureExtractor
            print("Loading DINOv2 extractor...")
            self.dino_extractor = DinoFeatureExtractor(
                model_name=dino_model_name,
                device=device,
                image_size=dino_image_size
            )
            print(f"DINOv2 embedding dimension: {self.dino_extractor.embedding_dim}")
    
    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts and normalize."""
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(self.device)
            text_embeddings = self.model.encode_text(tokens)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
        return text_embeddings
    
    @torch.no_grad()
    def extract_scene_features(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract scene-level features from a batch of images.
        
        Args:
            images: Batch of preprocessed images (B, C, H, W)
                    Should be normalized with CLIP mean/std
        
        Returns:
            scene_embeddings: L2-normalized image embeddings (B, D)
            prompt_scores: Cosine similarity to each prompt (B, num_prompts)
        """
        images = images.to(self.device)
        
        # Get image embeddings
        image_embeddings = self.model.encode_image(images)
        scene_embeddings = F.normalize(image_embeddings, dim=-1)
        
        # Compute similarity to prompts
        prompt_scores = scene_embeddings @ self.prompt_embeddings.T
        
        return scene_embeddings, prompt_scores
    
    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to CLIP embeddings (without prompt scores).
        
        Args:
            images: Batch of preprocessed images (B, C, H, W)
        
        Returns:
            embeddings: L2-normalized image embeddings (B, D)
        """
        images = images.to(self.device)
        embeddings = self.model.encode_image(images)
        return F.normalize(embeddings, dim=-1)
    

    
    @torch.no_grad()
    def extract_dino_scene_features(
        self,
        images: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Extract DINOv2 scene-level features from a batch of images.
        
        Args:
            images: Batch of preprocessed images (B, C, H, W)
                    Should be preprocessed with DINO preprocess transform
        
        Returns:
            dino_embeddings: L2-normalized scene embeddings (B, D_dino) or None if DINO not enabled
        """
        if not self.use_dino or self.dino_extractor is None:
            return None
        
        return self.dino_extractor.extract_scene_features(images)
    
    def get_zero_embedding(self) -> torch.Tensor:
        """Return a zero embedding for images with no vegetation (CLIP)."""
        return torch.zeros(self.embedding_dim, device=self.device)
    

