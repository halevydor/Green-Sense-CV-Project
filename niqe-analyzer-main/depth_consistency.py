import torch
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt

class DepthConsistencyEvaluator:
    def __init__(self, model_type='Intel/dpt-hybrid-midas'):
        """
        Initialize the DepthConsistencyEvaluator with a MiDaS model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading depth model ({model_type}) on {self.device}...")
        try:
            self.estimator = pipeline(task="depth-estimation", model=model_type, device=0 if self.device == "cuda" else -1)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.estimator = None

    def load_image(self, image_path_or_file):
        """
        Load an image from a path or file object (for Streamlit).
        Returns a PIL Image and a numpy RGB array.
        """
        if isinstance(image_path_or_file, str):
            img = Image.open(image_path_or_file).convert("RGB")
        else:
            img = Image.open(image_path_or_file).convert("RGB")
        return img, np.array(img)

    def estimate_depth(self, img_pil):
        """
        Estimate depth map from a PIL image.
        Returns the depth map as a normalized numpy array (0-1).
        """
        if self.estimator is None:
            return None
        
        result = self.estimator(img_pil)
        
        # Use 'predicted_depth' if available for raw values
        if 'predicted_depth' in result:
             depth_tensor = result['predicted_depth']
             if isinstance(depth_tensor, torch.Tensor):
                 depth_map = depth_tensor.cpu().numpy().squeeze()
             else:
                 depth_map = np.array(depth_tensor).squeeze()
        else:
            # Fallback to image
            depth_map = np.array(result['depth'])
            
        # Resize to original image size
        # img_pil.size is (width, height)
        if depth_map.shape[:2] != img_pil.size[::-1]:
            depth_map = cv2.resize(depth_map, img_pil.size, interpolation=cv2.INTER_CUBIC)
            
        # Normalize to 0-1 range for consistent metric calculation
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 0:
            depth_norm = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_map)
            
        return depth_norm

    def compute_smoothness_score(self, depth_map):
        """
        Compute Smoothness Score (30%) based on local depth gradient variance.
        Low variance in smooth areas is good, but we want to punish meaningful noise.
        Actually, 'smoothness' usually implies low gradient energy in flat areas.
        A higher score means BETTER consistency (smoother depth transitions).
        """
        # Calculate gradients using Sobel
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        sobelx = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Variance of gradients: High variance = chaotic depth / noise.
        # We want to invert this: Lower gradient variance is better (more consistent).
        grad_var = np.var(magnitude)
        
        # Normalize score: This is a heuristic scaling.
        # A variance of 0 would be perfect score. 
        # Typically gradient variance can be high. Let's map it roughly.
        # Max reasonable variance for normalized depth * 255 is surprisingly high.
        # Let's effectively measure 'smoothness' as 1 / (1 + variance) scaled.
        
        score = 100 / (1 + grad_var * 0.1) # Empirical scaling
        return np.clip(score, 0, 100)

    def compute_edge_alignment_score(self, img_rgb, depth_map):
        """
        Compute Edge Alignment Score (40%).
        Compares RGB edges (Canny) with Depth edges.
        Higher IoU/Overlap means depth boundaries respect object boundaries.
        """
        # RGB Edges
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges_rgb = cv2.Canny(gray, 100, 200)
        
        # Depth Edges
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        # Depth maps are often smoother, so we might need lower thresholds or normalization
        # Using automatic Canny or fixed lower thresholds for depth
        edges_depth = cv2.Canny(depth_uint8, 50, 150)
        
        # Dilate edges slightly to allow for minor misalignment
        kernel = np.ones((3,3), np.uint8)
        edges_rgb_dil = cv2.dilate(edges_rgb, kernel, iterations=1)
        edges_depth_dil = cv2.dilate(edges_depth, kernel, iterations=1)
        
        # Interaction (Overlap)
        intersection = np.logical_and(edges_rgb_dil > 0, edges_depth_dil > 0)
        union = np.logical_or(edges_rgb_dil > 0, edges_depth_dil > 0)
        
        if np.sum(union) == 0:
            return 0.0
            
        iou = np.sum(intersection) / np.sum(union)
        
        # IoU for edges is typically low (lines are thin). 
        # An IoU of 0.1-0.2 is actually quite good for edge alignment.
        # Scale: 0.2 IoU -> 100 score? 
        score = (iou * 5) * 100 
        return np.clip(score, 0, 100)

    def compute_distribution_score(self, depth_map):
        """
        Compute Distribution Quality Score (30%).
        Evaluates depth histogram spread / entropy.
        We want to avoid 'flat' images (over-concentrated) or single-plane depth.
        Higher entropy = richer depth information (generally consistent with a real 3D scene).
        """
        hist, _ = np.histogram(depth_map, bins=256, range=(0, 1), density=True)
        # Add small epsilon to avoid log(0)
        hist += 1e-7
        hist /= hist.sum() # Re-normalize
        
        entropy = -np.sum(hist * np.log2(hist))
        
        # Max entropy for 256 bins is 8.
        # A completely flat image has entropy 0.
        # We consider higher entropy better for 'Quality/Distribution'.
        score = (entropy / 8.0) * 100
        return np.clip(score, 0, 100)

    def calculate_consistency_score(self, img_rgb, depth_map):
        """
        Combine metrics into a final score.
        Weights: Smoothness (30%), Edge (40%), Distribution (30%).
        """
        s_score = self.compute_smoothness_score(depth_map)
        e_score = self.compute_edge_alignment_score(img_rgb, depth_map)
        d_score = self.compute_distribution_score(depth_map)
        
        final_score = (s_score * 0.30) + (e_score * 0.40) + (d_score * 0.30)
        
        return final_score, {
            "Total Score": final_score,
            "Smoothness Score": s_score,
            "Edge Alignment Score": e_score,
            "Distribution Score": d_score
        }

if __name__ == "__main__":
    # Test block
    evaluator = DepthConsistencyEvaluator()
    print("Evaluator initialized.")
    # Add dummy test if needed
