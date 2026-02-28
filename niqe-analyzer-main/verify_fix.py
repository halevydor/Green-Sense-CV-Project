from depth_consistency import DepthConsistencyEvaluator
from PIL import Image
import numpy as np

# Path to a real test image
img_path = r"C:\Users\noama\OneDrive\Desktop\מערכות תבוניות\סמסטר ג\Computer Vision\CV project\niqe code\test_imgs\dry1.png"

print(f"Testing on {img_path}")
try:
    evaluator = DepthConsistencyEvaluator()
    img_pil, img_np = evaluator.load_image(img_path)
    print(f"Image size: {img_pil.size}")
    
    depth_map = evaluator.estimate_depth(img_pil)
    print(f"Depth map shape: {depth_map.shape}")
    
    if depth_map.shape == img_np.shape[:2]:
        print("PASS: Depth map shape matches image shape.")
    else:
        print("FAIL: Shapes do not match!")

    score, metrics = evaluator.calculate_consistency_score(img_np, depth_map)
    print("Scores calculated successfully:")
    print(metrics)

except Exception as e:
    print(f"FAIL: Error occurred: {e}")
