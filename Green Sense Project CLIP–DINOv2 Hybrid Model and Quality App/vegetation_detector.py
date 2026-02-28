"""
Greenspace Quality Feature Pipeline - Vegetation Detection (HSV Color-Based)
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image
import cv2


def mask_road_signs(image: Image.Image, min_area_ratio: float = 0.002, max_area_ratio: float = 0.15) -> Image.Image:
    """
    Detect and inpaint road signs/urban signage to remove them from scene analysis.
    
    Uses HSV color detection to find sign-colored regions (red, blue, bright white),
    filters by rectangular shape and size, then inpaints them so CLIP doesn't
    misinterpret signs as contamination.
    
    Args:
        image: PIL Image (RGB)
        min_area_ratio: Minimum sign area as ratio of image (filters noise)
        max_area_ratio: Maximum sign area as ratio of image (filters large false positives)
    
    Returns:
        Cleaned PIL Image with road signs inpainted
    """
    img_np = np.array(image)
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, w = img_np.shape[:2]
    img_area = h * w
    min_area = int(img_area * min_area_ratio)
    max_area = int(img_area * max_area_ratio)
    
    # === Step 1: Detect sign-colored regions ===
    
    # Red signs (Stop signs, warning signs, prohibition signs)
    # Red wraps around in HSV: Hue 0-10 and 170-180
    mask_red1 = cv2.inRange(img_hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(img_hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Blue signs (Info signs, highway signs, parking signs)
    mask_blue = cv2.inRange(img_hsv, np.array([100, 70, 50]), np.array([130, 255, 255]))
    
    # Bright white / metallic (sign faces, poles, reflective surfaces)
    # High value, low saturation = white/silver
    mask_white = cv2.inRange(img_hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))
    
    # Combine all sign color masks
    sign_mask = cv2.bitwise_or(mask_red, mask_blue)
    sign_mask = cv2.bitwise_or(sign_mask, mask_white)
    
    # Morphological cleanup: close small gaps, remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    sign_mask = cv2.morphologyEx(sign_mask, cv2.MORPH_CLOSE, kernel)
    sign_mask = cv2.morphologyEx(sign_mask, cv2.MORPH_OPEN, kernel)
    
    # === Step 2: Filter by shape (rectangular) and size ===
    
    contours, _ = cv2.findContours(sign_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Build inpainting mask — only rectangular, sign-sized objects
    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        
        # Check rectangularity: bounding rect area vs contour area
        x, y, bw, bh = cv2.boundingRect(cnt)
        rect_area = bw * bh
        if rect_area == 0:
            continue
        
        rectangularity = area / rect_area  # 1.0 = perfect rectangle
        
        # Check aspect ratio — signs are typically not extremely elongated
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        
        # Accept if reasonably rectangular (>40% fill) and not too elongated (<5:1)
        if rectangularity > 0.4 and aspect < 5.0:
            # Expand the bounding box slightly to cover edges
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            inpaint_mask[y1:y2, x1:x2] = 255
    
    # === Step 3: Inpaint detected sign regions ===
    
    if np.sum(inpaint_mask) == 0:
        # No signs detected — return original
        return image
    
    # Convert RGB to BGR for OpenCV inpainting
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Telea inpainting: fills regions with surrounding texture
    inpainted_bgr = cv2.inpaint(img_bgr, inpaint_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    
    # Convert back to RGB PIL
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(inpainted_rgb)


class VegetationDetector:
    """Detect vegetation regions using HSV color-based analysis."""
    
    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: Device (kept for API compatibility)
        """
        self.device = device
        print("Using lightweight color-based vegetation detection")
        

    
    def detect_vegetation(
        self,
        image: Image.Image
    ) -> List[Dict[str, Any]]:
        """
        Detect vegetation regions in an image using HSV color analysis.
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            List of detections, each with:
                - box: (x_min, y_min, x_max, y_max) in pixel coordinates
                - score: confidence score
                - label: detected class label
        """
        img_np = np.array(image)
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # Green range (Hue 30-90)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
        
        # Yellow/Brown range (Hue 10-30) for dried vegetation
        lower_yellow = np.array([10, 40, 40])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        
        # Combine
        mask = cv2.bitwise_or(mask_green, mask_yellow)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        img_area = image.width * image.height
        min_area = img_area * 0.01  # 1% min area to avoid noise
        
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append({
                    'box': (x, y, x+w, y+h),
                    'score': 0.85, 
                    'label': 'vegetation'
                })
        
        return detections
