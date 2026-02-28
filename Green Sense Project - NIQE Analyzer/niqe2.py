


import os
import numpy as np
from PIL import Image
from niqe import niqe

search_dir = './test_imgs/contaminated_Gen_3'
count = 0
valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

from collections import defaultdict

print(f"Scanning directory: {search_dir}", flush=True)

folder_scores = defaultdict(list)

for root, dirs, files in os.walk(search_dir):
    for file in files:
        if os.path.splitext(file)[1].lower() in valid_extensions:
            img_path = os.path.join(root, file)
            try:
                # Load image, convert to grayscale (LA -> luminance + alpha, take luminance)
                # Matches the usage in niqe.py example
                img = Image.open(img_path).convert("LA")
                gray = np.array(img)[:, :, 0]
                
                score = niqe(gray)
                print(f"File: {file}, NIQE score: {score:.4f}", flush=True)
                folder_scores[root].append(score)
                count += 1
            except Exception as e:
                print(f"Error processing {file}: {e}", flush=True)

print(f"\nTotal images processed: {count}", flush=True)
print("\nStatistics per folder:", flush=True)
print("-" * 50, flush=True)

for folder, scores in folder_scores.items():
    if scores:
        avg_score = np.mean(scores)
        median_score = np.median(scores)
        print(f"Folder: {folder}", flush=True)
        print(f"  Images: {len(scores)}", flush=True)
        print(f"  Mean NIQE: {avg_score:.4f}", flush=True)
        print(f"  Median NIQE: {median_score:.4f}", flush=True)
        print("-" * 50, flush=True)