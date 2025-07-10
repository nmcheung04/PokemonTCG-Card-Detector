import os
import json
import shutil
import random
from PIL import Image

# Paths
LABELME_DIR = 'Labeled Images'
IMAGES_DIR = 'Raw Images'
OUT_ROOT = 'datasets/pokemon'
IMG_OUT = os.path.join(OUT_ROOT, 'images')
LBL_OUT = os.path.join(OUT_ROOT, 'labels')

# Create output directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(IMG_OUT, split), exist_ok=True)
    os.makedirs(os.path.join(LBL_OUT, split), exist_ok=True)

# Gather all JSON files
json_files = [f for f in os.listdir(LABELME_DIR) if f.endswith('.json')]
random.shuffle(json_files)

# Train/val split
split_idx = int(0.8 * len(json_files))
train_files = json_files[:split_idx]
val_files = json_files[split_idx:]
splits = {'train': train_files, 'val': val_files}

# Helper: polygon to bbox
def polygon_to_bbox(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, y_min, x_max, y_max

# Process each split
for split, files in splits.items():
    for json_file in files:
        json_path = os.path.join(LABELME_DIR, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        image_filename = data['imagePath']
        image_path = os.path.join(IMAGES_DIR, image_filename)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_filename} not found, skipping {json_file}")
            continue
            
        # Output image path
        out_img_path = os.path.join(IMG_OUT, split, image_filename)
        # Output label path
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        out_lbl_path = os.path.join(LBL_OUT, split, label_filename)
        # Copy image
        shutil.copy(image_path, out_img_path)
        # Get image size
        with Image.open(image_path) as img:
            w, h = img.size
        # Write YOLO label
        with open(out_lbl_path, 'w') as out_lbl:
            for shape in data['shapes']:
                if shape['label'] != 'card':
                    continue
                points = shape['points']
                x_min, y_min, x_max, y_max = polygon_to_bbox(points)
                # Convert to YOLO format
                x_center = (x_min + x_max) / 2.0 / w
                y_center = (y_min + y_max) / 2.0 / h
                bbox_w = (x_max - x_min) / w
                bbox_h = (y_max - y_min) / h
                # Only one class: 0
                out_lbl.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

print('Conversion complete! YOLOv8 dataset is ready in datasets/pokemon/') 