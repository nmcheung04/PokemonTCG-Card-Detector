import os

# Paths
IMG_OUT = 'datasets/pokemon/images'

# Count images in each split
for split in ['train', 'val']:
    split_path = os.path.join(IMG_OUT, split)
    if os.path.exists(split_path):
        image_count = len([f for f in os.listdir(split_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"{split.capitalize()} set: {image_count} images")
    else:
        print(f"{split.capitalize()} set: directory not found")

# Total count
total_train = len([f for f in os.listdir(os.path.join(IMG_OUT, 'train')) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(os.path.join(IMG_OUT, 'train')) else 0
total_val = len([f for f in os.listdir(os.path.join(IMG_OUT, 'val')) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(os.path.join(IMG_OUT, 'val')) else 0

print(f"\nTotal dataset: {total_train + total_val} images")
print(f"Train/Val split: {total_train}/{total_val}") 