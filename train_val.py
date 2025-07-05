import os
import shutil
import random
from tqdm import tqdm

# Input and output folders
input_dir = "augmented_dataset"
output_base = "final_dataset"
train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")

# Create output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# For each class
for class_name in tqdm(os.listdir(input_dir), desc="Splitting dataset"):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    
    if len(images) < 50:
        print(f"⚠️ Skipping {class_name}: only {len(images)} images found.")
        continue

    random.shuffle(images)
    train_images = images[:40]
    val_images = images[40:50]  # Next 10

    # Make sure class subfolders exist in output dirs
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Copy train images
    for img_name in train_images:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(train_dir, class_name, img_name)
        shutil.copy2(src, dst)

    # Copy val images
    for img_name in val_images:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(val_dir, class_name, img_name)
        shutil.copy2(src, dst)

print(f"\n✅ Train/Val split completed:")
print(f"   → Train folder: {os.path.abspath(train_dir)}")
print(f"   → Val folder:   {os.path.abspath(val_dir)}")
