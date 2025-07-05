import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import datasets, transforms
from tqdm import tqdm
import cv2

# Paths
input_dir = "organized_dataset/test"
output_dir = "augmented_dataset"

# --- Stronger custom augmentations ---
def add_strong_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.5)))

def add_strong_noise(img):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 30, arr.shape)  # More intense
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def adjust_strong_contrast(img):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.4, 1.6))

def adjust_strong_brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.4, 1.5))

def adjust_strong_sharpness(img):
    return ImageEnhance.Sharpness(img).enhance(random.uniform(0.2, 3.0))

def elastic_transform(img, alpha=20, sigma=5):
    # Apply elastic deformation like in MNIST paper
    image = np.array(img)
    shape = image.shape

    dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(distorted)

def erode_dilate(img):
    kernel = np.ones((2, 2), np.uint8)
    arr = np.array(img)
    if random.random() > 0.5:
        arr = cv2.erode(arr, kernel, iterations=1)
    else:
        arr = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(arr)

# Strong transform pool
strong_transforms = [
    add_strong_blur,
    add_strong_noise,
    adjust_strong_contrast,
    adjust_strong_brightness,
    adjust_strong_sharpness,
    elastic_transform,
    erode_dilate,
]

# Preprocess
resize_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
])

# Load dataset
dataset = datasets.ImageFolder(root=input_dir, transform=resize_transform)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

from collections import defaultdict
image_counts = defaultdict(int)
for img_path, label in dataset.samples:
    class_name = os.path.relpath(os.path.dirname(img_path), input_dir)
    image_counts[class_name] += 1

# Save originals + augmentations
for i, (img, label) in enumerate(tqdm(dataset, desc="Processing")):
    class_name = idx_to_class[label]
    save_dir = os.path.join(output_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save original
    img.save(os.path.join(save_dir, f"{image_counts[class_name]:04}.png"))
    image_counts[class_name] += 1

    # Save stronger augmentation
    transform = random.choice(strong_transforms)
    aug_img = transform(img)
    aug_img.save(os.path.join(save_dir, f"{image_counts[class_name]:04}.png"))
    image_counts[class_name] += 1

print(f"\n✅ Strongly augmented dataset saved in: {os.path.abspath(output_dir)}")
