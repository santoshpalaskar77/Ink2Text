import torch
import random
import numpy as np
import cv2

from torchvision.transforms import v2
from typing import List, Union
from PIL import Image
from collections import Counter

from ...globals import (
    IMG_CHANNELS,
    FIXED_IMG_SIZE,
    IMAGE_MEAN, IMAGE_STD,
    MAX_RESIZE_RATIO, MIN_RESIZE_RATIO
)
from .ocr_aug import ocr_augmentation_pipeline

# train_pipeline = default_augraphy_pipeline(scan_only=True)
train_pipeline = ocr_augmentation_pipeline()

general_transform_pipeline = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    v2.Grayscale(),

    v2.Resize(
        size=FIXED_IMG_SIZE - 1,
        interpolation=v2.InterpolationMode.BICUBIC,
        max_size=FIXED_IMG_SIZE,
        antialias=True
    ),

    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),
])


def resize_large_image(image, max_width=32767, max_height=32767):
    h, w = image.shape[:2]
    if w > max_width or h > max_height:
        print(f'Skipping image with height: {h}, width: {w} (exceeds maximum size).')
        return None  # Skip this image
    return image



def trim_white_border(image: np.ndarray):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image is not in RGB format or channel is not in third dimension")

    if image.dtype != np.uint8:
        raise ValueError(f"Image should be stored in uint8")

    corners = [tuple(image[0, 0]), tuple(image[0, -1]),
               tuple(image[-1, 0]), tuple(image[-1, -1])]
    bg_color = Counter(corners).most_common(1)[0][0]
    bg_color_np = np.array(bg_color, dtype=np.uint8)
    
    h, w = image.shape[:2]
    bg = np.full((h, w, 3), bg_color_np, dtype=np.uint8)

    diff = cv2.absdiff(image, bg)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    threshold = 15
    _, diff = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    x, y, w, h = cv2.boundingRect(diff) 

    trimmed_image = image[y:y+h, x:x+w]

    return trimmed_image


def add_white_border(image: np.ndarray, max_size: int) -> np.ndarray:
    randi = [random.randint(0, max_size) for _ in range(4)]
    pad_height_size = randi[1] + randi[3]
    pad_width_size  = randi[0] + randi[2]
    if (pad_height_size + image.shape[0] < 30):
        compensate_height = int((30 - (pad_height_size + image.shape[0])) * 0.5) + 1
        randi[1] += compensate_height
        randi[3] += compensate_height
    if (pad_width_size + image.shape[1] < 30):
        compensate_width = int((30 - (pad_width_size + image.shape[1])) * 0.5) + 1
        randi[0] += compensate_width
        randi[2] += compensate_width
    return v2.functional.pad(
        torch.from_numpy(image).permute(2, 0, 1),
        padding=randi,
        padding_mode='constant',
        fill=(255, 255, 255)
    )


def padding(images: List[torch.Tensor], required_size: int) -> List[torch.Tensor]:
    images = [
        v2.functional.pad(
            img,
            padding=[0, 0, required_size - img.shape[2], required_size - img.shape[1]]
        )
        for img in images
    ]
    return images


def random_resize(
    images: List[np.ndarray], 
    minr: float, 
    maxr: float
) -> List[np.ndarray]:
    if len(images[0].shape) != 3 or images[0].shape[2] != 3:
        raise ValueError("Image is not in RGB format or channel is not in third dimension")

    ratios = [random.uniform(minr, maxr) for _ in range(len(images))]
    return [
        cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LANCZOS4)
        for img, r in zip(images, ratios)
    ]


import cv2
import random

def rotate(image, angle_min, angle_max):
    height, width = image.shape[:2]
    
    # Ensure image dimensions are within acceptable limits
    max_size = 32767  # Close to SHRT_MAX for OpenCV
    if height >= max_size or width >= max_size:
        print(f"Image size too large for rotation: {height}x{width}")
        return None  # Set image to None if too large
    else:
        # Rotate the image
        angle = random.uniform(angle_min, angle_max)
        center = (width // 2, height // 2)
        rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # Compute new bounding dimensions
        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)

        # Adjust rotation matrix to account for translation
        rotation_mat[0, 2] += new_width // 2 - center[0]
        rotation_mat[1, 2] += new_height // 2 - center[1]

        # Perform the rotation
        rotated_image = cv2.warpAffine(image, rotation_mat, (new_width, new_height), borderValue=(255, 255, 255))
        return rotated_image



def ocr_aug(image: np.ndarray) -> np.ndarray:
    # Define the maximum allowed dimension size for OpenCV
    max_dim = 32767

    # Apply random rotation with 20% probability
    if random.random() < 0.2:
        image = rotate(image, -5, 5)

    # Add white border and permute for further processing
    image = add_white_border(image, max_size=25).permute(1, 2, 0).numpy()
    # Resize the image if it exceeds the maximum dimensions
    if image.shape[0] > max_dim or image.shape[1] > max_dim:
        image = None
    else:
        # Apply the train pipeline to the processed image
        image = train_pipeline(image)

    return image



def train_transform(images: List[Image.Image]) -> List[torch.Tensor]:
    max_height = 32767  # Set your maximum height
    max_width = 32767   # Set your maximum width
    MIN_HEIGHT = 12
    MIN_WIDTH = 30
    MAX_DIM = 32767  # Max allowable image dimension for resizing
    
    # Convert images to RGB and skip if they are grayscale and exceed max dimensions
    images = [np.array(img.convert('RGB')) if isinstance(img, Image.Image) else img for img in images]
    for i, img in enumerate(images):
        if img.shape[0] == 0 or img.shape[1] == 0:
            images[i] = None

    valid_images = []
    for img in images:
        if img is not None:
            # Check for zero dimensions
            if img.shape[0] == 0 or img.shape[1] == 0:
                print("Skipping image due to zero dimensions.")
                img = None
            
            # Skip if image dimensions are below minimum
            if img.shape[0] < MIN_HEIGHT or img.shape[1] < MIN_WIDTH:
                print(img.shape)
                print("Skipping image due to insufficient 33 dimensions.")
                img = None
            
            # Resize images that are too large
            if img.shape[0] > MAX_DIM or img.shape[1] > MAX_DIM:
                print(f"Resizing large image: original size ({img.shape[0]}, {img.shape[1]})")
                img = Image.fromarray(img).resize((MAX_DIM, MAX_DIM))
            
            # Add to valid images list
            valid_images.append(np.array(img))

    images = valid_images
    valid_images = []  # Initialize an empty list to store valid images

    for img in images:
        if img is not None:  # Check if the image is not None
            valid_images.append(np.array(img))  # Convert each image to a NumPy array and append it

    images = valid_images

    # Resize large images to max 32767x32767 or skip if too large
    images = [resize_large_image(img) for img in images]
    images = [img for img in images if img is not None]  # Skip None images

    images = random_resize(images, MIN_RESIZE_RATIO, MAX_RESIZE_RATIO)
    images = [trim_white_border(image) for image in images]

    valid_images = images
    for img1 in valid_images:
        if img1.shape[0] < MIN_HEIGHT or img1.shape[1] < MIN_WIDTH:
                print("Skipping image due to insufficient 24 dimensions.")
                img1 = None  # Skip this image
    images = valid_images
    valid_images = []
    for img in images:
        if img is not None:  # Check if the image is not None
            valid_images.append(np.array(img))
    # OCR augmentation
    images = valid_images
    images = [ocr_aug(image) for image in images]

    # general transform pipeline
    images = [general_transform_pipeline(image) for image in images]
    
    # padding to fixed size
    images = padding(images, FIXED_IMG_SIZE)
    return images



def inference_transform(images: List[Union[np.ndarray, Image.Image]]) -> List[torch.Tensor]:
    assert IMG_CHANNELS == 1, "Only support grayscale images for now"

    MIN_HEIGHT = 12
    MIN_WIDTH = 30
    MAX_DIM = 32767  # Max allowable image dimension for resizing
    
    # Convert any PIL Image objects to numpy arrays (RGB)
    images = [np.array(img.convert('RGB')) if isinstance(img, Image.Image) else img for img in images]
    for i, img in enumerate(images):
        if img.shape[0] == 0 or img.shape[1] == 0:
            images[i] = None

    # Filter and process images
    valid_images = []
    for img in images:
        if img is not None:
            # Check for zero dimensions
            if img.shape[0] == 0 or img.shape[1] == 0:
                print("Skipping image due to zero dimensions.")
                continue  # Skip this image
            
            # Skip if image dimensions are below minimum
            if img.shape[0] < MIN_HEIGHT or img.shape[1] < MIN_WIDTH:
                print(img.shape)
                print("Skipping image due to insufficient 33 dimensions.")
                img = None
            
            # Resize images that are too large
            if img.shape[0] > MAX_DIM or img.shape[1] > MAX_DIM:
                print(f"Resizing large image: original size ({img.shape[0]}, {img.shape[1]})")
                img = Image.fromarray(img).resize((MAX_DIM, MAX_DIM))
            
            # Add to valid images list
            valid_images.append(np.array(img))

    images = valid_images
    valid_images = []  # Initialize an empty list to store valid images

    for img in images:
        if img is not None:  # Check if the image is not None
            valid_images.append(np.array(img))  # Convert each image to a NumPy array and append it

    # Proceed with valid images
    valid_images = [trim_white_border(image) for image in valid_images]

    for img1 in valid_images:
        if img1.shape[0] < MIN_HEIGHT or img1.shape[1] < MIN_WIDTH:
                print("Skipping image due to insufficient 24 dimensions.")
                continue  # Skip this image
    valid_images = []
    for img in images:
        if img is not None:  # Check if the image is not None
            valid_images.append(np.array(img))
    # Apply general transform pipeline
    valid_images = [general_transform_pipeline(image) for image in valid_images]
    
    # Apply padding to fixed size
    valid_images = padding(valid_images, FIXED_IMG_SIZE)

    return valid_images

