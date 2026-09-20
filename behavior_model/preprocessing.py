import os
from pathlib import Path
from typing import Union, Tuple
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

from behavior_model.config import (
    IMAGE_SIZE,
    CROP_SIZE,
    RESIZE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Returns training transformations with data augmentations to prevent overfitting:
    Random resized crop, horizontal flips, color jitter, and rotation.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Returns standard validation / test transformations:
    Deterministic resize, center crop, and ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Reusable inference transformation pipeline matching the validation pipeline.
    """
    return get_val_transforms(image_size=image_size)


def load_and_preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    transform: transforms.Compose = None,
) -> torch.Tensor:
    """
    Loads an image from file path, PIL Image, or NumPy array, converts to RGB,
    applies ResNet18 normalization, and adds batch dimension (1, 3, 224, 224).
    """
    if transform is None:
        transform = get_inference_transforms()

    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path.resolve()}")
        if not img_path.is_file():
            raise ValueError(f"Path is not a valid file: {img_path.resolve()}")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not open image file '{img_path}': {e}")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Apply transform: shape (3, H, W)
    tensor = transform(image)
    # Add batch dimension: shape (1, 3, H, W)
    tensor = tensor.unsqueeze(0)
    return tensor
