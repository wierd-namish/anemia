"""
Standard PyTorch image preprocessing transformations.
"""

from typing import Tuple
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform(target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Returns standard inference transformation pipeline."""
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transform(target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Returns standard training transformation pipeline with data augmentations."""
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
