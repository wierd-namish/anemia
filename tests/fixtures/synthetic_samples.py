"""
Synthetic test fixtures for unit and integration testing.
Purely synthetic images used strictly for software verification.
"""

from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw


def generate_synthetic_nail(is_anemic: bool = False, size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Generates synthetic nail photograph with skin surround, nail bed, and lunula."""
    bg_color = (200, 175, 155) if is_anemic else (195, 145, 125)
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Nail bed
    bed_color = (235, 215, 210) if is_anemic else (215, 130, 135)
    draw.ellipse([size[0] * 0.23, size[1] * 0.17, size[0] * 0.77, size[1] * 0.83], fill=bed_color)

    # Lunula
    lunula_color = (245, 235, 230) if is_anemic else (240, 220, 220)
    draw.chord(
        [size[0] * 0.33, size[1] * 0.63, size[0] * 0.67, size[1] * 0.83],
        start=180,
        end=360,
        fill=lunula_color,
    )

    arr = np.array(img).astype(float)
    noise = np.random.normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_synthetic_anemia_nail(size: Tuple[int, int] = (224, 224), seed: int = 42) -> Image.Image:
    """Generates synthetic nail with pale/anemic nail bed tones."""
    np.random.seed(seed)
    return generate_synthetic_nail(is_anemic=True, size=size)


def generate_synthetic_healthy_nail(size: Tuple[int, int] = (224, 224), seed: int = 42) -> Image.Image:
    """Generates synthetic nail with vascularized/healthy pink nail bed tones."""
    np.random.seed(seed)
    return generate_synthetic_nail(is_anemic=False, size=size)


def generate_blurry_image(size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Generates featureless blurry image with near-zero Laplacian variance."""
    return Image.new("RGB", size, color=(180, 180, 180))


def generate_underexposed_image(size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Generates textured underexposed image with mean luminance < 40 and high sharpness."""
    np.random.seed(42)
    arr = np.random.normal(25, 8, (size[1], size[0], 3))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_overexposed_image(size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Generates textured overexposed image with mean luminance > 230 and high sharpness."""
    np.random.seed(42)
    arr = np.random.normal(240, 5, (size[1], size[0], 3))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_polished_nail(size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Generates nail image with bright blue nail polish and texture."""
    img = Image.new("RGB", size, color=(195, 145, 125))
    draw = ImageDraw.Draw(img)
    draw.ellipse([size[0] * 0.23, size[1] * 0.17, size[0] * 0.77, size[1] * 0.83], fill=(0, 50, 220))
    arr = np.array(img).astype(float)
    noise = np.random.normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_wood_desk_ood(size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Generates non-skin out-of-distribution wooden texture (uniform wood tone)."""
    return Image.new("RGB", size, color=(110, 70, 40))


def generate_fabric_ood(size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Generates non-skin out-of-distribution blue clothing textile."""
    return Image.new("RGB", size, color=(40, 60, 140))
