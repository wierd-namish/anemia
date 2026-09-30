"""
Image encoding, decoding, and conversion helpers.
"""

import base64
from io import BytesIO
from typing import Optional, Union
import numpy as np
from PIL import Image


def image_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    """Encodes a PIL Image into a data URI base64 JPEG string."""
    buffer = BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def bytes_to_pil_image(image_bytes: bytes) -> Image.Image:
    """Decodes raw bytes into a PIL RGB Image."""
    img = Image.open(BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
