"""
Medical-AI Image Quality Assessment Module.

Enforces clinical data-quality criteria before diagnostic analysis:
1. Spatial resolution check (minimum 128x128 pixels)
2. Optical blur / defocus check (Laplacian variance on grayscale nail ROI)
3. Luminance / illumination exposure check (underexposure <40, overexposure >230)
4. Specular glare / flash reflection check (blown-out highlights >250 intensity)
5. Nail polish / artificial pigment coverage (non-physiological color saturation)
"""

from typing import Any, Dict, Tuple
import cv2
import numpy as np
from PIL import Image

from anemia_ai.config.constants import (
    BLUR_LAPLACIAN_VAR_THRESHOLD,
    GLARE_PIXEL_RATIO_MAX,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    NON_PHYSIOLOGICAL_SATURATION_MAX,
    OVEREXPOSURE_THRESHOLD,
    UNDEREXPOSURE_THRESHOLD,
)
from anemia_ai.core.interfaces import BaseQualityChecker


def assess_image_quality(img: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluates whether an input image meets minimum clinical quality standards.

    Args:
        img: Input PIL Image (full frame or cropped nail ROI).

    Returns:
        (passed: bool, message: str, metrics: Dict[str, Any])
    """
    w, h = img.size

    # 1. Spatial Resolution Check
    if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
        return (
            False,
            "Image unsuitable for assessment. Image resolution is too low.",
            {
                "width": w,
                "height": h,
                "error_type": "low_resolution",
                "threshold": f"Minimum {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}",
            },
        )

    img_rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # 2. Optical Blur / Defocus Check
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < BLUR_LAPLACIAN_VAR_THRESHOLD:
        return (
            False,
            "Image unsuitable for assessment. Image is blurry. Please hold steady and tap to focus.",
            {
                "laplacian_variance": round(laplacian_var, 2),
                "threshold": BLUR_LAPLACIAN_VAR_THRESHOLD,
                "error_type": "severe_blur",
            },
        )

    # 3. Luminance / Illumination Exposure Check
    mean_luminance = float(np.mean(gray))
    if mean_luminance < UNDEREXPOSURE_THRESHOLD:
        return (
            False,
            "Image unsuitable for assessment. Lighting is too dark. Please capture in brighter lighting.",
            {
                "mean_luminance": round(mean_luminance, 2),
                "threshold": UNDEREXPOSURE_THRESHOLD,
                "error_type": "underexposed",
            },
        )
    if mean_luminance > OVEREXPOSURE_THRESHOLD:
        return (
            False,
            "Image unsuitable for assessment. Image is washed out / overexposed.",
            {
                "mean_luminance": round(mean_luminance, 2),
                "threshold": OVEREXPOSURE_THRESHOLD,
                "error_type": "overexposed",
            },
        )

    # 4. Specular Glare / Flash Reflection Check
    glare_mask = gray >= 250
    glare_ratio = float(np.sum(glare_mask) / gray.size)
    if glare_ratio > GLARE_PIXEL_RATIO_MAX:
        return (
            False,
            "Image unsuitable for assessment. Excessive glare or light reflection on nail surface.",
            {
                "glare_ratio": round(glare_ratio, 4),
                "max_ratio": GLARE_PIXEL_RATIO_MAX,
                "error_type": "excessive_glare",
            },
        )

    # 5. Nail Polish / Non-physiological Pigment Check
    hues = hsv[:, :, 0]
    sats = hsv[:, :, 1]
    unnatural_color_mask = ((hues > 35) & (hues < 140)) & (sats > 100)
    unnatural_ratio = float(np.sum(unnatural_color_mask) / hues.size)
    if unnatural_ratio > NON_PHYSIOLOGICAL_SATURATION_MAX:
        return (
            False,
            "Image unsuitable for assessment. Colored nail polish or artificial pigment detected.",
            {
                "unnatural_color_ratio": round(unnatural_ratio, 4),
                "max_allowed": NON_PHYSIOLOGICAL_SATURATION_MAX,
                "error_type": "nail_polish_detected",
            },
        )

    metrics = {
        "width": w,
        "height": h,
        "laplacian_variance": round(laplacian_var, 2),
        "mean_luminance": round(mean_luminance, 2),
        "glare_ratio": round(glare_ratio, 4),
        "unnatural_color_ratio": round(unnatural_ratio, 4),
    }
    return True, "Image quality acceptable for clinical screening.", metrics


class ImageQualityChecker(BaseQualityChecker):
    """Object-oriented quality assurance checker."""

    def assess(self, image: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
        return assess_image_quality(image)
