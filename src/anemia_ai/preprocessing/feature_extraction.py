"""
Handcrafted Feature Extraction for Baseline and Hugging Face JetX-GT Models.
"""

from typing import Dict, List
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

FEATURE_NAMES: List[str] = [
    "brightness_mean",
    "brightness_std",
    "brightness_p10",
    "brightness_p25",
    "brightness_p50",
    "brightness_p75",
    "brightness_p90",
    "redness_mean",
    "redness_std",
    "white_ratio",
    "pink_ratio",
    "r_mean",
    "r_std",
    "g_mean",
    "g_std",
    "b_mean",
    "b_std",
    "ratio_r_g",
    "ratio_r_b",
    "diff_r_b_norm",
    "hb_proxy_mean",
    "hb_proxy_std",
    "spatial_top_bottom_diff",
    "spatial_center_diff",
    "gradient_mean",
    "gradient_std",
    "local_var_mean",
    "local_var_std",
]


def extract_features(img: Image.Image) -> np.ndarray:
    """
    Extract 28 handcrafted color and texture features from a nail image.

    Args:
        img: PIL Image of the nail bed / region.

    Returns:
        np.ndarray of shape (28,), dtype float32.
    """
    img_rgb = img.convert("RGB").resize((224, 224))
    arr = np.array(img_rgb).astype(float)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    features: List[float] = []

    # 1. Brightness Features (7)
    features.append(float(brightness.mean()))
    features.append(float(brightness.std()))
    for p in [10, 25, 50, 75, 90]:
        features.append(float(np.percentile(brightness, p)))

    # 2. Redness Features (2)
    redness = r / (r + g + b + 1e-10)
    features.append(float(redness.mean()))
    features.append(float(redness.std()))

    # 3. Pallor Indices (2)
    white_ratio = float((brightness > 180).sum() / brightness.size)
    pink_ratio = float(((r > 150) & (g < 150) & (b < 150)).sum() / brightness.size)
    features.extend([white_ratio, pink_ratio])

    # 4. Per-channel Color Statistics (6)
    for ch in [r, g, b]:
        features.append(float(ch.mean()))
        features.append(float(ch.std()))

    # 5. Color Ratios & Chromatic Contrast (3)
    features.append(float((r.mean() + 1.0) / (g.mean() + 1.0)))
    features.append(float((r.mean() + 1.0) / (b.mean() + 1.0)))
    features.append(float((r.mean() - b.mean()) / 255.0))

    # 6. Hemoglobin Proxy Ratios (2)
    hb_proxy = r / (g + b + 1.0)
    features.append(float(hb_proxy.mean()))
    features.append(float(hb_proxy.std()))

    # 7. Spatial Gradient & Center Contrast (2)
    h, w = arr.shape[:2]
    top_region = brightness[: h // 3, :].mean()
    bottom_region = brightness[2 * h // 3 :, :].mean()
    features.append(float(top_region - bottom_region))

    center_region = brightness[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
    features.append(float(center_region - brightness.mean()))

    # 8. Edge and Gradient Distribution (2)
    gx = np.abs(np.diff(brightness, axis=1, prepend=brightness[:, :1]))
    gy = np.abs(np.diff(brightness, axis=0, prepend=brightness[:1, :]))
    gradient = np.sqrt(gx**2 + gy**2)
    features.append(float(gradient.mean()))
    features.append(float(gradient.std()))

    # 9. Local Texture Variance (2)
    local_mean = uniform_filter(brightness, size=7)
    local_var = uniform_filter((brightness - local_mean) ** 2, size=7)
    local_var = np.maximum(local_var, 0)
    local_std = np.sqrt(local_var)
    features.append(float(local_std.mean()))
    features.append(float(local_std.std()))

    vec = np.array(features, dtype=np.float32)
    if len(vec) != 28:
        raise ValueError(f"Expected exactly 28 features, but extracted {len(vec)}")
    return vec


def extract_jetx_27_features(img: Image.Image) -> np.ndarray:
    """
    Extract official 27 handcrafted color features from nail image for JetX-GT model.
    """
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(float)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    features = []

    # Brightness features (7)
    features.extend([brightness.mean(), brightness.std()])
    features.extend([np.percentile(brightness, p) for p in [10, 25, 50, 75, 90]])

    # Redness features (2)
    redness = r / (r + g + b + 1e-10)
    features.extend([redness.mean(), redness.std()])

    # Pallor features (2)
    white_ratio = (brightness > 180).sum() / brightness.size
    pink_ratio = ((r > 150) & (g < 150) & (b < 150)).sum() / brightness.size
    features.extend([white_ratio, pink_ratio])

    # Channel statistics (6)
    for ch in [r, g, b]:
        features.extend([ch.mean(), ch.std()])

    # Color ratios (3)
    features.extend([
        (r.mean() + 1) / (g.mean() + 1),
        (r.mean() + 1) / (b.mean() + 1),
        (r.mean() - b.mean()) / 255.0,
    ])

    # Hemoglobin proxy (2)
    hb = r / (g + b + 1.0)
    features.extend([hb.mean(), hb.std()])

    # Spatial features (2)
    h, w = arr.shape[:2]
    top = brightness[: h // 3, :].mean()
    bottom = brightness[2 * h // 3 :, :].mean()
    features.append(top - bottom)

    center = brightness[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
    features.append(center - brightness.mean())

    # Gradient features (2)
    gx = np.abs(np.diff(brightness, axis=1, prepend=brightness[:, :1]))
    gy = np.abs(np.diff(brightness, axis=0, prepend=brightness[:1, :]))
    gradient = np.sqrt(gx**2 + gy**2)
    features.extend([gradient.mean(), gradient.std()])

    # Local variance (2)
    local_mean = uniform_filter(brightness, size=7)
    local_var = uniform_filter((brightness - local_mean) ** 2, size=7)
    local_var = np.maximum(local_var, 0)
    features.extend([np.sqrt(local_var).mean(), np.sqrt(local_var).std()])

    return np.array(features, dtype=np.float32)


def extract_feature_dict(img: Image.Image) -> Dict[str, float]:
    """Extract features as a named dictionary for inspection and explanation."""
    feat_vec = extract_features(img)
    return {name: float(val) for name, val in zip(FEATURE_NAMES, feat_vec)}
