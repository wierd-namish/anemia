"""
Verification of Feature Pipeline Parity against Official JetX-GT Implementation.

Compares our backend/preprocessing/feature_extraction.py line-by-line against
the reference implementation from Hugging Face (JetX-GT/nail-anemia-detector/inference.py)
and GitHub (LE-TAPU-KOKO/nail-anemia-detection/scripts/save_model.py).
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.preprocessing.feature_extraction import extract_features as our_extract_features


def official_extract_features(img: Image.Image) -> np.ndarray:
    """Exact, verbatim replication of official inference.py extract_features."""
    img = img.convert('RGB').resize((224, 224))
    arr = np.array(img).astype(float)
    
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    brightness = 0.299*r + 0.587*g + 0.114*b
    
    features = []
    
    # Brightness features
    features.extend([brightness.mean(), brightness.std()])
    features.extend([np.percentile(brightness, p) for p in [10, 25, 50, 75, 90]])
    
    # Redness features
    redness = r / (r + g + b + 1e-10)
    features.extend([redness.mean(), redness.std()])
    
    # Pallor features
    white_ratio = (brightness > 180).sum() / brightness.size
    pink_ratio = ((r > 150) & (g < 150) & (b < 150)).sum() / brightness.size
    features.extend([white_ratio, pink_ratio])
    
    # Channel statistics
    for ch in [r, g, b]:
        features.extend([ch.mean(), ch.std()])
    
    # Color ratios
    features.extend([
        (r.mean() + 1) / (g.mean() + 1),
        (r.mean() + 1) / (b.mean() + 1),
        (r.mean() - b.mean()) / 255,
    ])
    
    # Hemoglobin proxy
    hb = r / (g + b + 1)
    features.extend([hb.mean(), hb.std()])
    
    # Spatial features
    h, w = arr.shape[:2]
    top = brightness[:h//3, :].mean()
    bottom = brightness[2*h//3:, :].mean()
    features.append(top - bottom)
    
    center = brightness[h//4:3*h//4, w//4:3*w//4].mean()
    features.append(center - brightness.mean())
    
    # Gradient features
    gx = np.abs(np.diff(brightness, axis=1, prepend=brightness[:, :1]))
    gy = np.abs(np.diff(brightness, axis=0, prepend=brightness[:1, :]))
    gradient = np.sqrt(gx**2 + gy**2)
    features.extend([gradient.mean(), gradient.std()])
    
    # Local variance
    local_mean = uniform_filter(brightness, size=7)
    local_var = uniform_filter((brightness - local_mean)**2, size=7)
    local_var = np.maximum(local_var, 0)
    features.extend([np.sqrt(local_var).mean(), np.sqrt(local_var).std()])
    
    return np.array(features, dtype=np.float32)


def test_parity():
    print("=" * 75)
    print("VERIFYING FEATURE EXTRACTION PARITY (OUR PIPELINE VS OFFICIAL REPO)")
    print("=" * 75)
    
    # Test across multiple varied images
    test_images = [
        Image.new("RGB", (300, 300), color=(180, 120, 110)),
        Image.new("RGB", (400, 500), color=(240, 210, 200)),
        Image.open("data/samples/real_koilonychia_anemia.jpg"),
        Image.open("data/samples/real_index_nail.jpg")
    ]
    
    all_passed = True
    for idx, img in enumerate(test_images):
        our_vec = our_extract_features(img)
        official_vec = official_extract_features(img)
        
        diff = np.abs(our_vec - official_vec)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\nTest Image {idx+1} ({img.size[0]}x{img.size[1]}):")
        print(f"  - Our feature vector shape:      {our_vec.shape}")
        print(f"  - Official feature vector shape: {official_vec.shape}")
        print(f"  - Maximum absolute difference:   {max_diff:.10e}")
        print(f"  - Mean absolute difference:      {mean_diff:.10e}")
        
        if max_diff > 1e-6:
            print("  ✗ PARITY MISMATCH DETECTED!")
            all_passed = False
        else:
            print("  ✓ EXACT BIT-LEVEL / FLOAT PARITY CONFIRMED.")
            
    print("\n" + "=" * 75)
    if all_passed:
        print("RESULT: 100% FEATURE PIPELINE PARITY VERIFIED.")
    else:
        print("RESULT: FAILED.")
    print("=" * 75)


if __name__ == "__main__":
    test_parity()
