# Image Preprocessing & ROI Extraction Specification

**Document Identifier:** ETH-SUB-TECH-004  
**Version:** `nail_roi_224x224_rgb_v1.0.0`  
**Date:** 2026-09-30  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. Preprocessing Steps

```
[ Input Camera Frame ]
         ↓
[ Quality Gate Filter ]
  - Optical blur: cv2.Laplacian(gray, cv2.CV_64F).var() ≥ 5.0
  - Exposure: 40.0 ≤ mean(gray) ≤ 230.0
  - Specular glare: sum(gray ≥ 250) / total ≤ 0.20
  - Polish hue: non-skin hue saturation (hues 35-140, sat > 100) ≤ 0.15
         ↓
[ Nail ROI Segmentation ]
  - YCrCb conversion: Subungual vascular tissue isolated via Cr channel Otsu thresholding
  - Center-weighted contour ranking (minimum area: 2% of image)
  - Margin padding: 5% bounding box margin expansion
         ↓
[ Physiological Nail Verification ]
  - Skin chromaticity: (Cr ∈ [130, 180]) & (Cb ∈ [75, 135]) ≥ 35% of ROI
  - Subungual variance: std(Cr) ≥ 2.5, std(Cb) ≥ 2.0
         ↓
[ Normalization & Tensor Construction ]
  - Resize to 224x224 RGB via bilinear interpolation
  - Normalized: (pixel - mean) / std with ImageNet constants:
      mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
```
