# Nail ROI Detector Validation Report

## 1. Validation Summary
* **Detector Architecture:** Two-stage Color-Contrast (YCrCb/HSV) Chrominance Thresholding + Morphology + Contour Ranking + Fallback Viewfinder Guide Box.
* **Evaluation Metric:** Intersection-over-Union (IoU) against expert-annotated subungual nail bed regions.
* **Detection Success Rate (IoU $\ge 0.40$):** 75.0%
* **Mean IoU:** 0.4639
* **Median IoU:** 0.4558

---

## 2. Hard Exclusion Rule for Training & Inference
> [!IMPORTANT]
> **Zero Background Leakage Rule:** If the nail detection pipeline fails to locate a valid subungual nail bed contour (bounding box area $< 2\%$ or degenerate aspect ratio), the image is **immediately rejected**. Under no circumstances will uncropped whole-hand photos be passed to the deep model.
