"""
Nail Region-of-Interest (ROI) Detection and Extraction Pipeline.

Isolates the subungual nail bed and plate from full-finger or hand photographs:
- Contour / color-contrast nail bed segmentation
- Bounding box extraction with margin expansion
- Visual guide-box crop for live UI alignment
- Physiological nail bed validation (rejects non-nail OOD inputs like wood desks, clothing, uniform skin)
- Optional debug visualization export (original, detected_nail, final_crop)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw

from anemia_ai.core.interfaces import BaseDetector


class NailDetector(BaseDetector):
    """Nail ROI detector and cropping engine."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        padding_ratio: float = 0.05,
    ):
        self.target_size = target_size
        self.padding_ratio = padding_ratio

    def validate_nail_roi(self, img_rgb: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates whether the detected or cropped ROI contains physiological nail/skin tissue
        with anatomical subungual contrast, rejecting flat backgrounds, wood desks, textiles, and uniform skin.
        """
        ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        # 1. Skin/Nail physiological chromaticity mask (YCrCb)
        skin_mask = (cr >= 130) & (cr <= 180) & (cb >= 75) & (cb <= 135)
        skin_ratio = float(np.mean(skin_mask))

        if skin_ratio < 0.35:
            return (
                False,
                "Nail was not detected. Surface does not contain skin or nail tissue.",
                {
                    "skin_ratio": round(skin_ratio, 4),
                    "rejection_reason": "non_skin_chromaticity",
                },
            )

        # 2. Subungual chromatic variation & structure
        cr_std = float(np.std(cr))
        cb_std = float(np.std(cb))

        if cr_std < 2.5 and cb_std < 2.0:
            return (
                False,
                "Nail was not detected. Image appears to be uniform background or flat skin without visible fingernail margins.",
                {
                    "skin_ratio": round(skin_ratio, 4),
                    "cr_std": round(cr_std, 2),
                    "cb_std": round(cb_std, 2),
                    "rejection_reason": "lacks_nail_bed_contrast",
                },
            )

        return (
            True,
            "Valid nail ROI confirmed.",
            {
                "skin_ratio": round(skin_ratio, 4),
                "cr_std": round(cr_std, 2),
                "cb_std": round(cb_std, 2),
            },
        )

    def detect_and_crop(
        self,
        img: Image.Image,
        guide_box: Optional[Tuple[float, float, float, float]] = None,
        save_debug_dir: Optional[Path] = None,
        debug_prefix: str = "sample",
    ) -> Tuple[Image.Image, Tuple[int, int, int, int], Dict[str, Any]]:
        """
        Detects nail region and returns cropped, normalized PIL Image.

        Args:
            img: Input PIL Image.
            guide_box: Optional normalized coordinates (ymin, xmin, ymax, xmax).
            save_debug_dir: Optional directory to save debug images.
            debug_prefix: Prefix for saved debug files.

        Returns:
            (cropped_img: Image.Image, bbox: (x1, y1, x2, y2), metadata: Dict)
        """
        w_orig, h_orig = img.size
        img_np = np.array(img.convert("RGB"))

        # 1. If visual guide box provided
        if guide_box is not None:
            ymin, xmin, ymax, xmax = guide_box
            x1 = int(max(0, xmin * w_orig))
            y1 = int(max(0, ymin * h_orig))
            x2 = int(min(w_orig, xmax * w_orig))
            y2 = int(min(h_orig, ymax * h_orig))
            method = "visual_guide_box"
        else:
            # 2. Automated color-contrast & contour ROI localization
            bbox, method = self._locate_nail_contour(img_np)
            x1, y1, x2, y2 = bbox

        # Ensure valid non-degenerate bounding box
        if x2 <= x1 + 20 or y2 <= y1 + 20:
            x1 = int(w_orig * 0.25)
            y1 = int(h_orig * 0.25)
            x2 = int(w_orig * 0.75)
            y2 = int(h_orig * 0.75)
            method = "center_fallback"

        # Add slight margin padding
        pad_w = int((x2 - x1) * self.padding_ratio)
        pad_h = int((y2 - y1) * self.padding_ratio)
        crop_x1 = max(0, x1 - pad_w)
        crop_y1 = max(0, y1 - pad_h)
        crop_x2 = min(w_orig, x2 + pad_w)
        crop_y2 = min(h_orig, y2 + pad_h)

        # Crop and resize to target model dimension (224x224)
        cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        final_crop = cropped.convert("RGB").resize(self.target_size)

        # Validate cropped ROI
        crop_np = np.array(final_crop)
        is_valid_roi, roi_msg, roi_metrics = self.validate_nail_roi(crop_np)

        metadata = {
            "method": method,
            "original_size": (w_orig, h_orig),
            "bbox": (crop_x1, crop_y1, crop_x2, crop_y2),
            "crop_dimensions": (crop_x2 - crop_x1, crop_y2 - crop_y1),
            "target_size": self.target_size,
            "is_valid_nail_roi": is_valid_roi,
            "roi_validation_message": roi_msg,
            "roi_metrics": roi_metrics,
        }

        # Save debug images if requested
        if save_debug_dir is not None:
            save_debug_dir = Path(save_debug_dir)
            save_debug_dir.mkdir(parents=True, exist_ok=True)

            img.save(save_debug_dir / f"{debug_prefix}_original.jpg")

            img_annotated = img.copy()
            draw = ImageDraw.Draw(img_annotated)
            outline_color = (0, 255, 0) if is_valid_roi else (255, 0, 0)
            draw.rectangle([crop_x1, crop_y1, crop_x2, crop_y2], outline=outline_color, width=4)
            img_annotated.save(save_debug_dir / f"{debug_prefix}_detected_nail.jpg")

            final_crop.save(save_debug_dir / f"{debug_prefix}_final_crop.jpg")

        return final_crop, (crop_x1, crop_y1, crop_x2, crop_y2), metadata

    def _locate_nail_contour(self, img_rgb: np.ndarray) -> Tuple[Tuple[int, int, int, int], str]:
        """Locates nail bed region using chrominance thresholding and morphology."""
        h, w, _ = img_rgb.shape

        ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb[:, :, 1]

        # Focus search on central 80%
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        center_mask = np.zeros((h, w), dtype=np.uint8)
        center_mask[margin_y : h - margin_y, margin_x : w - margin_x] = 255

        # Thresholding for reddish/pink subungual tissue
        _, thresh = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_and(thresh, center_mask)

        # Morphological opening/closing to consolidate nail bed region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            valid_contours = []
            img_center = np.array([w / 2.0, h / 2.0])

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > (w * h * 0.02):
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    cnt_center = np.array([x + cw / 2.0, y + ch / 2.0])
                    dist_to_center = np.linalg.norm(cnt_center - img_center)
                    valid_contours.append((area, dist_to_center, (x, y, x + cw, y + ch)))

            if valid_contours:
                valid_contours.sort(key=lambda item: (-item[0], item[1]))
                best_bbox = valid_contours[0][2]
                return best_bbox, "contour_nail_detector"

        return (int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)), "center_fallback"
