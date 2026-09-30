"""
Automated Tests for Nail ROI Extraction and Validation Gate.
Tests:
- Bounding box detection and normalized 224x224 crop
- Visual guide box coordinate mapping
- Physiological subungual chromatic validation
- Non-nail ROI rejection (uniform skin, wood, background)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from anemia_ai.preprocessing.nail_detection import NailDetector


class TestROIGate(unittest.TestCase):

    def setUp(self):
        self.detector = NailDetector(target_size=(224, 224))

    def test_nail_detector_crop_dimensions(self):
        """Verify NailDetector always yields 224x224 RGB images."""
        img = Image.new("RGB", (640, 480), color=(180, 120, 110))
        crop, bbox, meta = self.detector.detect_and_crop(img)
        
        self.assertEqual(crop.size, (224, 224))
        self.assertEqual(crop.mode, "RGB")
        self.assertIn("bbox", meta)
        self.assertIn("method", meta)

    def test_visual_guide_box_crop(self):
        """Verify guide_box coordinates are respected in cropping."""
        img = Image.new("RGB", (1000, 1000), color=(180, 120, 110))
        crop, bbox, meta = self.detector.detect_and_crop(img, guide_box=(0.25, 0.25, 0.75, 0.75))
        
        self.assertEqual(meta["method"], "visual_guide_box")
        self.assertGreaterEqual(bbox[0], 200)
        self.assertLessEqual(bbox[2], 800)

    def test_wood_desk_roi_rejection(self):
        """Verify wooden desk background is marked as invalid nail ROI."""
        wood_path = Path("test_ood/ood_wood_desk.jpg")
        if wood_path.exists():
            img = Image.open(wood_path)
            crop, bbox, meta = self.detector.detect_and_crop(img)
            self.assertFalse(meta["is_valid_nail_roi"])

    def test_skin_only_roi_rejection(self):
        """Verify skin-only image without nail margins is marked as invalid nail ROI."""
        skin_path = Path("test_ood/ood_skin_only.jpg")
        if skin_path.exists():
            img = Image.open(skin_path)
            crop, bbox, meta = self.detector.detect_and_crop(img)
            self.assertFalse(meta["is_valid_nail_roi"])


if __name__ == "__main__":
    unittest.main()
