"""
Unit tests for NailDetector and ROI physiological gating.
"""

import unittest
from anemia_ai.preprocessing.nail_detection import NailDetector
from tests.fixtures.synthetic_samples import (
    generate_synthetic_nail,
    generate_wood_desk_ood,
)


class TestNailDetector(unittest.TestCase):
    """Verifies ROI detection, bounding box extraction, and OOD non-nail rejection."""

    def setUp(self):
        self.detector = NailDetector(target_size=(224, 224), padding_ratio=0.05)

    def test_detect_and_crop_valid_nail(self):
        img = generate_synthetic_nail(is_anemic=False, size=(300, 300))
        crop, bbox, meta = self.detector.detect_and_crop(img)
        self.assertEqual(crop.size, (224, 224))
        self.assertTrue(meta["is_valid_nail_roi"])
        self.assertEqual(len(bbox), 4)

    def test_detect_and_crop_with_guide_box(self):
        img = generate_synthetic_nail(is_anemic=True, size=(400, 400))
        guide = (0.2, 0.2, 0.8, 0.8)
        crop, bbox, meta = self.detector.detect_and_crop(img, guide_box=guide)
        self.assertEqual(crop.size, (224, 224))
        self.assertEqual(meta["method"], "visual_guide_box")

    def test_reject_wood_desk_ood(self):
        desk = generate_wood_desk_ood(size=(300, 300))
        crop, bbox, meta = self.detector.detect_and_crop(desk)
        self.assertFalse(meta["is_valid_nail_roi"])
        self.assertIn("skin_ratio", meta["roi_metrics"])


if __name__ == "__main__":
    unittest.main()
