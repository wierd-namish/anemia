"""
Automated Tests for Image Quality Gate Module.
Tests:
- Defocus / blur rejection (Laplacian variance < 5.0)
- Underexposure rejection (Luminance < 40.0)
- Overexposure rejection (Luminance > 230.0)
- Non-physiological nail polish rejection
"""

import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from anemia_ai.preprocessing.image_quality import assess_image_quality


class TestQualityGate(unittest.TestCase):

    def test_blurry_image_rejection(self):
        """Verify completely flat/blurred image fails blur gate."""
        img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        passed, msg, metrics = assess_image_quality(img)
        self.assertFalse(passed)
        self.assertEqual(metrics["error_type"], "severe_blur")
        self.assertIn("blurry", msg.lower())

    def test_underexposed_image_rejection(self):
        """Verify very dark image fails underexposure gate."""
        noise = np.random.randint(0, 15, (300, 300, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        passed, msg, metrics = assess_image_quality(img)
        self.assertFalse(passed)
        self.assertIn(metrics["error_type"], ["underexposed", "severe_blur"])

    def test_overexposed_image_rejection(self):
        """Verify blown-out/washed-out image fails overexposure gate."""
        noise = np.random.randint(240, 256, (300, 300, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        passed, msg, metrics = assess_image_quality(img)
        self.assertFalse(passed)
        self.assertIn(metrics["error_type"], ["overexposed", "excessive_glare", "severe_blur"])

    def test_nail_polish_rejection(self):
        """Verify blue/cyan/bright green artificial nail polish is rejected."""
        arr = np.zeros((300, 300, 3), dtype=np.uint8)
        arr[:, :, 2] = 220 # High Blue
        arr[:, :, 0] = 20
        arr[:, :, 1] = 20
        arr = np.clip(arr.astype(np.int16) + np.random.randint(0, 30, (300, 300, 3)), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        passed, msg, metrics = assess_image_quality(img)
        self.assertFalse(passed)
        self.assertEqual(metrics["error_type"], "nail_polish_detected")
        self.assertTrue("polish" in msg.lower() or "pigment" in msg.lower())


if __name__ == "__main__":
    unittest.main()
