"""
Automated Tests for Out-Of-Distribution (OOD) and Historical JetX Failure Regression.
Tests:
- Wood desk background (Historical JetX gave 99.99% anemia -> MUST be INCONCLUSIVE)
- Skin-only photograph (no nail plate) -> MUST be INCONCLUSIVE
- Clothing / textile fabric -> MUST be INCONCLUSIVE
- Random ceramic / mug / desk objects -> MUST be INCONCLUSIVE
- Blurry, underexposed, and overexposed frames -> MUST be INCONCLUSIVE
- Verify NO OOD sample produces ANEMIA or NO_ANEMIA
"""

import unittest
from pathlib import Path
from PIL import Image

from backend.model.inference_pipeline import DiagnosticInferenceService
from backend.config import STATE_INCONCLUSIVE, STATE_ANEMIA, STATE_NO_ANEMIA


class TestOODRejection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = DiagnosticInferenceService()

    def test_historical_jetx_wood_desk_regression(self):
        """
        CRITICAL REGRESSION TEST:
        Historical JetX-GT baseline assigned 99.99% anemia to wooden background.
        Our clinical pipeline MUST intercept and reject it as INCONCLUSIVE.
        """
        wood_paths = [
            Path("tests/fixtures/ood/ood_wood_desk.jpg"),
            Path("test_ood/ood_wood_desk.jpg"),
        ]
        for wp in wood_paths:
            if wp.exists():
                img = Image.open(wp)
                result = self.service.predict_single(img)
                self.assertEqual(result["state"], STATE_INCONCLUSIVE, f"Failed on {wp.name}: got {result['state']}")
                self.assertIsNone(result["probability"])
                self.assertNotEqual(result["state"], STATE_ANEMIA)
                self.assertNotEqual(result["state"], STATE_NO_ANEMIA)

    def test_all_ood_samples_rejection(self):
        """Verify all synthetic/real OOD samples in fixtures are rejected as INCONCLUSIVE."""
        ood_dir = Path("tests/fixtures/ood")
        if not ood_dir.exists():
            ood_dir = Path("test_ood")
        ood_images = list(ood_dir.glob("*.jpg"))
        self.assertGreater(len(ood_images), 0, "No OOD test images found")
        
        for img_p in ood_images:
            img = Image.open(img_p)
            result = self.service.predict_single(img)
            
            self.assertEqual(
                result["state"],
                STATE_INCONCLUSIVE,
                f"OOD sample {img_p.name} was not rejected! Output state: {result['state']}, prob: {result.get('probability')}"
            )
            self.assertIsNone(result["probability"])
            self.assertIn("description", result)
            self.assertGreater(len(result["description"]), 0)


if __name__ == "__main__":
    unittest.main()
