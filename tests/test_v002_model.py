"""
Unit tests for EfficientNet-B0 v002 model checkpoint, architecture, and forward pass.
"""

import unittest
from pathlib import Path
import torch
import torchvision.models as models

BASE_DIR = Path(__file__).resolve().parent.parent

class TestEfficientNetB0V002Model(unittest.TestCase):
    def setUp(self):
        self.model_path = BASE_DIR / "experiments/efficientnet_b0_v002/best_model.pth"
        self.config_path = BASE_DIR / "experiments/efficientnet_b0_v002/config.json"

    def test_checkpoint_exists(self):
        self.assertTrue(self.model_path.exists(), "v002 checkpoint best_model.pth must exist")
        self.assertTrue(self.config_path.exists(), "v002 config.json must exist")

    def test_architecture_and_parameters(self):
        ckpt = torch.load(self.model_path, map_location="cpu")
        self.assertEqual(ckpt.get("architecture"), "efficientnet_b0")
        self.assertEqual(ckpt.get("model_version"), "efficientnet_b0_v002")

        state_dict = ckpt["state_dict"]
        # Verify first conv filter statistics are non-zero and learned
        first_conv = state_dict["features.0.0.weight"]
        self.assertFalse(torch.all(first_conv == 0))
        
        # Verify classifier linear weights
        clf_weight = state_dict["classifier.1.weight"]
        self.assertEqual(clf_weight.shape, (1, 1280))
        self.assertFalse(torch.all(clf_weight == 0))

    def test_forward_pass_non_constant(self):
        ckpt = torch.load(self.model_path, map_location="cpu")
        model = models.efficientnet_b0(weights=None)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(1280, 1)
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # Two distinct random inputs
        x1 = torch.randn(1, 3, 224, 224)
        x2 = torch.randn(1, 3, 224, 224) + 2.0

        with torch.no_grad():
            out1 = model(x1).item()
            out2 = model(x2).item()

        self.assertNotAlmostEqual(out1, out2, places=3, msg="Model output must depend on input tensor")

if __name__ == "__main__":
    unittest.main()
