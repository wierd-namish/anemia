"""
Patient leakage audit verification test.
"""

import unittest
import pandas as pd
from anemia_ai.config.settings import get_settings
from anemia_ai.evaluation.leakage import audit_partitions


class TestLeakageAudit(unittest.TestCase):
    """Verifies partition independence and absence of subject overlap."""

    def test_partition_leakage_audit(self):
        settings = get_settings()
        splits_dir = settings.data_dir / "splits"

        if not splits_dir.exists():
            self.skipTest("Data splits directory not found.")

        train_path = splits_dir / "train.csv"
        val_path = splits_dir / "val.csv"
        cal_path = splits_dir / "calibration.csv"
        test_path = splits_dir / "test.csv"

        if all(p.exists() for p in [train_path, val_path, cal_path, test_path]):
            splits = {
                "train": pd.read_csv(train_path),
                "val": pd.read_csv(val_path),
                "calibration": pd.read_csv(cal_path),
                "test": pd.read_csv(test_path),
            }
            res = audit_partitions(splits, patient_col="patient_id")
            self.assertIn("PASS", res["status"])
            for pair, count in res["pairwise_overlaps"].items():
                self.assertEqual(count, 0, f"Patient leakage found in {pair}!")


if __name__ == "__main__":
    unittest.main()
