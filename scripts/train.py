"""
CLI Entry point for reproducible model training.
Usage:
    python scripts/train.py --config configs/training/efficientnet_b0_v002.yaml
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from anemia_ai.config.settings import get_settings
from anemia_ai.core.logging import logger
from anemia_ai.training.trainer import ModelTrainer


def load_yaml_config(config_path: Path) -> dict:
    """Loads YAML or JSON configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        # Fallback simple json/dict parser
        return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Train Anemia AI Deep Vision Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/efficientnet_b0_v002.json",
        help="Path to training YAML or JSON configuration file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path

    # Fallback to json if yaml requested but json exists
    if not config_path.exists() and config_path.with_suffix(".json").exists():
        config_path = config_path.with_suffix(".json")

    logger.info("Loading training configuration from: %s", config_path)
    config = load_yaml_config(config_path)

    settings = get_settings()
    splits_dir = settings.data_dir / "splits"

    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    cal_path = splits_dir / "calibration.csv"
    test_path = splits_dir / "test.csv"

    if not all(p.exists() for p in [train_path, val_path, cal_path, test_path]):
        logger.error("Dataset split files missing in %s", splits_dir)
        sys.exit(1)

    splits = {
        "train": pd.read_csv(train_path),
        "val": pd.read_csv(val_path),
        "calibration": pd.read_csv(cal_path),
        "test": pd.read_csv(test_path),
    }

    trainer = ModelTrainer(config)
    summary = trainer.train(splits)

    print("\n" + "=" * 70)
    print("TRAINING FINISHED SUCCESSFULLY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
