"""
CLI Evaluation Script for Models and Ensembles on Untouched Partitions.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from anemia_ai.config.settings import get_settings
from anemia_ai.core.logging import logger
from anemia_ai.evaluation.metrics import aggregate_patient_predictions, compute_diagnostic_metrics
from anemia_ai.inference.pipeline import TwoModelEnsembleService


def main():
    parser = argparse.ArgumentParser(description="Evaluate Anemia AI Model Performance")
    parser.add_argument("--split", type=str, default="test", help="Split name to evaluate ('test', 'val', 'calibration')")
    args = parser.parse_args()

    settings = get_settings()
    split_file = settings.data_dir / "splits" / f"{args.split}.csv"
    if not split_file.exists():
        logger.error("Split file not found: %s", split_file)
        sys.exit(1)

    df_split = pd.read_csv(split_file)
    logger.info("Evaluating on split '%s' (%d images)...", args.split, len(df_split))

    service = TwoModelEnsembleService()
    if not service.is_ready():
        logger.error("Inference service is not ready.")
        sys.exit(1)

    calibrated_probs = []
    eff_probs = []
    jetx_probs = []
    targets = []
    patient_ids = []

    for idx, row in df_split.iterrows():
        img_p = Path(row["image_path"])
        if not img_p.is_absolute():
            img_p = BASE_DIR / img_p
        if not img_p.exists():
            continue

        img = Image.open(img_p).convert("RGB")
        res = service.predict_single(img)

        if res.get("probability") is not None:
            calibrated_probs.append(res["probability"])
            eff_probs.append(res.get("efficientnet_probability", 0.5))
            jetx_probs.append(res.get("jetx_gt_probability", 0.5))
            targets.append(int(row["anemia_label"]))
            patient_ids.append(row["patient_id"])

    df_results = pd.DataFrame({
        "patient_id": patient_ids,
        "anemia_label": targets,
        "calibrated_prob": calibrated_probs,
        "eff_prob": eff_probs,
        "jetx_prob": jetx_probs,
    })

    tau = service.threshold
    img_metrics = compute_diagnostic_metrics(df_results["anemia_label"].values, df_results["calibrated_prob"].values, threshold=tau)
    pt_y, pt_p = aggregate_patient_predictions(df_results, "calibrated_prob")
    pt_metrics = compute_diagnostic_metrics(pt_y, pt_p, threshold=tau)

    print("=" * 70)
    print(f"EVALUATION REPORT — SPLIT: {args.split.upper()} (Threshold tau={tau:.4f})")
    print("=" * 70)
    print(f"Image-Level (n={len(df_results)}):")
    print(f"  ROC-AUC:     {img_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:      {img_metrics['pr_auc']:.4f}")
    print(f"  Sensitivity: {img_metrics['sensitivity']*100:.2f}%")
    print(f"  Specificity: {img_metrics['specificity']*100:.2f}%")
    print(f"  Brier Score: {img_metrics['brier_score']:.4f}")
    print(f"\nPatient-Level (n={len(pt_y)}):")
    print(f"  ROC-AUC:     {pt_metrics['roc_auc']:.4f}")
    print(f"  Sensitivity: {pt_metrics['sensitivity']*100:.2f}%")
    print(f"  Specificity: {pt_metrics['specificity']*100:.2f}%")


if __name__ == "__main__":
    main()
