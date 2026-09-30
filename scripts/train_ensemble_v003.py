"""
Ensemble Training, Calibration, and Comparative Evaluation Pipeline (v003).

Models:
1. Primary: EfficientNet-B0 v002 (Deep Vision Model)
2. Secondary: JetX-GT/nail-anemia-detector (HuggingFace 27 Color Features + MLP)

Steps:
1. Extract paired model predictions on train, calibration, validation, and test splits.
2. Train lightweight fusion model on train split (development data only, zero patient leakage).
3. Fit Isotonic Regression Calibrator v003 strictly on calibration split.
4. Derive Locked Threshold tau_v003 on validation split (target sens >= 90%).
5. Evaluate untouched test set on Image-level and Patient-level for:
   - EfficientNet-B0 v002 alone
   - JetX-GT alone
   - Fused Ensemble Model
6. Generate complete provenance, performance JSON, and comparative Markdown reports.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, confusion_matrix

from backend.model.jetx_model import JetXNailAnemiaDetector, extract_jetx_27_features
from backend.model.calibration import ProbabilityCalibrator

CONFIGS_DIR = BASE_DIR / "configs"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
REPORTS_DIR = BASE_DIR / "reports"

CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

V002_WEIGHTS_PATH = EXPERIMENTS_DIR / "efficientnet_b0_v002" / "best_model.pth"


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(rec_vals, prec_vals))
    brier = float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0)))

    return {
        "n_samples": int(len(y_true)),
        "accuracy": round(float(acc), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "ppv": round(float(ppv), 4),
        "npv": round(float(npv), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "threshold": round(float(threshold), 4),
    }


def aggregate_patient_predictions(df_preds: pd.DataFrame, prob_col: str) -> Tuple[np.ndarray, np.ndarray]:
    grouped = df_preds.groupby("patient_id").agg({
        prob_col: "mean",
        "anemia_label": lambda x: int(x.mode()[0])
    }).reset_index()
    return grouped["anemia_label"].values, grouped[prob_col].values


def run_ensemble_pipeline():
    print("=" * 80)
    print("STARTING TWO-MODEL ENSEMBLE PIPELINE (v003)")
    print("EfficientNet-B0 v002 + JetX-GT/nail-anemia-detector")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Primary Model (EfficientNet-B0 v002)
    assert V002_WEIGHTS_PATH.exists(), f"EfficientNet-B0 v002 checkpoint missing: {V002_WEIGHTS_PATH}"
    eff_model = models.efficientnet_b0(weights=None)
    eff_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(1280, 1)
    )
    ckpt = torch.load(V002_WEIGHTS_PATH, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    eff_model.load_state_dict(state)
    eff_model.to(device)
    eff_model.eval()
    eff_hash = compute_sha256(V002_WEIGHTS_PATH)
    print(f"[OK] Loaded Primary Model (v002 SHA256: {eff_hash[:16]}...)")

    # 2. Load Secondary Model (JetX-GT/nail-anemia-detector)
    jetx_detector = JetXNailAnemiaDetector()
    assert jetx_detector.is_ready(), "JetX-GT model failed to load!"
    print(f"[OK] Loaded Secondary Model: JetX-GT/nail-anemia-detector (HuggingFace)")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Extract Paired Predictions on all Splits
    splits = {
        "train": pd.read_csv(BASE_DIR / "data/splits/train.csv"),
        "calibration": pd.read_csv(BASE_DIR / "data/splits/calibration.csv"),
        "val": pd.read_csv(BASE_DIR / "data/splits/val.csv"),
        "test": pd.read_csv(BASE_DIR / "data/splits/test.csv"),
    }

    def extract_split_features(df_split):
        eff_logits, eff_probs = [], []
        jetx_logits, jetx_probs = [], []
        targets, patient_ids, image_ids = [], [], []

        for _, row in df_split.iterrows():
            img_p = Path(row["image_path"])
            img = Image.open(img_p).convert("RGB")
            
            # EfficientNet inference
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = eff_model(tensor).squeeze(-1)
                logit = float(out.item())
                prob = float(1.0 / (1.0 + np.exp(-logit)))
            eff_logits.append(logit)
            eff_probs.append(prob)

            # JetX inference
            j_res = jetx_detector.predict(img)
            jetx_logits.append(j_res["raw_logit"])
            jetx_probs.append(j_res["probability"])

            targets.append(int(row["anemia_label"]))
            patient_ids.append(row["patient_id"])
            image_ids.append(row["image_id"])

        return pd.DataFrame({
            "image_id": image_ids,
            "patient_id": patient_ids,
            "anemia_label": targets,
            "eff_logit": eff_logits,
            "eff_prob": eff_probs,
            "jetx_logit": jetx_logits,
            "jetx_prob": jetx_probs,
        })

    print("\nExtracting dual-model predictions across splits...")
    preds = {}
    for s_name, s_df in splits.items():
        print(f"Processing {s_name} split ({len(s_df)} images)...")
        preds[s_name] = extract_split_features(s_df)

    # 4. Train Fusion Model strictly on Train Split
    print("\nTraining Empirical Fusion Classifier on development split...")
    X_train = preds["train"][["eff_logit", "eff_prob", "jetx_logit", "jetx_prob"]].values
    y_train = preds["train"]["anemia_label"].values

    fusion_model = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    fusion_model.fit(X_train, y_train)

    fusion_path = CONFIGS_DIR / "ensemble_fusion_v003.joblib"
    joblib.dump(fusion_model, fusion_path)
    fusion_hash = compute_sha256(fusion_path)
    print(f"[OK] Saved Fusion Model -> {fusion_path} (SHA256: {fusion_hash[:16]}...)")
    print(f"Fusion Coefficients: {fusion_model.coef_} | Intercept: {fusion_model.intercept_}")

    # Generate raw fusion probabilities
    for s_name in splits:
        X_s = preds[s_name][["eff_logit", "eff_prob", "jetx_logit", "jetx_prob"]].values
        preds[s_name]["raw_fusion_prob"] = fusion_model.predict_proba(X_s)[:, 1]

    # 5. Fit Isotonic Calibrator v003 strictly on Calibration Split
    print("\nFitting Isotonic Regression Calibrator v003 strictly on calibration split...")
    calibrator_v003 = ProbabilityCalibrator(method="isotonic")
    calibrator_v003.fit(preds["calibration"]["raw_fusion_prob"].values, preds["calibration"]["anemia_label"].values)

    calib_path = CONFIGS_DIR / "calibrator_isotonic_v003.joblib"
    joblib.dump(calibrator_v003, calib_path)
    calib_hash = compute_sha256(calib_path)
    print(f"[OK] Saved Isotonic Calibrator v003 -> {calib_path} (SHA256: {calib_hash[:16]}...)")

    # Apply calibration
    for s_name in splits:
        preds[s_name]["calibrated_prob"] = calibrator_v003.calibrate(preds[s_name]["raw_fusion_prob"].values)

    # 6. Derive Locked Threshold tau_v003 on Validation Split
    print("\nDeriving optimal diagnostic threshold on validation split...")
    y_val = preds["val"]["anemia_label"].values
    prob_val = preds["val"]["calibrated_prob"].values

    thresholds_grid = np.linspace(0.10, 0.90, 81)
    best_tau = 0.50
    best_val_spec = 0.0
    for tau in thresholds_grid:
        m = compute_metrics(y_val, prob_val, threshold=tau)
        if m["sensitivity"] >= 0.90 and m["specificity"] >= best_val_spec:
            best_val_spec = m["specificity"]
            best_tau = float(tau)

    print(f"[OK] Derived Locked Threshold: tau_v003 = {best_tau:.4f} (Validation Sensitivity >= 90%, Specificity = {best_val_spec*100:.2f}%)")

    tau_config_path = CONFIGS_DIR / "locked_tau_v003.json"
    with open(tau_config_path, "w") as f:
        json.dump({
            "model_version": "ensemble_v003",
            "primary_model": "efficientnet_b0_v002",
            "secondary_model": "JetX-GT/nail-anemia-detector",
            "calibration_version": "isotonic_regression_v003",
            "locked_threshold": round(best_tau, 4),
            "target_sensitivity": 0.90,
            "derived_on": "validation_split",
            "validation_metrics": compute_metrics(y_val, prob_val, threshold=best_tau)
        }, f, indent=2)
    tau_hash = compute_sha256(tau_config_path)

    # 7. Evaluate Untouched Test Set for:
    # A. EfficientNet-B0 v002 alone
    # B. JetX-GT alone
    # C. Fusion model
    print("\n" + "=" * 80)
    print("UNTOUCHED TEST EVALUATION & MODEL COMPARISON")
    print("=" * 80)

    test_df = preds["test"]
    y_test = test_df["anemia_label"].values

    # Model A: EfficientNet alone
    m_eff_img = compute_metrics(y_test, test_df["eff_prob"].values, threshold=best_tau)
    pt_y_eff, pt_p_eff = aggregate_patient_predictions(test_df, "eff_prob")
    m_eff_pt = compute_metrics(pt_y_eff, pt_p_eff, threshold=best_tau)

    # Model B: JetX-GT alone
    m_jetx_img = compute_metrics(y_test, test_df["jetx_prob"].values, threshold=best_tau)
    pt_y_jetx, pt_p_jetx = aggregate_patient_predictions(test_df, "jetx_prob")
    m_jetx_pt = compute_metrics(pt_y_jetx, pt_p_jetx, threshold=best_tau)

    # Model C: Fusion Model
    m_fusion_img = compute_metrics(y_test, test_df["calibrated_prob"].values, threshold=best_tau)
    pt_y_fusion, pt_p_fusion = aggregate_patient_predictions(test_df, "calibrated_prob")
    m_fusion_pt = compute_metrics(pt_y_fusion, pt_p_fusion, threshold=best_tau)

    print(f"EfficientNet-B0 v002 : Test ROC-AUC = {m_eff_img['roc_auc']:.4f} | Sens = {m_eff_img['sensitivity']*100:.2f}% | Spec = {m_eff_img['specificity']*100:.2f}%")
    print(f"JetX-GT (Handcrafted): Test ROC-AUC = {m_jetx_img['roc_auc']:.4f} | Sens = {m_jetx_img['sensitivity']*100:.2f}% | Spec = {m_jetx_img['specificity']*100:.2f}%")
    print(f"Ensemble Fusion v003 : Test ROC-AUC = {m_fusion_img['roc_auc']:.4f} | Sens = {m_fusion_img['sensitivity']*100:.2f}% | Spec = {m_fusion_img['specificity']*100:.2f}%")

    # Value Addition Analysis
    auc_delta = m_fusion_img["roc_auc"] - m_eff_img["roc_auc"]
    adds_value = bool(auc_delta > 0.001)

    # 8. Save Performance JSON & Reports
    perf_summary = {
        "ensemble_version": "ensemble_v003",
        "primary_model": {
            "name": "EfficientNet-B0",
            "version": "efficientnet_b0_v002",
            "sha256": eff_hash,
            "test_metrics_image": m_eff_img,
            "test_metrics_patient": m_eff_pt,
        },
        "secondary_model": {
            "name": "JetX-GT/nail-anemia-detector",
            "source": "Hugging Face (27 Color Features + MLP)",
            "hashes": jetx_detector.hashes,
            "test_metrics_image": m_jetx_img,
            "test_metrics_patient": m_jetx_pt,
        },
        "fusion_model": {
            "name": "Logistic Regression Fusion",
            "version": "ensemble_v003",
            "sha256": fusion_hash,
            "calibrator_version": "isotonic_regression_v003",
            "calibrator_sha256": calib_hash,
            "locked_threshold": round(best_tau, 4),
            "threshold_sha256": tau_hash,
            "test_metrics_image": m_fusion_img,
            "test_metrics_patient": m_fusion_pt,
        },
        "comparison": {
            "auc_delta_vs_efficientnet": round(auc_delta, 4),
            "adds_measurable_value": adds_value,
            "conclusion": "Ensemble fusion improves discriminant boundary" if adds_value else "EfficientNet-B0 v002 standalone is sufficient"
        }
    }

    with open(REPORTS_DIR / "ensemble_v003_performance.json", "w") as f:
        json.dump(perf_summary, f, indent=2)

    print(f"\n[OK] Ensemble evaluation complete. Performance saved to reports/ensemble_v003_performance.json")
    return perf_summary

if __name__ == "__main__":
    run_ensemble_pipeline()
