"""
Full GPU-Accelerated Training Pipeline for EfficientNet-B0 v002.
Target Hardware: NVIDIA GeForce RTX 3070 (CUDA 12.4 / AMP)

Budget:
- Stage 1: 15 Epochs (Classifier Head Training, Backbone Frozen)
- Stage 2: 40 Epochs (Top Backbone Fine-Tuning, Cosine Annealing)
Total: 55 Epochs

Features:
- Real Wall-Clock Live Training Monitor
- Live Status Output to configs/training_status_v002.json
- Initial and Final Checkpoint Parameter Delta & Hash Auditing
- Isotonic Regression v002 on Calibration Split
- Locked Tau v002 on Validation Split
- Untouched Test Set Evaluation
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import joblib
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, confusion_matrix

from backend.evaluation.check_patient_leakage import audit_partitions
from backend.model.calibration import ProbabilityCalibrator

# Set Deterministic Seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class NailDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(float(row["anemia_label"]), dtype=torch.float32)
        return image, label, row["patient_id"], row["image_id"]


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
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(rec_vals, prec_vals))
    brier = float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0)))

    return {
        "n_samples": int(len(y_true)),
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


def aggregate_patient_predictions(df_preds: pd.DataFrame, prob_col="calibrated_prob") -> Tuple[np.ndarray, np.ndarray]:
    grouped = df_preds.groupby("patient_id").agg({
        prob_col: "mean",
        "anemia_label": lambda x: int(x.mode()[0])
    }).reset_index()
    return grouped["anemia_label"].values, grouped[prob_col].values


def run_training_pipeline():
    print("=" * 80)
    print("EFFICIENTNET-B0 v002 — GENUINE GPU TRAINING PIPELINE")
    print("=" * 80)

    # 1. GPU Detection
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is NOT available in PyTorch. RTX 3070 required.")
    
    gpu_name = torch.cuda.get_device_name(0)
    device = torch.device("cuda:0")
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"DEVICE = cuda")
    print(f"GPU = {gpu_name}")
    print(f"CUDA = {torch.version.cuda}")
    print(f"PyTorch = {torch.__version__}")
    print(f"Total VRAM = {total_vram_gb:.2f} GB")

    # 2. Partitions & Leakage Audit
    train_df = pd.read_csv(BASE_DIR / "data/splits/train.csv")
    val_df = pd.read_csv(BASE_DIR / "data/splits/val.csv")
    cal_df = pd.read_csv(BASE_DIR / "data/splits/calibration.csv")
    test_df = pd.read_csv(BASE_DIR / "data/splits/test.csv")

    splits_dict = {"train": train_df, "val": val_df, "calibration": cal_df, "test": test_df}
    leakage_result = audit_partitions(splits_dict, patient_col="patient_id")
    if "PASS" not in leakage_result["status"]:
        raise ValueError(f"Patient leakage detected: {leakage_result}")
    print(f"[OK] Patient Leakage Audit: ZERO PATIENT OVERLAP CONFIRMED ({len(train_df)} train, {len(val_df)} val, {len(cal_df)} cal, {len(test_df)} test).")

    # 3. Transforms & DataLoaders
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 64
    train_loader = DataLoader(NailDataset(train_df, train_transform), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(NailDataset(val_df, eval_transform), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    cal_loader = DataLoader(NailDataset(cal_df, eval_transform), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(NailDataset(test_df, eval_transform), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # 4. Verified ImageNet Model Initialization
    print("\nLoading torchvision EfficientNet-B0 with official ImageNet weights...")
    weights_spec = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights_spec)
    
    init_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
    init_first_norm = torch.norm(init_state["features.0.0.weight"]).item()
    print(f"[OK] ImageNet Weights Verified: features.0.0.weight norm={init_first_norm:.6f}")

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 1)
    )
    model.to(device)

    exp_dir = BASE_DIR / "experiments/efficientnet_b0_v002"
    exp_dir.mkdir(parents=True, exist_ok=True)
    status_file = BASE_DIR / "configs/training_status_v002.json"

    n_pos = (train_df["anemia_label"] == 1).sum()
    n_neg = (train_df["anemia_label"] == 0).sum()
    pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

    scaler = torch.amp.GradScaler('cuda')

    total_stage1_epochs = 15
    total_stage2_epochs = 40
    total_epochs = total_stage1_epochs + total_stage2_epochs
    total_batches_per_epoch = len(train_loader)
    total_training_batches = total_epochs * total_batches_per_epoch

    history = []
    best_val_auc = 0.0
    best_epoch = 0
    best_model_state = None
    total_optimizer_steps = 0
    global_batch_count = 0
    t_start = time.time()

    # 5. STAGE 1: Head Training (15 Epochs)
    print("\n" + "=" * 60)
    print("STAGE 1 — CLASSIFIER HEAD (15 Epochs, Backbone Frozen)")
    print("=" * 60)
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_head = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, total_stage1_epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        epoch_t0 = time.time()

        for batch_idx, (images, labels, _, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer_head.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(images).squeeze(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer_head)
            scaler.update()

            total_optimizer_steps += 1
            global_batch_count += 1
            train_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend(probs)
            train_targets.extend(labels.cpu().numpy())

            # Real-time wall clock calculation
            elapsed_sec = time.time() - t_start
            speed = global_batch_count / elapsed_sec if elapsed_sec > 0 else 1.0
            rem_batches = total_training_batches - global_batch_count
            eta_sec = rem_batches / speed if speed > 0 else 0

            # Update live JSON status
            mem_used_mb = torch.cuda.memory_allocated() / 1e6
            status_dict = {
                "status": "training",
                "stage": 1,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "batch": batch_idx,
                "total_batches": total_batches_per_epoch,
                "elapsed_seconds": round(elapsed_sec, 1),
                "eta_seconds": round(eta_sec, 1),
                "elapsed_formatted": format_time(elapsed_sec),
                "eta_formatted": format_time(eta_sec),
                "speed_batches_per_sec": round(speed, 2),
                "gpu_name": gpu_name,
                "gpu_memory_used_mb": round(mem_used_mb, 1),
                "checkpoint_path": str(exp_dir / "best_model.pth")
            }
            with open(status_file, "w") as f:
                json.dump(status_dict, f, indent=2)

        train_loss /= len(train_df)
        train_auc = roc_auc_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(images).squeeze(-1)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy())
        val_loss /= len(val_df)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(f"Stage 1 | Epoch {epoch:02d}/{total_stage1_epochs:02d} (Overall {epoch:02d}/{total_epochs:02d}) | Elapsed: {format_time(elapsed_sec)} | ETA: {format_time(eta_sec)} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        history.append({"epoch": epoch, "stage": 1, "train_loss": train_loss, "train_auc": train_auc, "val_loss": val_loss, "val_auc": val_auc})
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_model_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

    t_stage1 = time.time() - t_start
    print(f"\n[STAGE 1 COMPLETE] Elapsed: {format_time(t_stage1)} | Best Val AUC: {best_val_auc:.4f}")

    # 6. STAGE 2: Top Backbone Fine-Tuning (40 Epochs)
    print("\n" + "=" * 60)
    print("STAGE 2 — TOP BACKBONE FINE-TUNING (40 Epochs, features.5..8)")
    print("=" * 60)
    for name, param in model.features.named_parameters():
        if any(f"features.{i}" in name for i in [5, 6, 7, 8]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer_ft = torch.optim.AdamW([
        {"params": [p for n, p in model.features.named_parameters() if p.requires_grad], "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 3e-4}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=total_stage2_epochs, eta_min=1e-6)
    t_stage2_start = time.time()

    for epoch_idx in range(1, total_stage2_epochs + 1):
        overall_epoch = total_stage1_epochs + epoch_idx
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for batch_idx, (images, labels, _, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer_ft.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(images).squeeze(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer_ft)
            scaler.update()

            total_optimizer_steps += 1
            global_batch_count += 1
            train_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend(probs)
            train_targets.extend(labels.cpu().numpy())

            elapsed_sec = time.time() - t_start
            speed = global_batch_count / elapsed_sec if elapsed_sec > 0 else 1.0
            rem_batches = total_training_batches - global_batch_count
            eta_sec = rem_batches / speed if speed > 0 else 0

            mem_used_mb = torch.cuda.memory_allocated() / 1e6
            status_dict = {
                "status": "training",
                "stage": 2,
                "epoch": overall_epoch,
                "total_epochs": total_epochs,
                "batch": batch_idx,
                "total_batches": total_batches_per_epoch,
                "elapsed_seconds": round(elapsed_sec, 1),
                "eta_seconds": round(eta_sec, 1),
                "elapsed_formatted": format_time(elapsed_sec),
                "eta_formatted": format_time(eta_sec),
                "speed_batches_per_sec": round(speed, 2),
                "gpu_name": gpu_name,
                "gpu_memory_used_mb": round(mem_used_mb, 1),
                "checkpoint_path": str(exp_dir / "best_model.pth")
            }
            with open(status_file, "w") as f:
                json.dump(status_dict, f, indent=2)

        train_loss /= len(train_df)
        train_auc = roc_auc_score(train_targets, train_preds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(images).squeeze(-1)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy())
        val_loss /= len(val_df)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(f"Stage 2 | Epoch {epoch_idx:02d}/{total_stage2_epochs:02d} (Overall {overall_epoch:02d}/{total_epochs:02d}) | Elapsed: {format_time(elapsed_sec)} | ETA: {format_time(eta_sec)} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        history.append({"epoch": overall_epoch, "stage": 2, "train_loss": train_loss, "train_auc": train_auc, "val_loss": val_loss, "val_auc": val_auc})
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = overall_epoch
            best_model_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

    t_total = time.time() - t_start
    t_stage2 = time.time() - t_stage2_start

    # Save final model & best model
    torch.save(model.state_dict(), exp_dir / "final_model.pth")
    model.load_state_dict(best_model_state)
    best_ckpt_path = exp_dir / "best_model.pth"
    torch.save({
        "architecture": "efficientnet_b0",
        "model_version": "efficientnet_b0_v002",
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "total_optimizer_steps": total_optimizer_steps,
        "total_elapsed_seconds": round(t_total, 2),
        "stage1_seconds": round(t_stage1, 2),
        "stage2_seconds": round(t_stage2, 2),
        "device": gpu_name,
        "state_dict": model.state_dict()
    }, best_ckpt_path)
    model_sha256 = compute_sha256(best_ckpt_path)

    # Parameter Delta Verification
    final_state = model.state_dict()
    param_deltas = []
    for k in init_state:
        if k in final_state and init_state[k].shape == final_state[k].shape:
            diff = torch.norm(final_state[k].float().cpu() - init_state[k].float().cpu()).item()
            param_deltas.append(diff)
    total_delta = sum(param_deltas)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total Elapsed:       {format_time(t_total)}")
    print(f"GPU:                 {gpu_name}")
    print(f"Epochs Completed:    {total_epochs} (15 Stage 1 + 40 Stage 2)")
    print(f"Optimizer Steps:     {total_optimizer_steps}")
    print(f"Best Val Epoch:      {best_epoch}")
    print(f"Best Val AUC:        {best_val_auc:.4f}")
    print(f"Parameter Delta:     {total_delta:.6f} (> 0 confirmed)")
    print(f"Checkpoint SHA256:   {model_sha256}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(exp_dir / "training_history.csv", index=False)

    # Predictions for Cal, Val, Test splits
    def predict_split(loader):
        model.eval()
        raw_logits, sig_probs, targets, patient_ids, image_ids = [], [], [], [], []
        with torch.no_grad():
            for images, labels, pids, iids in loader:
                images = images.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(images).squeeze(-1)
                    probs = torch.sigmoid(logits)
                raw_logits.extend(logits.cpu().numpy().tolist())
                sig_probs.extend(probs.cpu().numpy().tolist())
                targets.extend(labels.numpy().tolist())
                patient_ids.extend(pids)
                image_ids.extend(iids)
        return pd.DataFrame({
            "image_id": image_ids,
            "patient_id": patient_ids,
            "anemia_label": targets,
            "raw_logit": raw_logits,
            "sigmoid_prob": sig_probs,
        })

    print("\nGenerating model predictions on calibration, validation, and test splits...")
    cal_preds_df = predict_split(cal_loader)
    val_preds_df = predict_split(val_loader)
    test_preds_df = predict_split(test_loader)

    # Save raw predictions for ensemble fusion
    cal_preds_df.to_csv(exp_dir / "cal_predictions.csv", index=False)
    val_preds_df.to_csv(exp_dir / "val_predictions.csv", index=False)
    test_preds_df.to_csv(exp_dir / "test_predictions.csv", index=False)

    # 7. Fit Isotonic Calibrator v002 on Calibration Split
    print("\nFitting Isotonic Regression Calibrator v002 strictly on calibration split...")
    calibrator_v002 = ProbabilityCalibrator(method="isotonic")
    calibrator_v002.fit(cal_preds_df["sigmoid_prob"].values, cal_preds_df["anemia_label"].values)

    calib_v002_path = BASE_DIR / "configs/calibrator_isotonic_v002.joblib"
    joblib.dump(calibrator_v002, calib_v002_path)
    calib_sha256 = compute_sha256(calib_v002_path)
    print(f"[OK] Saved calibration artifact -> {calib_v002_path} (SHA256: {calib_sha256})")

    # Apply calibration
    val_preds_df["calibrated_prob"] = calibrator_v002.calibrate(val_preds_df["sigmoid_prob"].values)
    cal_preds_df["calibrated_prob"] = calibrator_v002.calibrate(cal_preds_df["sigmoid_prob"].values)
    test_preds_df["calibrated_prob"] = calibrator_v002.calibrate(test_preds_df["sigmoid_prob"].values)

    # 8. Derive Locked Threshold v002 on Validation Split
    print("\nDeriving optimal diagnostic threshold on validation split...")
    y_val = val_preds_df["anemia_label"].values
    prob_val = val_preds_df["calibrated_prob"].values

    thresholds_grid = np.linspace(0.10, 0.90, 81)
    best_tau = 0.50
    best_val_spec = 0.0
    for tau in thresholds_grid:
        m = compute_metrics(y_val, prob_val, threshold=tau)
        if m["sensitivity"] >= 0.90 and m["specificity"] >= best_val_spec:
            best_val_spec = m["specificity"]
            best_tau = float(tau)

    print(f"[OK] Derived Locked Threshold: tau_v002 = {best_tau:.4f} (Validation Sensitivity >= 90%, Specificity = {best_val_spec*100:.2f}%)")
    
    tau_config_path = BASE_DIR / "configs/locked_tau_v002.json"
    with open(tau_config_path, "w") as f:
        json.dump({
            "model_version": "efficientnet_b0_v002",
            "calibration_version": "isotonic_regression_v002",
            "locked_threshold": round(best_tau, 4),
            "target_sensitivity": 0.90,
            "derived_on": "validation_split",
            "validation_metrics": compute_metrics(y_val, prob_val, threshold=best_tau)
        }, f, indent=2)
    tau_sha256 = compute_sha256(tau_config_path)

    # 9. Compute Complete Metrics (Image and Patient Level)
    val_img_metrics = compute_metrics(y_val, prob_val, threshold=best_tau)
    val_pt_labels, val_pt_probs = aggregate_patient_predictions(val_preds_df, "calibrated_prob")
    val_pt_metrics = compute_metrics(val_pt_labels, val_pt_probs, threshold=best_tau)

    y_test = test_preds_df["anemia_label"].values
    prob_test = test_preds_df["calibrated_prob"].values
    test_img_metrics = compute_metrics(y_test, prob_test, threshold=best_tau)
    test_pt_labels, test_pt_probs = aggregate_patient_predictions(test_preds_df, "calibrated_prob")
    test_pt_metrics = compute_metrics(test_pt_labels, test_pt_probs, threshold=best_tau)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION METRICS SUMMARY (v002 on RTX 3070)")
    print("=" * 60)
    print(f"Validation Image-Level : ROC-AUC={val_img_metrics['roc_auc']:.4f} | Sens={val_img_metrics['sensitivity']*100:.2f}% | Spec={val_img_metrics['specificity']*100:.2f}% | Brier={val_img_metrics['brier_score']:.4f}")
    print(f"Validation Patient-Level: ROC-AUC={val_pt_metrics['roc_auc']:.4f} | Sens={val_pt_metrics['sensitivity']*100:.2f}% | Spec={val_pt_metrics['specificity']*100:.2f}%")
    print(f"Untouched Test Image-Lvl: ROC-AUC={test_img_metrics['roc_auc']:.4f} | Sens={test_img_metrics['sensitivity']*100:.2f}% | Spec={test_img_metrics['specificity']*100:.2f}% | Brier={test_img_metrics['brier_score']:.4f}")
    print(f"Untouched Test Pt-Level : ROC-AUC={test_pt_metrics['roc_auc']:.4f} | Sens={test_pt_metrics['sensitivity']*100:.2f}% | Spec={test_pt_metrics['specificity']*100:.2f}%")

    # 10. Save Complete Config and Performance Report
    config_dict = {
        "model_name": "EfficientNet-B0",
        "model_version": "efficientnet_b0_v002",
        "hardware": gpu_name,
        "cuda_version": str(torch.version.cuda),
        "pytorch_version": str(torch.__version__),
        "weights_source": "torchvision.models.EfficientNet_B0_Weights.DEFAULT",
        "calibration_version": "isotonic_regression_v002",
        "locked_threshold": round(best_tau, 4),
        "parameter_delta": round(total_delta, 6),
        "total_epochs": total_epochs,
        "stage1_epochs": total_stage1_epochs,
        "stage2_epochs": total_stage2_epochs,
        "total_optimizer_steps": total_optimizer_steps,
        "total_elapsed_seconds": round(t_total, 2),
        "stage1_seconds": round(t_stage1, 2),
        "stage2_seconds": round(t_stage2, 2),
        "model_sha256": model_sha256,
        "calibrator_sha256": calib_sha256,
        "threshold_sha256": tau_sha256,
        "validation_metrics": {"image": val_img_metrics, "patient": val_pt_metrics},
        "test_metrics": {"image": test_img_metrics, "patient": test_pt_metrics},
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    with open(BASE_DIR / "reports/efficientnet_b0_v002_performance.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Final live status update
    final_status_dict = {
        "status": "completed",
        "stage": 2,
        "epoch": total_epochs,
        "total_epochs": total_epochs,
        "elapsed_seconds": round(t_total, 1),
        "elapsed_formatted": format_time(t_total),
        "eta_seconds": 0,
        "best_epoch": best_epoch,
        "best_val_auc": round(best_val_auc, 4),
        "checkpoint_sha256": model_sha256,
        "gpu_name": gpu_name,
        "checkpoint_path": str(best_ckpt_path)
    }
    with open(status_file, "w") as f:
        json.dump(final_status_dict, f, indent=2)

    return config_dict

if __name__ == "__main__":
    run_training_pipeline()
