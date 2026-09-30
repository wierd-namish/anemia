"""
Reproducible Training Engine for Deep Convolutional Anemia Models.
"""

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from anemia_ai.calibration.calibrator import ProbabilityCalibrator
from anemia_ai.core.logging import logger
from anemia_ai.evaluation.leakage import audit_partitions
from anemia_ai.evaluation.metrics import aggregate_patient_predictions, compute_diagnostic_metrics
from anemia_ai.models.cnn_factory import create_cnn_backbone
from anemia_ai.preprocessing.transforms import get_inference_transform, get_training_transform
from anemia_ai.training.dataset import NailDataset
from anemia_ai.utils.hashing import compute_sha256
from anemia_ai.utils.timing import format_duration


class ModelTrainer:
    """Configurable training engine for deep vision models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.seed = config.get("seed", 42)
        self._set_seed()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu"
        )
        self.output_dir = Path(config.get("output_dir", "experiments/efficientnet_b0_v002"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self, data_splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Runs two-stage training, calibration, and test evaluation."""
        logger.info("Starting model training on %s", self.device)

        # 1. Audit Splits
        leakage = audit_partitions(data_splits, patient_col="patient_id")
        if "FAIL" in leakage["status"]:
            raise ValueError(f"Data leakage detected: {leakage}")

        train_df = data_splits["train"]
        val_df = data_splits["val"]
        cal_df = data_splits["calibration"]
        test_df = data_splits["test"]

        # 2. Data Loaders
        batch_size = self.config.get("batch_size", 64)
        train_loader = DataLoader(
            NailDataset(train_df, get_training_transform()),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            NailDataset(val_df, get_inference_transform()),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        cal_loader = DataLoader(
            NailDataset(cal_df, get_inference_transform()),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            NailDataset(test_df, get_inference_transform()),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        # 3. Model Architecture
        model_name = self.config.get("architecture", "efficientnet_b0")
        model = create_cnn_backbone(architecture=model_name, pretrained=True)
        model.to(self.device)

        init_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

        # 4. Loss & Mixed Precision
        n_pos = (train_df["anemia_label"] == 1).sum()
        n_neg = (train_df["anemia_label"] == 0).sum()
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device))

        use_amp = torch.cuda.is_available() and self.config.get("use_amp", True)
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        stage1_epochs = self.config.get("stage1_epochs", 15)
        stage2_epochs = self.config.get("stage2_epochs", 40)
        total_epochs = stage1_epochs + stage2_epochs

        history = []
        best_val_auc = 0.0
        best_epoch = 0
        best_model_state = None
        t_start = time.perf_counter()

        # Stage 1: Freeze Backbone, Train Classifier
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

        opt_head = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(1, stage1_epochs + 1):
            model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []

            for images, labels, _, _ in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                opt_head.zero_grad()

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(images).squeeze(-1)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(opt_head)
                    scaler.update()
                else:
                    logits = model(images).squeeze(-1)
                    loss = criterion(logits, labels)
                    loss.backward()
                    opt_head.step()

                train_loss += loss.item() * len(labels)
                train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
                train_targets.extend(labels.cpu().numpy())

            train_loss /= len(train_df)
            train_auc = roc_auc_score(train_targets, train_preds)

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for images, labels, _, _ in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = model(images).squeeze(-1)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * len(labels)
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            val_loss /= len(val_df)
            val_auc = roc_auc_score(val_targets, val_preds)

            logger.info("Stage 1 | Epoch %02d/%02d | Train AUC: %.4f | Val AUC: %.4f", epoch, stage1_epochs, train_auc, val_auc)
            history.append({"epoch": epoch, "stage": 1, "train_loss": train_loss, "train_auc": train_auc, "val_loss": val_loss, "val_auc": val_auc})

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_model_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

        # Stage 2: Fine-Tune Top Backbone
        for name, param in model.features.named_parameters():
            if any(f"features.{i}" in name for i in [5, 6, 7, 8]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        opt_ft = torch.optim.AdamW([
            {"params": [p for n, p in model.features.named_parameters() if p.requires_grad], "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 3e-4},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=stage2_epochs, eta_min=1e-6)

        for ep_idx in range(1, stage2_epochs + 1):
            epoch = stage1_epochs + ep_idx
            model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []

            for images, labels, _, _ in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                opt_ft.zero_grad()

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(images).squeeze(-1)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(opt_ft)
                    scaler.update()
                else:
                    logits = model(images).squeeze(-1)
                    loss = criterion(logits, labels)
                    loss.backward()
                    opt_ft.step()

                train_loss += loss.item() * len(labels)
                train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
                train_targets.extend(labels.cpu().numpy())

            train_loss /= len(train_df)
            train_auc = roc_auc_score(train_targets, train_preds)
            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for images, labels, _, _ in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = model(images).squeeze(-1)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * len(labels)
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            val_loss /= len(val_df)
            val_auc = roc_auc_score(val_targets, val_preds)

            logger.info("Stage 2 | Epoch %02d/%02d | Train AUC: %.4f | Val AUC: %.4f", ep_idx, stage2_epochs, train_auc, val_auc)
            history.append({"epoch": epoch, "stage": 2, "train_loss": train_loss, "train_auc": train_auc, "val_loss": val_loss, "val_auc": val_auc})

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_model_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

        total_elapsed = time.perf_counter() - t_start

        # Save Best Model Checkpoint
        model.load_state_dict(best_model_state)
        best_ckpt_path = self.output_dir / "best_model.pth"
        torch.save({
            "architecture": model_name,
            "model_version": self.config.get("model_version", "efficientnet_b0_v002"),
            "best_epoch": best_epoch,
            "best_val_auc": float(best_val_auc),
            "total_elapsed_seconds": round(total_elapsed, 2),
            "state_dict": model.state_dict(),
        }, best_ckpt_path)
        model_sha256 = compute_sha256(best_ckpt_path)

        # Calibration on Calibration Split
        def predict_loader(loader):
            model.eval()
            sig_probs, targets, pids = [], [], []
            with torch.no_grad():
                for images, labels, p_ids, _ in loader:
                    images = images.to(self.device)
                    logits = model(images).squeeze(-1)
                    probs = torch.sigmoid(logits)
                    sig_probs.extend(probs.cpu().numpy().tolist())
                    targets.extend(labels.numpy().tolist())
                    pids.extend(p_ids)
            return pd.DataFrame({"patient_id": pids, "anemia_label": targets, "sigmoid_prob": sig_probs})

        cal_preds = predict_loader(cal_loader)
        val_preds = predict_loader(val_loader)
        test_preds = predict_loader(test_loader)

        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(cal_preds["sigmoid_prob"].values, cal_preds["anemia_label"].values)
        cal_preds["calibrated_prob"] = calibrator.calibrate(cal_preds["sigmoid_prob"].values)
        val_preds["calibrated_prob"] = calibrator.calibrate(val_preds["sigmoid_prob"].values)
        test_preds["calibrated_prob"] = calibrator.calibrate(test_preds["sigmoid_prob"].values)

        # Derive Threshold
        best_tau = 0.50
        best_spec = 0.0
        for tau in np.linspace(0.10, 0.90, 81):
            m = compute_diagnostic_metrics(val_preds["anemia_label"].values, val_preds["calibrated_prob"].values, threshold=tau)
            if m["sensitivity"] >= 0.90 and m["specificity"] >= best_spec:
                best_spec = m["specificity"]
                best_tau = float(tau)

        val_metrics = compute_diagnostic_metrics(val_preds["anemia_label"].values, val_preds["calibrated_prob"].values, threshold=best_tau)
        test_metrics = compute_diagnostic_metrics(test_preds["anemia_label"].values, test_preds["calibrated_prob"].values, threshold=best_tau)

        report = {
            "model_version": self.config.get("model_version", "efficientnet_b0_v002"),
            "best_epoch": best_epoch,
            "best_val_auc": round(float(best_val_auc), 4),
            "locked_threshold": round(best_tau, 4),
            "checkpoint_path": str(best_ckpt_path),
            "model_sha256": model_sha256,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "elapsed_time": format_duration(total_elapsed),
        }
        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
