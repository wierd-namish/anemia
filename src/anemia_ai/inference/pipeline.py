"""
Inference Pipeline and Orchestration for Anemia Assessment.

Implements end-to-end evaluation:
Image Acquisition -> Quality Gate -> ROI Extraction -> Physiological OOD Validation ->
Model Forward Pass -> Fusion & Probability Calibration -> Locked Thresholding -> Result Schema
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from PIL import Image

from anemia_ai.config.constants import (
    ASSESSMENT_RESULT_DISCLAIMER,
    CALIBRATION_V002_VERSION,
    CALIBRATION_V003_VERSION,
    ENSEMBLE_VERSION,
    PERSISTENT_CLINICAL_DISCLAIMER,
    PREPROCESSING_VERSION,
    PRIMARY_MODEL_VERSION,
    SECONDARY_MODEL_VERSION,
    STATE_ANEMIA,
    STATE_INCONCLUSIVE,
    STATE_NO_ANEMIA,
    THRESHOLD_V002_VERSION,
    THRESHOLD_V003_VERSION,
)
from anemia_ai.config.settings import get_settings
from anemia_ai.core.logging import logger
from anemia_ai.inference.aggregation import aggregate_multi_nail_predictions
from anemia_ai.models.efficientnet import EfficientNetB0Model
from anemia_ai.models.jetx_nail import JetXNailModel
from anemia_ai.preprocessing.image_quality import assess_image_quality
from anemia_ai.preprocessing.nail_detection import NailDetector
from anemia_ai.utils.image import image_to_base64_jpeg
from anemia_ai.utils.timing import benchmark_timer


class TwoModelEnsembleService:
    """
    Two-Model Ensemble Pipeline: EfficientNet-B0 v002 + JetX-GT/nail-anemia-detector
    with empirical logistic fusion, isotonic calibration v003, and locked threshold v003.
    """

    def __init__(self):
        self.settings = get_settings()
        self.detector = NailDetector(target_size=(224, 224), padding_ratio=0.05)
        self.effnet_model = None
        self.jetx_model = None
        self.fusion_model = None
        self.calibrator = None
        self.threshold = self.settings.load_locked_threshold("v003")
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads all required ensemble artifacts."""
        # 1. Primary Model
        try:
            self.effnet_model = EfficientNetB0Model()
        except Exception as e:
            logger.error("Failed to load EfficientNetB0Model: %s", e)
            self.effnet_model = None

        # 2. Secondary Model
        try:
            self.jetx_model = JetXNailModel()
        except Exception as e:
            logger.error("Failed to load JetXNailModel: %s", e)
            self.jetx_model = None

        # 3. Fusion Model
        if self.settings.fusion_model_path.exists():
            try:
                self.fusion_model = joblib.load(self.settings.fusion_model_path)
            except Exception as e:
                logger.error("Failed to load Fusion Model: %s", e)
                self.fusion_model = None

        # 4. Isotonic Calibrator v003
        if self.settings.calibrator_v003_path.exists():
            try:
                self.calibrator = joblib.load(self.settings.calibrator_v003_path)
            except Exception as e:
                logger.error("Failed to load Calibrator v003: %s", e)
                self.calibrator = None

        # 5. Locked Threshold v003
        self.threshold = self.settings.load_locked_threshold("v003")

    @property
    def model(self):
        return self.effnet_model

    def is_ready(self) -> bool:
        """Checks if all components are ready for inference."""
        if self.effnet_model is None or self.jetx_model is None or self.calibrator is None:
            self._load_artifacts()
        return (
            self.effnet_model is not None
            and self.effnet_model.is_ready()
            and self.jetx_model is not None
            and self.jetx_model.is_ready()
            and self.calibrator is not None
        )

    def predict_single(
        self,
        image: Image.Image,
        guide_box: Optional[Tuple[float, float, float, float]] = None,
        return_roi_base64: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs the complete ensemble diagnostic pipeline on a single fingernail photograph.
        """
        req_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        if not self.is_ready():
            return {
                "request_id": req_id,
                "success": False,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "reason": "model_unavailable",
                "description": "The ensemble diagnostic service is currently unavailable.",
                "threshold": self.threshold,
                "model_version": ENSEMBLE_VERSION,
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            }

        # Step 1: Image Quality Gate
        q_pass, q_msg, q_metrics = assess_image_quality(image)
        if not q_pass:
            return {
                "request_id": req_id,
                "success": True,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "reason": q_metrics.get("error_type", "insufficient_image_quality"),
                "description": q_msg,
                "threshold": self.threshold,
                "model_version": ENSEMBLE_VERSION,
                "quality_metrics": q_metrics,
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            }

        # Step 2: Nail ROI Localization
        try:
            cropped_roi, bbox, roi_meta = self.detector.detect_and_crop(image, guide_box=guide_box)
        except Exception as e:
            return {
                "request_id": req_id,
                "success": True,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "reason": "roi_extraction_failed",
                "description": f"Image could not be assessed reliably. Nail ROI localization failed: {str(e)}",
                "threshold": self.threshold,
                "model_version": ENSEMBLE_VERSION,
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            }

        # Step 3: ROI Validation (OOD Defense)
        if not roi_meta.get("is_valid_nail_roi", False):
            return {
                "request_id": req_id,
                "success": True,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "reason": roi_meta.get("roi_metrics", {}).get("rejection_reason", "nail_not_detected"),
                "description": f"Image could not be assessed reliably. {roi_meta.get('roi_validation_message', 'Nail was not detected.')}",
                "threshold": self.threshold,
                "model_version": ENSEMBLE_VERSION,
                "roi_metadata": roi_meta,
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            }

        # Step 4: Model 1 Inference (EfficientNet-B0 v002)
        t_eff_start = time.perf_counter()
        try:
            eff_res = self.effnet_model.predict(cropped_roi)
            eff_prob = eff_res["probability"]
            eff_logit = eff_res["raw_logit"]
        except Exception as e:
            return {
                "request_id": req_id,
                "success": False,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "reason": "primary_inference_error",
                "description": f"Primary model inference error: {str(e)}",
                "threshold": self.threshold,
                "model_version": ENSEMBLE_VERSION,
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            }
        eff_latency_ms = round((time.perf_counter() - t_eff_start) * 1000, 2)

        # Step 5: Model 2 Inference (JetX-GT)
        t_jetx_start = time.perf_counter()
        try:
            jetx_res = self.jetx_model.predict(cropped_roi)
            jetx_prob = jetx_res["probability"]
            jetx_logit = jetx_res["raw_logit"]
        except Exception as e:
            return {
                "request_id": req_id,
                "success": False,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "reason": "secondary_inference_error",
                "description": f"Secondary Hugging Face model inference error: {str(e)}",
                "threshold": self.threshold,
                "model_version": ENSEMBLE_VERSION,
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            }
        jetx_latency_ms = round((time.perf_counter() - t_jetx_start) * 1000, 2)

        # Step 6: Fusion & Calibration
        t_fusion_start = time.perf_counter()
        if self.fusion_model is not None:
            features_vec = np.array([[eff_logit, eff_prob, jetx_logit, jetx_prob]])
            raw_fusion_prob = float(self.fusion_model.predict_proba(features_vec)[0, 1])
        else:
            raw_fusion_prob = (eff_prob + jetx_prob) / 2.0

        # Isotonic Calibration v003
        try:
            calibrated_prob = float(self.calibrator.calibrate(np.array([raw_fusion_prob]))[0])
            calibrated_prob = round(max(0.001, min(0.999, calibrated_prob)), 4)
        except Exception:
            calibrated_prob = round(raw_fusion_prob, 4)

        fusion_latency_ms = round((time.perf_counter() - t_fusion_start) * 1000, 2)
        total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Step 7: Final State Determination at Locked Threshold
        if calibrated_prob >= self.threshold:
            state = STATE_ANEMIA
            clinical_verdict = "ANEMIA DETECTED (ELEVATED RISK)"
            confidence = calibrated_prob
            risk_level = "HIGH RISK" if calibrated_prob >= 0.65 else "MODERATE RISK"
            description = "Model-estimated ensemble probability indicates potential anemia (subungual pallor detected)."
        else:
            state = STATE_NO_ANEMIA
            clinical_verdict = "NO ANEMIA DETECTED (NORMAL / HEALTHY)"
            confidence = round(1.0 - calibrated_prob, 4)
            risk_level = "LOW RISK" if calibrated_prob <= 0.35 else "BORDERLINE NORMAL"
            description = "Model-estimated ensemble probability indicates healthy nail vascularization (no anemia detected)."

        # Subungual Erythema & Vascularization metric
        crop_rgb = np.array(cropped_roi)
        r_mean = float(np.mean(crop_rgb[:, :, 0]))
        g_mean = float(np.mean(crop_rgb[:, :, 1]))
        b_mean = float(np.mean(crop_rgb[:, :, 2]))
        erythema_index = round(float(np.log(max(1.0, r_mean)) - np.log(max(1.0, g_mean))), 4)

        # Preview ROI Image Base64
        roi_b64 = None
        if return_roi_base64:
            try:
                roi_b64 = image_to_base64_jpeg(cropped_roi, quality=90)
            except Exception as e:
                logger.warning("Failed to encode ROI base64: %s", e)

        device_name = str(getattr(self.effnet_model, "device", "cpu"))

        logger.info(
            "Inference complete: req_id=%s, state=%s, verdict=%s, p_anemia=%.4f, conf=%.4f, latency=%.1fms",
            req_id,
            state,
            clinical_verdict,
            calibrated_prob,
            confidence,
            total_latency_ms,
        )

        return {
            "request_id": req_id,
            "success": True,
            "state": state,
            "clinical_verdict": clinical_verdict,
            "confidence": confidence,
            "probability": calibrated_prob,
            "anemia_probability": calibrated_prob,
            "healthy_probability": round(1.0 - calibrated_prob, 4),
            "risk_level": risk_level,
            "raw_fusion_probability": round(raw_fusion_prob, 4),
            "efficientnet_probability": round(eff_prob, 4),
            "efficientnet_raw_logit": round(eff_logit, 6),
            "jetx_gt_probability": round(jetx_prob, 4),
            "threshold": self.threshold,
            "device": device_name,
            "model_name": "EfficientNet-B0 + JetX-GT Ensemble",
            "model_version": ENSEMBLE_VERSION,
            "primary_model": PRIMARY_MODEL_VERSION,
            "secondary_model": SECONDARY_MODEL_VERSION,
            "calibration_version": CALIBRATION_V003_VERSION,
            "threshold_version": THRESHOLD_V003_VERSION,
            "latency_ms": {
                "efficientnet": eff_latency_ms,
                "jetx_gt": jetx_latency_ms,
                "fusion": fusion_latency_ms,
                "total": total_latency_ms,
            },
            "vascularization_metrics": {
                "erythema_index": erythema_index,
                "mean_red": round(r_mean, 2),
                "mean_green": round(g_mean, 2),
                "mean_blue": round(b_mean, 2),
            },
            "description": description,
            "disclaimer": ASSESSMENT_RESULT_DISCLAIMER,
            "roi_metadata": {
                "bbox": roi_meta.get("bbox"),
                "method": roi_meta.get("method"),
                "erythema_index": erythema_index,
            },
            "roi_image_base64": roi_b64,
        }

    def predict_multiple(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Evaluates multiple nail images belonging to a single patient."""
        req_id = str(uuid.uuid4())
        single_results = [self.predict_single(im) for im in images]
        aggregated = aggregate_multi_nail_predictions(single_results, threshold=self.threshold)
        aggregated["request_id"] = req_id
        aggregated["disclaimer"] = ASSESSMENT_RESULT_DISCLAIMER
        return aggregated


class DiagnosticInferenceService:
    """
    Standalone primary model inference pipeline (EfficientNet-B0 v002 alone).
    """

    def __init__(self):
        self.settings = get_settings()
        self.detector = NailDetector(target_size=(224, 224), padding_ratio=0.05)
        self.model = None
        self.calibrator = None
        self.threshold = self.settings.load_locked_threshold("v002")
        self._load_model_and_calibrator()

    def _load_model_and_calibrator(self) -> None:
        try:
            self.model = EfficientNetB0Model(checkpoint_path=self.settings.v002_model_path)
        except Exception as e:
            logger.error("Failed to load EfficientNet-B0 v002: %s", e)
            self.model = None

        if self.settings.calibrator_v002_path.exists():
            try:
                self.calibrator = joblib.load(self.settings.calibrator_v002_path)
            except Exception as e:
                logger.error("Failed to load Calibrator v002: %s", e)
                self.calibrator = None

        self.threshold = self.settings.load_locked_threshold("v002")

    def is_ready(self) -> bool:
        if self.model is None or self.calibrator is None:
            self._load_model_and_calibrator()
        return self.model is not None and self.model.is_ready() and self.calibrator is not None

    def predict_single(
        self,
        image: Image.Image,
        guide_box: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
        req_id = str(uuid.uuid4())

        if not self.is_ready():
            return {
                "request_id": req_id,
                "success": False,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "reason": "model_unavailable",
                "description": "The diagnostic assessment service is currently unavailable.",
                "model_name": "EfficientNet-B0",
                "model_version": PRIMARY_MODEL_VERSION,
                "calibration_version": CALIBRATION_V002_VERSION,
                "threshold": self.threshold,
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            }

        q_pass, q_msg, q_metrics = assess_image_quality(image)
        if not q_pass:
            return {
                "request_id": req_id,
                "success": True,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "reason": q_metrics.get("error_type", "insufficient_image_quality"),
                "description": q_msg,
                "model_name": "EfficientNet-B0",
                "model_version": PRIMARY_MODEL_VERSION,
                "calibration_version": CALIBRATION_V002_VERSION,
                "threshold": self.threshold,
                "quality_metrics": q_metrics,
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            }

        try:
            cropped_roi, bbox, roi_meta = self.detector.detect_and_crop(image, guide_box=guide_box)
        except Exception as e:
            return {
                "request_id": req_id,
                "success": True,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "reason": "roi_extraction_failed",
                "description": f"Image could not be assessed reliably. Nail ROI localization failed: {str(e)}",
                "model_name": "EfficientNet-B0",
                "model_version": PRIMARY_MODEL_VERSION,
                "calibration_version": CALIBRATION_V002_VERSION,
                "threshold": self.threshold,
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            }

        if not roi_meta.get("is_valid_nail_roi", False):
            return {
                "request_id": req_id,
                "success": True,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "reason": roi_meta.get("roi_metrics", {}).get("rejection_reason", "nail_not_detected"),
                "description": f"Image could not be assessed reliably. {roi_meta.get('roi_validation_message', 'Nail was not detected.')}",
                "model_name": "EfficientNet-B0",
                "model_version": PRIMARY_MODEL_VERSION,
                "calibration_version": CALIBRATION_V002_VERSION,
                "threshold": self.threshold,
                "roi_metadata": roi_meta,
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            }

        try:
            res = self.model.predict(cropped_roi)
            logit = res["raw_logit"]
            raw_prob = res["probability"]
        except Exception as e:
            return {
                "request_id": req_id,
                "success": False,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "reason": "inference_error",
                "description": f"Model inference error: {str(e)}",
                "model_name": "EfficientNet-B0",
                "model_version": PRIMARY_MODEL_VERSION,
                "calibration_version": CALIBRATION_V002_VERSION,
                "threshold": self.threshold,
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            }

        try:
            calibrated_prob = float(self.calibrator.calibrate(np.array([raw_prob]))[0])
            calibrated_prob = round(max(0.001, min(0.999, calibrated_prob)), 4)
        except Exception:
            calibrated_prob = round(raw_prob, 4)

        if calibrated_prob >= self.threshold:
            state = STATE_ANEMIA
            description = "Model-estimated probability indicates potential anemia."
        else:
            state = STATE_NO_ANEMIA
            description = "Model-estimated probability indicates no anemia detected."

        return {
            "request_id": req_id,
            "success": True,
            "state": state,
            "probability": calibrated_prob,
            "raw_logit": round(logit, 6),
            "raw_sigmoid": round(raw_prob, 6),
            "threshold": self.threshold,
            "model_name": "EfficientNet-B0",
            "model_version": PRIMARY_MODEL_VERSION,
            "calibration_version": CALIBRATION_V002_VERSION,
            "description": description,
            "disclaimer": ASSESSMENT_RESULT_DISCLAIMER,
            "roi_metadata": {
                "bbox": roi_meta.get("bbox"),
                "method": roi_meta.get("method"),
            },
        }

    def predict_multiple(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Processes multiple nail images belonging to a single subject."""
        req_id = str(uuid.uuid4())
        results = [self.predict_single(img) for img in images]
        valid_predictions = [r for r in results if r.get("state") in (STATE_ANEMIA, STATE_NO_ANEMIA)]

        if not valid_predictions:
            return {
                "request_id": req_id,
                "success": True,
                "state": STATE_INCONCLUSIVE,
                "probability": None,
                "aggregated_probability": None,
                "valid_image_count": 0,
                "total_image_count": len(images),
                "model_name": "EfficientNet-B0",
                "model_version": PRIMARY_MODEL_VERSION,
                "calibration_version": CALIBRATION_V002_VERSION,
                "threshold": self.threshold,
                "description": "All submitted nail images were rejected by quality control or physiological validation.",
                "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
                "per_image_results": results,
            }

        probs = [r["probability"] for r in valid_predictions]
        mean_prob = round(float(np.mean(probs)), 4)
        state = STATE_ANEMIA if mean_prob >= self.threshold else STATE_NO_ANEMIA
        desc = f"Aggregated mean probability across {len(valid_predictions)} valid nail images indicates {'anemia' if state == STATE_ANEMIA else 'no anemia'}."

        return {
            "request_id": req_id,
            "success": True,
            "state": state,
            "probability": mean_prob,
            "aggregated_probability": mean_prob,
            "aggregation_method": "mean",
            "valid_image_count": len(valid_predictions),
            "total_image_count": len(images),
            "threshold": self.threshold,
            "model_name": "EfficientNet-B0",
            "model_version": PRIMARY_MODEL_VERSION,
            "calibration_version": CALIBRATION_V002_VERSION,
            "description": desc,
            "disclaimer": ASSESSMENT_RESULT_DISCLAIMER,
            "per_image_results": results,
        }


# Backwards compatibility alias
InferencePipeline = DiagnosticInferenceService
