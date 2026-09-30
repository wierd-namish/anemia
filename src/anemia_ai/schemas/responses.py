"""
Pydantic schemas for structured API responses.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health check and runtime status response."""
    status: str = Field(..., description="Overall service status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Whether inference models are loaded in memory")
    model_name: str = Field(..., description="Name of the active model pipeline")
    model_version: str = Field(..., description="Active pipeline version")
    primary_model: str = Field(..., description="Primary vision model identifier")
    secondary_model: str = Field(..., description="Secondary feature model identifier")
    device: str = Field(..., description="Inference compute device (cpu/cuda)")
    gpu: str = Field(..., description="GPU device name if available")
    api_version: Optional[str] = Field("v1", description="API version")


class ModelInfoResponse(BaseModel):
    """Comprehensive metadata describing the deployed model and training lineage."""
    model: str = Field(..., description="Active ensemble or model name")
    primary_model: str = Field(..., description="Primary deep learning architecture")
    secondary_model: str = Field(..., description="Secondary handcrafted model")
    fusion_version: str = Field(..., description="Ensemble fusion model version")
    calibrator_version: str = Field(..., description="Probability calibration version")
    threshold_version: str = Field(..., description="Locked decision threshold version")
    threshold: float = Field(..., description="Locked operating decision threshold")
    locked_threshold: float = Field(..., description="Locked operating decision threshold")
    device: str = Field(..., description="Execution compute device")
    gpu: str = Field(..., description="GPU device name")
    cuda_version: Optional[str] = Field(None, description="CUDA runtime version")
    input_type: str = Field("Fingernail ROI (224x224 RGB)", description="Expected input modality")
    preprocessing: str = Field(..., description="Preprocessing pipeline version")
    clinical_development_population: str = Field(..., description="Population dataset lineage")
    disclaimer: str = Field(..., description="Medical and investigational disclaimer")
    research_status_disclaimer: str = Field(..., description="Regulatory and research status")


class QualityMetrics(BaseModel):
    """Image quality assessment metrics."""
    width: Optional[int] = None
    height: Optional[int] = None
    laplacian_variance: Optional[float] = None
    mean_luminance: Optional[float] = None
    glare_ratio: Optional[float] = None
    unnatural_color_ratio: Optional[float] = None
    error_type: Optional[str] = None
    threshold: Optional[Union[float, str]] = None


class RoiMetadata(BaseModel):
    """Nail region-of-interest detection metadata."""
    bbox: Optional[Tuple[int, int, int, int]] = None
    method: Optional[str] = None
    original_size: Optional[Tuple[int, int]] = None
    crop_dimensions: Optional[Tuple[int, int]] = None
    target_size: Optional[Tuple[int, int]] = None
    is_valid_nail_roi: Optional[bool] = None
    roi_validation_message: Optional[str] = None
    roi_metrics: Optional[Dict[str, Any]] = None


class LatencyBreakdown(BaseModel):
    """Per-stage inference latency in milliseconds."""
    efficientnet: Optional[float] = None
    jetx_gt: Optional[float] = None
    fusion: Optional[float] = None
    total: float


class PredictionResponse(BaseModel):
    """Structured assessment output for single fingernail photograph."""
    request_id: str = Field(..., description="Unique UUID for request tracing")
    success: bool = Field(..., description="Whether inference completed without fatal error")
    state: str = Field(..., description="Diagnostic state: ANEMIA, NO_ANEMIA, or INCONCLUSIVE")
    probability: Optional[float] = Field(None, description="Calibrated anemia risk probability [0.0 - 1.0]")
    raw_fusion_probability: Optional[float] = Field(None, description="Pre-calibration fusion score")
    efficientnet_probability: Optional[float] = Field(None, description="EfficientNet-B0 probability")
    efficientnet_raw_logit: Optional[float] = Field(None, description="EfficientNet-B0 raw output logit")
    jetx_gt_probability: Optional[float] = Field(None, description="JetX-GT color model probability")
    threshold: float = Field(..., description="Locked operating decision threshold")
    device: Optional[str] = Field("cpu", description="Compute device used for inference")
    model_version: str = Field(..., description="Active ensemble or model version")
    primary_model: Optional[str] = Field(None, description="Primary model version")
    secondary_model: Optional[str] = Field(None, description="Secondary model version")
    calibration_version: Optional[str] = Field(None, description="Calibration model version")
    threshold_version: Optional[str] = Field(None, description="Locked threshold version")
    latency_ms: Optional[Union[LatencyBreakdown, Dict[str, float]]] = None
    description: str = Field(..., description="Human-readable result explanation")
    disclaimer: str = Field(..., description="Medical and regulatory disclaimer")
    roi_metadata: Optional[Union[RoiMetadata, Dict[str, Any]]] = None
    roi_image_base64: Optional[str] = Field(None, description="Base64 encoded cropped nail ROI JPEG preview")
    quality_metrics: Optional[Union[QualityMetrics, Dict[str, Any]]] = None
    reason: Optional[str] = None


class MultiPredictionResponse(BaseModel):
    """Structured assessment output for multi-nail aggregation."""
    request_id: str = Field(..., description="Unique UUID for request tracing")
    success: bool = Field(..., description="Whether aggregation completed")
    state: str = Field(..., description="Aggregated diagnostic state")
    probability: Optional[float] = Field(None, description="Mean aggregated calibrated probability")
    aggregation: str = Field("mean", description="Aggregation method used")
    valid_images: int = Field(..., description="Number of nail images passing quality checks")
    total_images: int = Field(..., description="Total number of submitted nail images")
    threshold: float = Field(..., description="Locked operating decision threshold")
    per_image_results: List[Dict[str, Any]] = Field(..., description="Individual assessment results per image")
    description: Optional[str] = None
    disclaimer: Optional[str] = None


class ErrorResponse(BaseModel):
    """Consistent JSON error response."""
    success: bool = False
    error: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
