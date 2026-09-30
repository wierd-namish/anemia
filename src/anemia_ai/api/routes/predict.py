"""
Inference assessment routes for single and multiple fingernail photographs.
"""

from io import BytesIO
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image

from anemia_ai.core.logging import logger
from anemia_ai.schemas.responses import MultiPredictionResponse, PredictionResponse
from anemia_ai.services.assessment_service import get_assessment_service

router = APIRouter(tags=["Prediction"])


async def decode_upload_image(file: UploadFile) -> Image.Image:
    """Safely decodes an uploaded file into a PIL RGB Image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a valid image.")

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is empty.")
        image = Image.open(BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Failed to decode image %s: %s", file.filename, e)
        raise HTTPException(status_code=400, detail=f"Failed to decode image '{file.filename}': {str(e)}")


@router.post("/predict", response_model=PredictionResponse)
async def predict_single_nail(
    file: UploadFile = File(..., description="Fingernail image file (JPEG, PNG, WEBP)"),
    return_roi_base64: bool = Query(True, description="Whether to return Base64 thumbnail of cropped ROI"),
) -> Dict[str, Any]:
    """
    Evaluates a single fingernail photograph captured via phone camera or uploaded from file.
    Processes image strictly in-memory without persistent disk storage.
    """
    image = await decode_upload_image(file)
    service = get_assessment_service()
    result = service.assess_single_nail(image, return_roi_base64=return_roi_base64)
    return result


@router.post("/predict-multiple", response_model=MultiPredictionResponse)
async def predict_multiple_nails(
    files: List[UploadFile] = File(..., description="2 to 4 fingernail image files"),
) -> Dict[str, Any]:
    """
    Evaluates 2 to 4 fingernail photographs from the same patient.
    Aggregates per-nail predictions using mean calibrated probability.
    """
    if len(files) < 2 or len(files) > 4:
        raise HTTPException(
            status_code=400,
            detail="Multi-nail assessment requires between 2 and 4 nail images.",
        )

    images: List[Image.Image] = []
    for f in files:
        img = await decode_upload_image(f)
        images.append(img)

    service = get_assessment_service()
    result = service.assess_multiple_nails(images)
    return result
