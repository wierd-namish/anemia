"""
Inference assessment routes for single and multiple fingernail photographs.
"""

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from PIL import Image

from anemia_ai.core.logging import logger
from anemia_ai.schemas.responses import MultiPredictionResponse, PredictionResponse
from anemia_ai.services.assessment_service import get_assessment_service
from anemia_ai.utils.image import decode_base64_to_image

router = APIRouter(tags=["Prediction"])


async def decode_upload_image(file: UploadFile) -> Image.Image:
    """Safely decodes an uploaded file into a PIL RGB Image."""
    if file.content_type and not file.content_type.startswith("image/"):
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
    file: Optional[UploadFile] = File(None, description="Fingernail image file (JPEG, PNG, WEBP)"),
    request: Request = None,
    return_roi_base64: bool = Query(True, description="Whether to return Base64 thumbnail of cropped ROI"),
) -> Dict[str, Any]:
    """
    Evaluates a single fingernail photograph captured via phone camera or uploaded from file.
    Supports both multipart form upload (file) and JSON body with base64 encoded image (image_base64).
    Processes image strictly in-memory without persistent disk storage.
    """
    image: Optional[Image.Image] = None
    guide_box: Optional[Tuple[float, float, float, float]] = None

    # 1. Try Multipart File Upload
    if file is not None and file.filename:
        image = await decode_upload_image(file)
    else:
        # 2. Try JSON / Base64 Body
        if request is not None:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    body = await request.json()
                    b64_str = body.get("image_base64") or body.get("image") or body.get("data")
                    if not b64_str:
                        raise HTTPException(status_code=400, detail="JSON body must contain 'image_base64' string.")
                    image = decode_base64_to_image(b64_str)
                    if "guide_box" in body and body["guide_box"]:
                        guide_box = tuple(body["guide_box"])
                    if "return_roi_base64" in body:
                        return_roi_base64 = bool(body["return_roi_base64"])
                except HTTPException:
                    raise
                except Exception as e:
                    logger.warning("Failed to parse JSON image payload: %s", e)
                    raise HTTPException(status_code=400, detail=f"Invalid JSON image payload: {str(e)}")

    if image is None:
        raise HTTPException(status_code=400, detail="Missing required image. Provide multipart 'file' or JSON 'image_base64'.")

    service = get_assessment_service()
    result = service.assess_single_nail(image, guide_box=guide_box, return_roi_base64=return_roi_base64)
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
