"""POST /predict - upload a chest X-ray, get prediction + Grad-CAM + LLM report."""
import base64
import json
import logging
import os
import traceback
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.llm.report_generator import get_report_generator
from app.ml.inference import get_inference_service
from app.ml.preprocess import load_image_from_bytes
from app.models_db import PredictionRecord, User
from app.rate_limit import limiter
from app.routers.auth import get_current_user
from app.schemas import PredictionResponse

logger = logging.getLogger("medical_ai.predict")

router = APIRouter(prefix="/predict", tags=["prediction"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE_MB = 10
GRADCAM_OUTPUT_DIR = os.environ.get("GRADCAM_OUTPUT_DIR", "static/gradcam_outputs")

# TEMPORARY DEBUG SWITCH: when true, unexpected errors return the full
# traceback in the HTTP response body so it's visible in Swagger UI /
# Streamlit directly, without depending on terminal log output being
# visible (useful on Windows where uvicorn --reload subprocess stderr can
# be unreliable in some terminals). Set to False before any real deployment
# -- exposing stack traces to API clients is a security/info-leak risk.
DEBUG_EXPOSE_ERRORS = os.environ.get("DEBUG_EXPOSE_ERRORS", "true").lower() == "true"


@router.post("", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

    try:
        image = load_image_from_bytes(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file. Ensure it is a valid JPEG/PNG.")

    try:
        inference_service = get_inference_service()
        result = inference_service.predict(image)

        report_generator = get_report_generator()
        llm_report = report_generator.generate_report(
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            class_probabilities=result.class_probabilities,
        )

        # Persist Grad-CAM overlay to disk
        os.makedirs(GRADCAM_OUTPUT_DIR, exist_ok=True)
        record_id = str(uuid.uuid4())
        gradcam_path = os.path.join(GRADCAM_OUTPUT_DIR, f"{record_id}.png")
        with open(gradcam_path, "wb") as f:
            f.write(base64.b64decode(result.gradcam_base64_png))

        record = PredictionRecord(
            id=record_id,
            owner_id=current_user.id,
            original_filename=file.filename or "unknown",
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            class_probabilities_json=json.dumps(result.class_probabilities),
            gradcam_image_path=gradcam_path,
            llm_report=llm_report,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return PredictionResponse(
            id=record.id,
            predicted_class=record.predicted_class,
            confidence=record.confidence,
            class_probabilities=result.class_probabilities,
            gradcam_image_base64=result.gradcam_base64_png,
            llm_report=record.llm_report,
            created_at=record.created_at,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        # print() as a belt-and-suspenders fallback in case the logging
        # module's handlers aren't wired to this console for some reason.
        print(tb, flush=True)
        logger.exception("Unhandled error in /predict")
        if DEBUG_EXPOSE_ERRORS:
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}\n\n{tb}",
            )
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")