"""GET /history - list and retrieve stored prediction records.

All endpoints require authentication and are scoped to the logged-in
user's own predictions only -- one user can never see or delete another
user's history, even by guessing record IDs.
"""
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models_db import PredictionRecord, User
from app.routers.auth import get_current_user
from app.schemas import PredictionHistoryDetail, PredictionHistoryItem

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=list[PredictionHistoryItem])
def list_history(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    predicted_class: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.owner_id == current_user.id)
        .order_by(PredictionRecord.created_at.desc())
    )
    if predicted_class:
        query = query.filter(PredictionRecord.predicted_class == predicted_class.upper())
    records = query.offset(offset).limit(limit).all()
    return records


@router.get("/{record_id}", response_model=PredictionHistoryDetail)
def get_record(
    record_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    record = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.id == record_id, PredictionRecord.owner_id == current_user.id)
        .first()
    )
    if not record:
        # Same 404 whether the record doesn't exist OR belongs to someone
        # else -- deliberately not distinguishing "not found" from
        # "not yours" to avoid leaking which record IDs exist.
        raise HTTPException(status_code=404, detail="Record not found")

    return PredictionHistoryDetail(
        id=record.id,
        original_filename=record.original_filename,
        predicted_class=record.predicted_class,
        confidence=record.confidence,
        created_at=record.created_at,
        class_probabilities=json.loads(record.class_probabilities_json),
        llm_report=record.llm_report,
        gradcam_image_path=record.gradcam_image_path,
    )


@router.delete("/{record_id}")
def delete_record(
    record_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    record = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.id == record_id, PredictionRecord.owner_id == current_user.id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    db.delete(record)
    db.commit()
    return {"status": "deleted", "id": record_id}