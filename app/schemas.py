"""Pydantic schemas for request/response validation."""
from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: str
    username: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class PredictionResponse(BaseModel):
    id: str
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    gradcam_image_base64: str
    llm_report: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionHistoryItem(BaseModel):
    id: str
    original_filename: str
    predicted_class: str
    confidence: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionHistoryDetail(PredictionHistoryItem):
    class_probabilities: Dict[str, float]
    llm_report: Optional[str] = None
    gradcam_image_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

    # Pydantic reserves the "model_" prefix for its own internals and warns
    # on fields like `model_loaded`. It's a false positive here (this is our
    # field, not a pydantic internal), so we explicitly clear the protected
    # namespace check rather than rename a perfectly clear field name.
    model_config = ConfigDict(protected_namespaces=())