"""ORM models for users and prediction history."""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("PredictionRecord", back_populates="owner")


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id = Column(String, primary_key=True, default=_uuid)
    owner_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)  # nullable for pre-auth legacy rows
    patient_ref = Column(String, nullable=True)  # optional external patient/study identifier
    original_filename = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    class_probabilities_json = Column(Text, nullable=False)  # JSON-encoded dict
    gradcam_image_path = Column(String, nullable=True)       # path to saved heatmap overlay
    llm_report = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    owner = relationship("User", back_populates="predictions")