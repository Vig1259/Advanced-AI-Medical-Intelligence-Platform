"""SQLAlchemy database setup. Defaults to SQLite; swap DATABASE_URL for Postgres in prod."""

import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./medical_ai.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    # Import models so SQLAlchemy registers them
    from app import models_db  # noqa: F401

    # Create tables that don't already exist
    Base.metadata.create_all(bind=engine)

    # Development-only schema update for SQLite
    inspector = inspect(engine)

    if "prediction_history" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("prediction_history")]

        if "owner_id" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE prediction_history "
                        "ADD COLUMN owner_id TEXT"
                    )
                )
                print("Added missing owner_id column.")
