"""
Advanced AI Medical Intelligence Platform - FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Docs available at /docs (Swagger) and /redoc.
"""
import logging
import os
import sys
import traceback

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.database import init_db
from app.ml.inference import get_inference_service
from app.rate_limit import limiter
from app.routers import auth, history, predict
from app.schemas import HealthResponse

# Force logging to stdout with a level that always shows exceptions,
# regardless of whatever uvicorn's own logging config does per-platform.
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("medical_ai.main")

DEBUG_EXPOSE_ERRORS = os.environ.get("DEBUG_EXPOSE_ERRORS", "true").lower() == "true"

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "REST API for chest X-ray pneumonia detection, Grad-CAM explainability, "
        "and LLM-assisted draft reporting. For research/demo use only -- not a "
        "certified medical device and not for clinical diagnostic use."
    ),
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def catch_all_exception_handler(request: Request, exc: Exception):
    """
    Global safety net: catches ANY unhandled exception anywhere in the
    request lifecycle (route body, dependency resolution, response_model
    validation, etc.) -- not just what individual route try/excepts cover.
    This guarantees the real error is visible in the HTTP response even if
    terminal output is unreliable (seen on some Windows setups).

    TEMPORARY DEBUG measure -- set DEBUG_EXPOSE_ERRORS=false before any
    real deployment; returning stack traces to clients is an info leak.
    """
    tb = traceback.format_exc()
    print(f"\n{'='*70}\nUNHANDLED EXCEPTION on {request.method} {request.url.path}\n{tb}{'='*70}\n", flush=True)
    sys.stdout.flush()
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")

    if DEBUG_EXPOSE_ERRORS:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"{type(exc).__name__}: {exc}",
                "path": str(request.url.path),
                "traceback": tb,
            },
        )
    return JSONResponse(status_code=500, content={"error": "Internal server error."})


app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(history.router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def on_startup():
    init_db()
    get_inference_service()  # warm up / load model once at startup


@app.get("/", tags=["meta"])
def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "disclaimer": "Research/demo system only. Not a certified medical device.",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    service = get_inference_service()
    return HealthResponse(
        status="ok",
        model_loaded=service.model is not None,
        device=str(service.device),
    )