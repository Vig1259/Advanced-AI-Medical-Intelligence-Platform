"""Centralized app configuration, loaded from environment variables."""
import os
import sys

from dotenv import find_dotenv, load_dotenv

# find_dotenv() searches the current working directory AND walks upward
# through parent directories looking for a .env file -- more forgiving than
# load_dotenv()'s bare default, which only checks the exact current working
# directory. This matters a lot if uvicorn is ever launched from a different
# folder than where .env actually lives (e.g. a duplicated/nested project
# folder), which otherwise fails completely silently.
_dotenv_path = find_dotenv(usecwd=True)

if _dotenv_path:
    load_dotenv(_dotenv_path, override=True)
    print(f"[config] Loaded .env from: {_dotenv_path}", flush=True)
else:
    print(
        "[config] WARNING: No .env file found (searched from current working "
        f"directory: {os.getcwd()}). All env-based settings (GEMINI_API_KEY, etc.) "
        "will be missing unless set some other way (e.g. OS-level env vars, "
        "docker-compose environment section).",
        file=sys.stderr,
        flush=True,
    )


class Settings:
    APP_NAME: str = "Advanced AI Medical Intelligence Platform"
    APP_VERSION: str = "1.0.0"

    DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./medical_ai.db")
    MODEL_CHECKPOINT_PATH: str = os.environ.get("MODEL_CHECKPOINT_PATH", "models/chest_xray_densenet121.pt")
    GRADCAM_OUTPUT_DIR: str = os.environ.get("GRADCAM_OUTPUT_DIR", "static/gradcam_outputs")

    GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    CORS_ORIGINS: list[str] = os.environ.get("CORS_ORIGINS", "*").split(",")

    # --- Auth ---
    JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


settings = Settings()

if not settings.GEMINI_API_KEY:
    print(
        "[config] WARNING: GEMINI_API_KEY is not set. The LLM report generator will "
        "use the deterministic fallback template for every request until this is fixed.",
        file=sys.stderr,
        flush=True,
    )
else:
    print(f"[config] GEMINI_API_KEY loaded (ends in ...{settings.GEMINI_API_KEY[-4:]}).", flush=True)

if not settings.JWT_SECRET_KEY:
    import secrets as _secrets
    settings.JWT_SECRET_KEY = _secrets.token_hex(32)
    print(
        "[config] WARNING: JWT_SECRET_KEY not set in .env -- generated a random "
        "one for THIS PROCESS ONLY. All existing login tokens will be invalidated "
        "every time the server restarts, and this is NOT safe for production/multi-"
        "instance deployment. Set a fixed JWT_SECRET_KEY in .env (e.g. output of "
        "`python -c \"import secrets; print(secrets.token_hex(32))\"`).",
        file=sys.stderr,
        flush=True,
    )
