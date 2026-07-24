# Advanced AI Medical Intelligence Platform

An end-to-end AI system for **chest X-ray pneumonia detection**, combining a
deep learning classifier, Grad-CAM explainability, LLM-generated draft
reports, a REST API, a prediction-history database, and a web UI.

> ⚠️ **Disclaimer:** This is a research/demo project built for a technical
> assessment. It is **not a certified medical device** and must never be used
> for actual clinical diagnosis. All AI outputs require review by a licensed
> physician.

---

## 1. Architecture

```
                ┌─────────────────┐
                │  Streamlit UI   |   (upload image, view results/history)
                └────────┬────────┘
                         │ HTTP
                ┌────────▼─────────┐
                │   FastAPI REST   │  /predict  /history  /health
                │       API        │
                └───┬─────────┬────┘
                    │         │
         ┌──────────▼───┐   ┌──▼──────────────┐
         │ DL Inference │   │  SQLite / DB    │
         │ (DenseNet121)│   │  prediction     │
         │ + Grad-CAM   │   │  history        │
         └──────┬───────┘   └─────────────────┘
                │
        ┌───────▼─────────┐
        │   GEMNINI AI    │  draft report generation
        │ (LLM reporting) │
        └─────────────────┘
```

**Flow:** user uploads a chest X-ray → CNN classifies NORMAL/PNEUMONIA →
Grad-CAM produces a heatmap explaining the decision → the prediction +
Grad-CAM summary is sent to an LLM to draft a structured, hedged report →
everything is persisted to the database → results are returned to the UI.

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch, torchvision (DenseNet121, transfer learning) |
| Explainable AI | Grad-CAM (custom implementation, hooks into last conv block) |
| LLM | GEMINI AI (`gemini-2.5-flash` by default, configurable) |
| API | FastAPI + Pydantic |
| Database | SQLAlchemy ORM, SQLite by default (swap `DATABASE_URL` for Postgres) |
| Frontend | Streamlit |
| Deployment | Docker + docker-compose |
| Testing | pytest + FastAPI TestClient |

## 3. Project Structure

```
medical-ai-platform/
├── app/
│   ├── main.py                 # FastAPI app entrypoint
|   ├── auth.py                 # Get authorized
|   ├── rate_limit.py           # Limit for execute or login in 1min is 5attemts
│   ├── config.py               # env-based settings
│   ├── database.py             # SQLAlchemy engine/session
│   ├── models_db.py            # ORM models (PredictionRecord)
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── ml/
│   │   ├── model.py             # DenseNet121 classifier definition
│   │   ├── preprocess.py        # image transforms
│   │   ├── gradcam.py           # Grad-CAM implementation
│   │   └── inference.py         # inference service (model + Grad-CAM)
│   ├── llm/
│   │   └── report_generator.py  # GENAI-based report drafting
│   └── routers/
│       ├── predict.py           # POST /predict
│       ├── history.py           # GET/DELETE /history
|       └── auth.py              # Get authorized
├── training/
│   ├── dataset.py               # dataset loader (ImageFolder-based)
│   ├── train.py                 # full training loop
│   └── evaluate.py              # test-set evaluation + plots
├── frontend/
│   └── streamlit_app.py         # web UI
├── tests/
│   └── test_api.py              # API tests
├── models/                      # trained checkpoint goes here (not in git)
├── static/gradcam_outputs/      # saved heatmap overlays
├── requirements.txt
├── .env
└── README.md
```

## 4. Dataset

Model targets the public **Chest X-Ray Images (Pneumonia)** dataset
(Kermany et al., Guangzhou Women and Children's Medical Center), available on
Kaggle:

```
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

Expected layout after download:

```
data/
  train/{NORMAL,PNEUMONIA}/*.jpeg
  val/{NORMAL,PNEUMONIA}/*.jpeg
  test/{NORMAL,PNEUMONIA}/*.jpeg
```

The dataset is imbalanced (~3x more PNEUMONIA than NORMAL images); `train.py`
computes class weights automatically and applies them in the loss function.

## 5. Setup

### 5.1 Local (Python)

```bash
git clone <your-repo-url>
cd medical-ai-platform
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

> No GPU, or just want a faster/lighter install? Use
> `pip install -r requirements-deploy.txt` instead — same functionality,
> CPU-only torch wheels (~200MB vs ~2.5GB).

Create your config file and fill in the required values:

```bash
cp .env.example .env
```

In `.env`, set:
- `GEMINI_API_KEY` — your Google Gemini API key (from https://aistudio.google.com/apikey)
- `JWT_SECRET_KEY` — a fixed secret for signing login tokens. Generate one with:
```bash
  python -c "import secrets; print(secrets.token_hex(32))"
```
  Without this, the app auto-generates a random secret every time it
  restarts, which invalidates every existing login token on each restart —
  fine for a quick local test, but set a fixed value for anything beyond that.

### 5.2 Train the model

```bash
# after downloading the Kaggle dataset into ./data/
python training/train.py --data-root data --epochs 15 --batch-size 32
```

This saves the best checkpoint (by validation AUC) to
`models/chest_xray_densenet121.pt`. Evaluate it with:

```bash
python training/evaluate.py --checkpoint models/chest_xray_densenet121.pt --data-root data
```

This produces a confusion matrix and ROC curve under `docs/eval_plots/`.

If no checkpoint is present, the API still runs (falls back to the
ImageNet-pretrained backbone with an untrained head) so you can verify the
full pipeline end-to-end before/while training completes. Predictions from
the untrained head are **not meaningful** — they exist purely to confirm
the API/DB/Grad-CAM/LLM wiring works.

### 5.3 Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger docs: `http://localhost:8000/docs`
- Health check (no auth required): `http://localhost:8000/health`

All other endpoints require a login. Easiest way to test from Swagger UI:
1. Expand `POST /auth/register`, click **Try it out**, submit a username/password to create an account.
2. Click the **Authorize** button (top right of the Swagger page), log in with those same credentials.
3. All subsequent requests from the Swagger UI will now include your token automatically.

### 5.4 Run the frontend

```bash
streamlit run frontend/streamlit_app.py
```

Opens at `http://localhost:8501`. Set `API_BASE_URL` env var if the API is
hosted elsewhere. On first load, use the **Create Account** tab to
register, then log in — the app won't show the upload/history screens
until you're authenticated.

### 5.5 Docker (recommended for deployment)

```bash
cp .env.example .env   # fill in OPENAI_API_KEY
docker compose up --build
```

- API: `http://localhost:8000`
- Frontend: `http://localhost:8501`

## 6. API Reference

Except for `/auth/register`, `/auth/login`, and `/health`, all endpoints
require a valid Bearer token. Obtain one via `/auth/login`, then include
it as `Authorization: Bearer <token>` on every request.

### `POST /auth/register`
Creates a new user account. Body: `{"username": "...", "password": "..."}`
(password minimum 8 characters). Returns the created user (no password/hash).

### `POST /auth/login`
Authenticates and returns a JWT access token (`Bearer` type, valid for
`ACCESS_TOKEN_EXPIRE_MINUTES`, default 60). Accepts standard OAuth2 form
fields (`username`, `password`) so it works directly with Swagger UI's
"Authorize" button. **Rate limited to 5 requests/minute per IP** to slow
brute-force attempts.

### `POST /predict` 🔒
Upload an X-ray image (JPEG/PNG, ≤10MB). Returns prediction, confidence,
class probabilities, base64 Grad-CAM overlay, and an LLM-generated draft
report. Persists the result to the database, tagged to the authenticated
user. **Rate limited to 10 requests/minute per IP.**

### `GET /history?limit=20&offset=0&predicted_class=PNEUMONIA` 🔒
Lists the authenticated user's own past predictions (paginated, optionally
filtered by class). Users can never see another user's history.

### `GET /history/{id}` 🔒
Full detail for a single prediction record owned by the authenticated user,
including the stored LLM report. Returns 404 (not 403) if the record
belongs to someone else, to avoid confirming which record IDs exist.

### `DELETE /history/{id}` 🔒
Deletes a prediction record owned by the authenticated user.

### `GET /health`
Liveness/readiness check — confirms the model is loaded and reports the
compute device in use. No auth required.

🔒 = requires `Authorization: Bearer <token>` header.

Full interactive schema, including the ability to log in and test protected
routes directly, is available at `/docs` (Swagger UI → click **Authorize**).

## 7. Explainable AI

Grad-CAM (Selvaraju et al., 2017) is implemented from scratch in
`app/ml/gradcam.py`. It hooks into the final normalization layer of the
DenseNet121 feature extractor, computes gradients of the predicted class
score with respect to those feature maps, and produces a localization
heatmap highlighting the image regions most responsible for the prediction.
The heatmap is overlaid on the original X-ray and returned alongside the
numeric prediction so a clinician can visually sanity-check the model's
reasoning.

## 8. LLM Integration

`app/llm/report_generator.py` sends the structured prediction (class,
confidence, full probability distribution, Grad-CAM region description) to
the GEMINI AI API with a system prompt that constrains it to produce a
hedged, structured draft report (Impression / Observations /
Confidence & Limitations / Suggested Next Steps) — explicitly avoiding
definitive diagnostic language. Every report ends with a disclaimer stating
it is not a medical diagnosis. If no API key is configured or the call
fails, a deterministic template report is used instead so the system
degrades gracefully rather than failing the request.

## 9. Testing

```bash
pytest tests/ -v
```

 **Note:** `tests/test_api.py` currently tests the pre-authentication
 version of the API. It needs updating to register/log in a test user and
 attach the resulting token before calling `/predict` and `/history` —
 otherwise those tests will now fail with 401s. [Update before final
 submission / mark as a known gap.]

## 10. Known Limitations / Future Work

- Model was not trained in this sandbox (no GPU/network access available
  during development) — `train.py` is ready to run against the Kaggle
  dataset and should be executed before any real evaluation of accuracy.
- Grad-CAM's textual "region description" fed to the LLM is currently a
  simple heuristic; a production version should compute the heatmap's
  actual centroid/quadrant and describe that directly.
- SQLite is fine for a demo; switch `DATABASE_URL` to Postgres for
  concurrent production workloads.
- No DICOM support — currently JPEG/PNG only.

## 11. License / Attribution

Built as a technical assessment project. Dataset: Kermany et al., "Chest
X-Ray Images (Pneumonia)" (Guangzhou Women and Children's Medical Center),
via Kaggle. DenseNet121 architecture: Huang et al., 2017. Grad-CAM:
Selvaraju et al., 2017.
