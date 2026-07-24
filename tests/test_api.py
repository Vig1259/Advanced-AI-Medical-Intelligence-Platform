"""
Basic API tests using FastAPI's TestClient.

Run: pytest tests/ -v

Note: /predict tests use a tiny synthetic image and an untrained/ImageNet
backbone if no real checkpoint is present -- they verify the pipeline wiring
(status codes, response shape, DB persistence), not model accuracy.
"""
import io
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_medical_ai.db")

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "name" in resp.json()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "device" in body


def test_predict_rejects_bad_content_type():
    resp = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert resp.status_code == 400


def test_predict_and_history_roundtrip():
    img_bytes = make_test_image_bytes()
    resp = client.post("/predict", files={"file": ("test.png", img_bytes, "image/png")})
    assert resp.status_code == 200
    body = resp.json()

    assert body["predicted_class"] in {"NORMAL", "PNEUMONIA"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert "gradcam_image_base64" in body
    assert "llm_report" in body

    record_id = body["id"]

    hist_resp = client.get("/history")
    assert hist_resp.status_code == 200
    ids = [r["id"] for r in hist_resp.json()]
    assert record_id in ids

    detail_resp = client.get(f"/history/{record_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["id"] == record_id


def test_history_detail_404_for_unknown_id():
    resp = client.get("/history/does-not-exist")
    assert resp.status_code == 404


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    if os.path.exists("test_medical_ai.db"):
        os.remove("test_medical_ai.db")
