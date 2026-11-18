"""HTTP contract tests for /health and /predict."""

import io

import app.main as main
from PIL import Image
from tests.conftest import FakePredictor


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (20, 140, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_when_model_loaded(loaded_client):
    response = loaded_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_when_model_missing(unloaded_client):
    response = unloaded_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_predict_returns_schema(loaded_client):
    response = loaded_client.post(
        "/predict",
        files={"file": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "top_prediction",
        "top_5",
        "inference_ms",
        "plant",
        "disease",
        "is_healthy",
    }
    assert body["top_prediction"] == {
        "label": "Tomato___Early_blight",
        "confidence": 0.9,
    }
    assert len(body["top_5"]) == 5
    assert body["plant"] == "Tomato"
    assert body["disease"] == "Early Blight"
    assert body["is_healthy"] is False
    assert isinstance(body["inference_ms"], float)


def test_predict_healthy_label(loaded_client, monkeypatch):
    class HealthyPredictor(FakePredictor):
        def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
            return [{"label": "Tomato___healthy", "confidence": 0.99}]

    monkeypatch.setattr(main, "predictor", HealthyPredictor())
    response = loaded_client.post(
        "/predict",
        files={"file": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["plant"] == "Tomato"
    assert body["disease"] == "Healthy"
    assert body["is_healthy"] is True


def test_predict_rejects_non_image_bytes(loaded_client):
    response = loaded_client.post(
        "/predict",
        files={"file": ("leaf.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 400


def test_predict_rejects_unsupported_content_type(loaded_client):
    response = loaded_client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_rejects_oversize_file(loaded_client):
    payload = b"\xff\xd8" + (b"a" * (10 * 1024 * 1024 + 1))
    response = loaded_client.post(
        "/predict",
        files={"file": ("big.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 413


def test_predict_returns_503_when_model_missing(unloaded_client):
    response = unloaded_client.post(
        "/predict",
        files={"file": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503
