"""Fixtures that keep tests off the real ONNX model."""

from pathlib import Path

import app.main as main
import pytest
from fastapi.testclient import TestClient
from PIL import Image


class FakePredictor:
    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        ranked = [
            {"label": "Tomato___Early_blight", "confidence": 0.9},
            {"label": "Tomato___Late_blight", "confidence": 0.05},
            {"label": "Tomato___Leaf_Mold", "confidence": 0.02},
            {"label": "Tomato___Target_Spot", "confidence": 0.02},
            {"label": "Tomato___healthy", "confidence": 0.01},
        ]
        return ranked[:top_k]


@pytest.fixture
def loaded_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "PlantDiseasePredictor", lambda _path: FakePredictor())
    with TestClient(main.app) as client:
        yield client


@pytest.fixture
def unloaded_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "predictor", None)
    monkeypatch.setattr(main, "MODEL_PATH", Path("missing-model.onnx"))
    with TestClient(main.app) as client:
        yield client
