"""
LeafScan — Plant Disease Detection API
Built with FastAPI + ONNX Runtime
"""

import io
import time
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference import PlantDiseasePredictor

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LeafScan API",
    description="Plant disease detection via EfficientNet-B2 ONNX model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ─────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "models" / "plant_disease_model.onnx"
predictor: PlantDiseasePredictor | None = None


@app.on_event("startup")
async def load_model() -> None:
    global predictor
    if not MODEL_PATH.exists():
        log.warning(
            f"Model not found at {MODEL_PATH}. "
            "Copy plant_disease_model.onnx to backend/models/ and restart."
        )
        return
    predictor = PlantDiseasePredictor(str(MODEL_PATH))
    log.info(f"✅ Model loaded from {MODEL_PATH}")


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
    }


# ── Prediction endpoint ────────────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a leaf image and returns disease predictions.

    Returns:
        {
            "top_prediction": { "label": str, "confidence": float },
            "top_5": [ { "label": str, "confidence": float }, ... ],
            "inference_ms": float,
            "plant": str,
            "disease": str,
            "is_healthy": bool
        }
    """
    # Guard: model must be loaded
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check that plant_disease_model.onnx exists in backend/models/.",
        )

    # Guard: file type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a JPEG, PNG, or WebP image.",
        )

    # Read image bytes
    raw = await file.read()

    # Guard: file size
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.",
        )

    # Open with Pillow
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    # Run inference
    t0 = time.perf_counter()
    results = predictor.predict(image, top_k=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    log.info(
        f"Predicted: {results[0]['label']} ({results[0]['confidence']:.1%}) "
        f"in {elapsed_ms:.1f} ms"
    )

    # Parse top prediction into plant / disease parts
    top_label: str = results[0]["label"]
    plant, disease = _parse_label(top_label)

    return {
        "top_prediction": results[0],
        "top_5": results,
        "inference_ms": round(elapsed_ms, 2),
        "plant": plant,
        "disease": disease,
        "is_healthy": "healthy" in top_label.lower(),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────
def _parse_label(label: str) -> tuple[str, str]:
    """
    Converts 'Tomato___Early_blight' → ('Tomato', 'Early Blight')
    """
    parts = label.split("___", 1)
    plant = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip().title() if len(parts) > 1 else "Unknown"
    return plant, disease
