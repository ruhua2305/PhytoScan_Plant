# 🌿 LeafScan — AI Plant Disease Detector

A full-stack web app that diagnoses plant diseases from leaf photos using your fine-tuned **EfficientNet-B2 ONNX model**.

**Stack:** FastAPI · ONNX Runtime · React · Vite

---

## Project Structure

```
plant-disease-app/
├── backend/
│   ├── main.py          # FastAPI app — CORS, /predict, /health endpoints
│   ├── inference.py     # ONNX model loader, image preprocessing, softmax
│   ├── requirements.txt
│   └── models/          # ← put plant_disease_model.onnx here
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js   # Dev proxy to backend
    └── src/
        ├── App.jsx           # State machine: idle → preview → loading → result
        ├── index.css         # Full design system (no Tailwind dependency)
        ├── components/
        │   ├── Header.jsx
        │   ├── UploadZone.jsx    # Drag-and-drop with specimen-tray aesthetics
        │   ├── LoadingSpinner.jsx # Scan-line animation during inference
        │   ├── DiagnosisCard.jsx  # Full result card with treatment steps
        │   └── HowItWorks.jsx
        └── utils/
            └── diseaseInfo.js    # Rich metadata for all 38 PlantVillage classes
```

---

## Quick Start

### 1. Set up the backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy your ONNX model
mkdir -p models
cp /path/to/plant_disease_model.onnx models/

# Start the API server
uvicorn main:app --reload --port 8000
```

> **GPU acceleration:** Replace `onnxruntime` with `onnxruntime-gpu` in requirements.txt
> if you have a CUDA-capable GPU.

Visit `http://localhost:8000/docs` to explore the auto-generated API documentation.

### 2. Set up the frontend

```bash
cd frontend

npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## API Reference

### `POST /predict`

Upload a leaf image and receive a disease prediction.

**Request:** `multipart/form-data` with a `file` field (JPEG, PNG, or WebP, max 10 MB).

**Response:**
```json
{
  "top_prediction": { "label": "Tomato___Early_blight", "confidence": 0.9421 },
  "top_5": [
    { "label": "Tomato___Early_blight",    "confidence": 0.9421 },
    { "label": "Tomato___Late_blight",     "confidence": 0.0312 },
    { "label": "Tomato___Target_Spot",     "confidence": 0.0148 },
    { "label": "Tomato___Bacterial_spot",  "confidence": 0.0083 },
    { "label": "Tomato___Septoria_leaf_spot", "confidence": 0.0036 }
  ],
  "inference_ms": 47.3,
  "plant": "Tomato",
  "disease": "Early Blight",
  "is_healthy": false
}
```

### `GET /health`

Returns `{ "status": "ok", "model_loaded": true }`.

---

## Supported Plants & Diseases (38 classes)

| Plant      | Diseases Detected |
|------------|-------------------|
| Apple      | Apple Scab, Black Rot, Cedar Rust, Healthy |
| Corn       | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape      | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Tomato     | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, TYLCV, Mosaic Virus, Healthy |
| Potato     | Early Blight, Late Blight, Healthy |
| Peach      | Bacterial Spot, Healthy |
| Bell Pepper | Bacterial Spot, Healthy |
| Cherry     | Powdery Mildew, Healthy |
| Strawberry | Leaf Scorch, Healthy |
| Orange     | Citrus Greening (HLB) |
| Blueberry  | Healthy |
| Raspberry  | Healthy |
| Soybean    | Healthy |
| Squash     | Powdery Mildew |

---

## Environment Variables

| Variable         | Default                    | Description |
|------------------|----------------------------|-------------|
| `VITE_API_URL`   | `http://localhost:8000`    | Backend URL for the frontend |

Create `frontend/.env.local`:
```
VITE_API_URL=http://localhost:8000
```

---

## Production Deployment

**Backend:** Deploy with `uvicorn main:app --host 0.0.0.0 --port 8000` behind Nginx, or containerize with Docker.

**Frontend:** Run `npm run build` and serve the `dist/` folder via Nginx, Vercel, or any static host.

Update CORS origins in `main.py` for your production domain.

---

## Disclaimer

For educational and research purposes only. Always consult a certified agronomist or plant pathologist for critical crop management decisions.
