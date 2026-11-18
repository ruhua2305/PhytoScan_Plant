# LeafScan — Plant Disease Detection API

FastAPI service that classifies a leaf photo into one of 38 PlantVillage classes with an EfficientNet-B2 model exported to ONNX.

## Architecture

```
client
  |
  | POST /predict (JPEG, PNG, or WebP, max 10 MB)
  v
FastAPI (backend/app/main.py)
  |
  | Pillow decode + resize + ImageNet normalize
  v
ONNX Runtime (backend/app/inference.py)
  |
  v
backend/models/plant_disease_model.onnx
```

## Run with Docker

From the repository root:

```bash
docker compose up --build
```

- API: `http://localhost:8000` — `GET /health` is the container health check.
- Frontend: `http://localhost:8080`. The nginx container serves the React build and proxies `/predict` and `/health` to the API, so the browser talks to one origin.

## Run locally

Requires Python 3.11+.

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt ".[dev]"
uvicorn app.main:app --reload --port 8000
```

On Unix, activate with `source .venv/bin/activate`.

## Example request

```bash
curl -F "file=@leaf.jpg;type=image/jpeg" http://localhost:8000/predict
```

A successful response has this shape:

```json
{
  "top_prediction": { "label": "Tomato___Early_blight", "confidence": 0.9 },
  "top_5": [{ "label": "Tomato___Early_blight", "confidence": 0.9 }],
  "inference_ms": 0.0,
  "plant": "Tomato",
  "disease": "Early Blight",
  "is_healthy": false
}
```

`GET /health` returns `{ "status": "ok", "model_loaded": true }` once the model file is present.

## Metrics

These numbers have not been measured on this checkout. Do not treat the placeholders as results.

| Metric | Value |
| --- | --- |
| Top-1 accuracy | TODO |
| Top-5 accuracy | TODO |
| p50 inference latency | TODO |
| p95 inference latency | TODO |
| Evaluation dataset | TODO |

## Frontend checks

From `frontend/`:

```bash
npm ci
npm test            # Vitest unit tests
npx playwright test # E2E against the running compose stack (needs docker compose up)
```

## Checks

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)

CI runs `ruff check`, `pytest --cov`, `npm test` (Vitest), a Playwright end-to-end job against `docker compose up`, and `docker build` (no push) on every push and pull request. Replace `OWNER/REPO` after the repository has a GitHub remote.

## Disclaimer

For education and research only. Confirm crop decisions with a qualified agronomist or plant pathologist.
