# Tầng phục vụ CPU-only (Mục 7.7). KHÔNG torch, KHÔNG CUDA -> image nhỏ.
FROM python:3.11-slim
WORKDIR /app

# Chỉ dependency cho INFERENCE
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Cần src/ là package: giữ __init__.py các cấp
COPY src/__init__.py ./src/__init__.py
COPY src/models/__init__.py src/models/energy_ood.py ./src/models/
COPY src/deploy/ ./src/deploy/
COPY model_bundles/ ./model_bundles/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.deploy.app:app", "--host", "0.0.0.0", "--port", "8080"]
