"""API chẩn đoán bệnh lá — FastAPI + ONNX Runtime (CPU-only, không CUDA).
Tải bundle từ Hugging Face (private repos) khi startup.
Luồng: validate -> leaf-gate -> loài -> [prototype].
Chạy dev: uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
"""
import io
import json
import os
import tempfile
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageOps
from huggingface_hub import snapshot_download

app = FastAPI(title="Plant Disease Diagnosis API", version="1.0")

# --- Ánh xạ loài -> repo_id trên Hugging Face ---
REPO_MAP = {
    "apple": "ruhua2305/apple_effv2s_onnx",
    "corn": "ruhua2305/corn_effv2s_onnx",
    "grape": "ruhua2305/grape_effv2s_onnx",
    "pepper_bell": "ruhua2305/pepper_bell_effv2s_onnx",
    "tomato": "ruhua2305/tomato_effv2s_onnx",
}
LEAF_GATE_REPO = "ruhua2305/leaf_gate_MobileNetV3Small"

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 12_000_000
ALLOWED_MAGIC = {b"\xff\xd8\xff": "jpg", b"\x89PNG\r\n\x1a\n": "png"}

_sessions, _classes, _preproc, _cal, _thr = {}, {}, {}, {}, {}
_leaf_sess, _leaf_T = None, 1.0


def _load_session(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _load_model_from_hf(repo_id: str, token: str = None):
    """Tải toàn bộ file của repo về thư mục tạm (có cache) và trả về đường dẫn."""
    # Dùng thư mục cache mặc định của huggingface-hub (~/.cache/huggingface/hub)
    # Để tránh lưu vào /tmp (mất khi restart), nhưng Render có thể dùng /tmp
    local_dir = snapshot_download(
        repo_id=repo_id,
        token=token,
        local_dir_use_symlinks=False,   # copy thực tế để ONNX load nhanh
        cache_dir=os.path.join(tempfile.gettempdir(), "hf_cache")  # có thể tùy chỉnh
    )
    return local_dir


@app.on_event("startup")
def _startup():
    """Nạp mọi model từ Hugging Face (private repos) khi startup."""
    global _leaf_sess, _leaf_T

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[WARN] HF_TOKEN không được set – không thể tải private repo!")

    # Tải các model loài
    for species, repo_id in REPO_MAP.items():
        try:
            print(f"📥 Đang tải model {species} từ {repo_id} ...")
            bundle_dir = _load_model_from_hf(repo_id, token=token)

            # Load ONNX
            model_path = os.path.join(bundle_dir, "model.onnx")
            if not os.path.exists(model_path):
                print(f"[ERROR] Không tìm thấy model.onnx trong {bundle_dir}")
                continue
            _sessions[species] = _load_session(model_path)

            # Load class_mapping
            cm_path = os.path.join(bundle_dir, "class_mapping.json")
            if os.path.exists(cm_path):
                with open(cm_path, encoding="utf-8") as f:
                    cm = json.load(f)
                    # Giả định cấu trúc: {"idx_to_class": {"0": "healthy", ...}}
                    if "idx_to_class" in cm:
                        idx_map = cm["idx_to_class"]
                        # Đảm bảo keys là int
                        _classes[species] = [idx_map[str(i)] for i in range(len(idx_map))]
                    else:
                        # Fallback: nếu là list trực tiếp
                        _classes[species] = cm if isinstance(cm, list) else list(cm.values())
            else:
                # Nếu không có, tạo mapping mặc định 0->"unknown"
                _classes[species] = ["unknown"]

            # Load preprocessing
            pp_path = os.path.join(bundle_dir, "preprocessing.json")
            if os.path.exists(pp_path):
                with open(pp_path, encoding="utf-8") as f:
                    _preproc[species] = json.load(f)
            else:
                # Giá trị mặc định (có thể không chính xác)
                _preproc[species] = {"img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

            # Load calibration (temperature)
            cal_path = os.path.join(bundle_dir, "calibration.json")
            if os.path.exists(cal_path):
                with open(cal_path, encoding="utf-8") as f:
                    _cal[species] = json.load(f)
            else:
                _cal[species] = {"temperature": 1.0}

            # Load thresholds
            thr_path = os.path.join(bundle_dir, "thresholds.json")
            if os.path.exists(thr_path):
                with open(thr_path, encoding="utf-8") as f:
                    _thr[species] = json.load(f)
            else:
                _thr[species] = {}

            print(f"✅ Đã tải xong model {species}")

        except Exception as e:
            print(f"[ERROR] Không thể tải model {species}: {e}")
            # Vẫn tiếp tục với các loài khác

    # Tải leaf_gate
    try:
        print(f"📥 Đang tải leaf_gate từ {LEAF_GATE_REPO} ...")
        lg_dir = _load_model_from_hf(LEAF_GATE_REPO, token=token)
        lg_model = os.path.join(lg_dir, "model.onnx")
        if os.path.exists(lg_model):
            _leaf_sess = _load_session(lg_model)
            # Nhiệt độ cho leaf_gate
            lg_cal = os.path.join(lg_dir, "calibration.json")
            if os.path.exists(lg_cal):
                _leaf_T = json.load(open(lg_cal, encoding="utf-8")).get("temperature", 1.0)
            print("✅ Đã tải xong leaf_gate")
        else:
            print("[WARN] leaf_gate/model.onnx không tồn tại")
    except Exception as e:
        print(f"[ERROR] Không thể tải leaf_gate: {e}")

    print(f"[startup] loài: {list(_sessions.keys())}; leaf_gate={'on' if _leaf_sess else 'off'}")


# ------------------ Các hàm hỗ trợ (giữ nguyên) ------------------
def validate_image_upload(raw: bytes) -> Image.Image:
    """Kiểm tra an toàn upload (Mục 7.6) — trả PIL RGB đã fix EXIF."""
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, detail={"code": "file_too_large", "message": "Ảnh vượt quá 8MB."})
    if not any(raw.startswith(sig) for sig in ALLOWED_MAGIC):
        raise HTTPException(415, detail={"code": "unsupported_media", "message": "Chỉ hỗ trợ JPEG/PNG hợp lệ."})
    img = Image.open(io.BytesIO(raw))
    if img.width * img.height > MAX_PIXELS:
        raise HTTPException(413, detail={"code": "too_many_pixels", "message": "Ảnh quá lớn để xử lý an toàn."})
    return ImageOps.exif_transpose(img).convert("RGB")


def _preprocess(img: Image.Image, pp: dict) -> np.ndarray:
    mean = np.array(pp["mean"], dtype=np.float32)
    std = np.array(pp["std"], dtype=np.float32)
    img = img.resize((pp["img_size"], pp["img_size"]))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _energy(logits, T=1.0):
    z = logits / T
    return float(-T * (z.max() + np.log(np.exp(z - z.max()).sum())))


# ------------------ Endpoints ------------------
@app.post("/predict")
async def predict(species: str, file: UploadFile = File(...)):
    """Chẩn đoán 1 ảnh cho 1 loài. Threshold LẤY TỪ bundle (đã calibration, Mục 5.4)."""
    if species not in _sessions:
        raise HTTPException(404, detail={"code": "species_unavailable",
                                         "message": f"Có: {list(_sessions.keys())}"})
    img = validate_image_upload(await file.read())
    x = _preprocess(img, _preproc[species])
    T = _cal[species].get("temperature", 1.0)
    thr = _thr[species]
    species_logits = _sessions[species].run(["logits"], {"input": x})[0]

    # (1) Leaf-gate + energy OOD
    if _leaf_sess is not None:
        lg = _leaf_sess.run(None, {"input": x})[0]
        p_leaf = float(_softmax(lg / _leaf_T)[0, 1])
        if p_leaf < thr.get("binary_leaf", 0.5):
            return {"status": "rejected", "reason": "not_a_leaf", "p_leaf": round(p_leaf, 4)}
        if _energy(species_logits, T) > thr.get("T_energy", 1e9):
            return {"status": "rejected", "reason": "unknown_or_ood",
                    "energy": round(_energy(species_logits, T), 4)}

    # (2) Chẩn đoán loài (đã calibration)
    probs = _softmax(species_logits / T)[0]
    order = probs.argsort()[::-1]
    classes = _classes[species]
    top1, top2 = probs[order[0]], (probs[order[1]] if len(order) > 1 else 0.0)
    if top1 < thr.get("T_low_confidence", 0.0):
        return {"status": "low_confidence", "confidence": round(float(top1), 4)}
    if (top1 - top2) < thr.get("T_margin", 0.0):
        return {"status": "ambiguous",
                "candidates": [classes[order[0]], classes[order[1]]],
                "confidence": round(float(top1), 4)}
    return {"status": "ok", "species": species, "prediction": classes[order[0]],
            "confidence": round(float(top1), 4),
            "top3": [{"label": classes[i], "prob": round(float(probs[i]), 4)} for i in order[:3]]}


@app.get("/health")
def health():
    """Liveness cho load balancer / Docker healthcheck."""
    return {"status": "healthy", "species": list(_sessions.keys())}


@app.get("/ready")
def ready():
    """Readiness: xác nhận bundle đã load đủ + leaf-gate sẵn sàng."""
    ok = len(_sessions) > 0
    return {"status": "ready" if ok else "not_ready", "species_loaded": list(_sessions.keys())}


# --- /api/* cho frontend PhytoScan + phục vụ tĩnh (Mục 7.9-7.10) ---
from fastapi.staticfiles import StaticFiles          # noqa: E402
from src.deploy.api_compat import router as api_router  # noqa: E402

app.include_router(api_router)
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")