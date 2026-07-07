"""API chẩn đoán bệnh lá — FastAPI + ONNX Runtime (CPU-only, không CUDA).
Tải bundle từ Hugging Face (private repos) khi startup.
Luồng: validate -> leaf-gate -> loài -> [prototype].
Chạy dev: uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
"""
import io
import json
import os
import tempfile
import glob
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
    local_dir = snapshot_download(
        repo_id=repo_id,
        token=token,
        local_dir_use_symlinks=False,
        cache_dir=os.path.join(tempfile.gettempdir(), "hf_cache")
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

            # ---- Tìm file .onnx (đệ quy) ----
            onnx_files = glob.glob(os.path.join(bundle_dir, "**", "*.onnx"), recursive=True)
            if not onnx_files:
                print(f"[ERROR] Không tìm thấy file .onnx nào trong {bundle_dir}")
                continue
            model_path = onnx_files[0]
            print(f"[INFO] Tìm thấy ONNX: {os.path.relpath(model_path, bundle_dir)}")
            _sessions[species] = _load_session(model_path)

            # ---- Tìm class_mapping.json (đệ quy) ----
            cm_files = glob.glob(os.path.join(bundle_dir, "**", "class_mapping.json"), recursive=True)
            if cm_files:
                with open(cm_files[0], encoding="utf-8") as f:
                    cm = json.load(f)
                    if "idx_to_class" in cm:
                        idx_map = cm["idx_to_class"]
                        _classes[species] = [idx_map[str(i)] for i in range(len(idx_map))]
                    else:
                        _classes[species] = cm if isinstance(cm, list) else list(cm.values())
            else:
                _classes[species] = ["unknown"]

            # ---- Tìm preprocessing.json (đệ quy) ----
            pp_files = glob.glob(os.path.join(bundle_dir, "**", "preprocessing.json"), recursive=True)
            if pp_files:
                with open(pp_files[0], encoding="utf-8") as f:
                    _preproc[species] = json.load(f)
            else:
                _preproc[species] = {"img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

            # ---- Tìm calibration.json (đệ quy) ----
            cal_files = glob.glob(os.path.join(bundle_dir, "**", "calibration.json"), recursive=True)
            if cal_files:
                with open(cal_files[0], encoding="utf-8") as f:
                    _cal[species] = json.load(f)
            else:
                _cal[species] = {"temperature": 1.0}

            # ---- Tìm thresholds.json (đệ quy) ----
            thr_files = glob.glob(os.path.join(bundle_dir, "**", "thresholds.json"), recursive=True)
            if thr_files:
                with open(thr_files[0], encoding="utf-8") as f:
                    _thr[species] = json.load(f)
            else:
                _thr[species] = {}

            print(f"✅ Đã tải xong model {species} (class_mapping: {len(_classes[species])} classes)")

        except Exception as e:
            print(f"[ERROR] Không thể tải model {species}: {e}")

    # ---- Tải leaf_gate ----
    try:
        print(f"📥 Đang tải leaf_gate từ {LEAF_GATE_REPO} ...")
        lg_dir = _load_model_from_hf(LEAF_GATE_REPO, token=token)
        lg_onnx = glob.glob(os.path.join(lg_dir, "**", "*.onnx"), recursive=True)
        if lg_onnx:
            _leaf_sess = _load_session(lg_onnx[0])
            print(f"[INFO] Tìm thấy leaf_gate ONNX: {os.path.relpath(lg_onnx[0], lg_dir)}")
            lg_cal = glob.glob(os.path.join(lg_dir, "**", "calibration.json"), recursive=True)
            if lg_cal:
                with open(lg_cal[0], encoding="utf-8") as f:
                    _leaf_T = json.load(f).get("temperature", 1.0)
            print("✅ Đã tải xong leaf_gate")
        else:
            print("[WARN] Không tìm thấy file .onnx cho leaf_gate")
    except Exception as e:
        print(f"[ERROR] Không thể tải leaf_gate: {e}")

    print(f"[startup] loài: {list(_sessions.keys())}; leaf_gate={'on' if _leaf_sess else 'off'}")


# ------------------ Các hàm hỗ trợ ------------------
def validate_image_upload(raw: bytes) -> Image.Image:
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
    if species not in _sessions:
        raise HTTPException(404, detail={"code": "species_unavailable",
                                         "message": f"Có: {list(_sessions.keys())}"})
    img = validate_image_upload(await file.read())
    x = _preprocess(img, _preproc[species])
    T = _cal[species].get("temperature", 1.0)
    thr = _thr[species]
    species_logits = _sessions[species].run(["logits"], {"input": x})[0]

    if _leaf_sess is not None:
        lg = _leaf_sess.run(None, {"input": x})[0]
        p_leaf = float(_softmax(lg / _leaf_T)[0, 1])
        if p_leaf < thr.get("binary_leaf", 0.5):
            return {"status": "rejected", "reason": "not_a_leaf", "p_leaf": round(p_leaf, 4)}
        if _energy(species_logits, T) > thr.get("T_energy", 1e9):
            return {"status": "rejected", "reason": "unknown_or_ood",
                    "energy": round(_energy(species_logits, T), 4)}

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
    return {"status": "healthy", "species": list(_sessions.keys())}


@app.get("/ready")
def ready():
    ok = len(_sessions) > 0
    return {"status": "ready" if ok else "not_ready", "species_loaded": list(_sessions.keys())}


# --- /api/* cho frontend PhytoScan + phục vụ tĩnh ---
from fastapi.staticfiles import StaticFiles
from src.deploy.api_compat import router as api_router

app.include_router(api_router)
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")