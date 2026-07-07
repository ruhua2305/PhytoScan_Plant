"""Router /api/* khớp hợp đồng frontend PhytoScan."""
import os
import re
import time
import httpx
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
router = APIRouter(prefix="/api")

# --- Cấu hình LLM ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "15"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# --- i18n ---
SPECIES_I18N = {
    "apple": {"en": "Apple", "vi": "Táo"},
    "corn": {"en": "Corn", "vi": "Ngô"},
    "grape": {"en": "Grape", "vi": "Nho"},
    "pepper_bell": {"en": "Bell Pepper", "vi": "Ớt chuông"},
    "tomato": {"en": "Tomato", "vi": "Cà chua"},
}

DISEASE_I18N = {
    "apple": {
        "apple_scab": {"en": "Apple Scab", "vi": "Ghẻ táo"},
        "cedar_apple_rust": {"en": "Cedar Apple Rust", "vi": "Gỉ sắt tuyết tùng táo"},
        "healthy": {"en": "Healthy", "vi": "Khỏe mạnh"},
    },
    "corn": {
        "cercospora": {"en": "Cercospora Leaf Spot", "vi": "Đốm lá Cercospora"},
        "common_rust": {"en": "Common Rust", "vi": "Gỉ sắt thường"},
        "northern_blight": {"en": "Northern Leaf Blight", "vi": "Cháy lá phương Bắc"},
        "healthy": {"en": "Healthy", "vi": "Khỏe mạnh"},
    },
    "grape": {
        "black_rot": {"en": "Black Rot", "vi": "Thối đen"},
        "healthy": {"en": "Healthy", "vi": "Khỏe mạnh"},
    },
    "pepper_bell": {
        "bacterial_spot": {"en": "Bacterial Spot", "vi": "Đốm khuẩn"},
        "healthy": {"en": "Healthy", "vi": "Khỏe mạnh"},
    },
    "tomato": {
        "bacterial_spot": {"en": "Bacterial Spot", "vi": "Đốm khuẩn"},
        "early_blight": {"en": "Early Blight", "vi": "Đốm sớm"},
        "late_blight": {"en": "Late Blight", "vi": "Mốc sương"},
        "leaf_mold": {"en": "Leaf Mold", "vi": "Mốc lá"},
        "septoria": {"en": "Septoria Leaf Spot", "vi": "Đốm lá Septoria"},
        "spider_mites": {"en": "Spider Mites", "vi": "Nhện đỏ"},
        "tylcv": {"en": "Yellow Leaf Curl Virus", "vi": "Virus xoăn vàng lá"},
        "mosaic_virus": {"en": "Mosaic Virus", "vi": "Virus khảm"},
        "healthy": {"en": "Healthy", "vi": "Khỏe mạnh"},
    },
}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

DISEASE_KEYWORDS = [
    (("scab",), {"en": "Apple Scab", "vi": "Ghẻ táo"}),
    (("cedar", "rust"), {"en": "Cedar Apple Rust", "vi": "Gỉ sắt tuyết tùng táo"}),
    (("cercospora",), {"en": "Cercospora Leaf Spot", "vi": "Đốm lá Cercospora"}),
    (("gray_leaf",), {"en": "Cercospora Leaf Spot", "vi": "Đốm lá Cercospora"}),
    (("common", "rust"), {"en": "Common Rust", "vi": "Gỉ sắt thường"}),
    (("northern",), {"en": "Northern Leaf Blight", "vi": "Cháy lá phương Bắc"}),
    (("black", "rot"), {"en": "Black Rot", "vi": "Thối đen"}),
    (("early", "blight"), {"en": "Early Blight", "vi": "Đốm sớm"}),
    (("late", "blight"), {"en": "Late Blight", "vi": "Mốc sương"}),
    (("bacterial",), {"en": "Bacterial Spot", "vi": "Đốm khuẩn"}),
    (("leaf", "mold"), {"en": "Leaf Mold", "vi": "Mốc lá"}),
    (("septoria",), {"en": "Septoria Leaf Spot", "vi": "Đốm lá Septoria"}),
    (("spider",), {"en": "Spider Mites", "vi": "Nhện đỏ"}),
    (("mite",), {"en": "Spider Mites", "vi": "Nhện đỏ"}),
    (("tylcv",), {"en": "Yellow Leaf Curl Virus", "vi": "Virus xoăn vàng lá"}),
    (("yellow", "curl"), {"en": "Yellow Leaf Curl Virus", "vi": "Virus xoăn vàng lá"}),
    (("mosaic",), {"en": "Mosaic Virus", "vi": "Virus khảm"}),
    (("healthy",), {"en": "Healthy", "vi": "Khỏe mạnh"}),
]

def translate_species(species: str, language: str) -> str:
    entry = SPECIES_I18N.get(species)
    if entry is None:
        return species.replace("_", " ").strip().title()
    return entry.get(language, entry["en"])

def translate_disease(species: str, class_name: str, language: str) -> str:
    norm = _norm(class_name)
    entry = DISEASE_I18N.get(species, {}).get(norm)
    if entry:
        return entry.get(language, entry["en"])
    for keywords, kw_entry in DISEASE_KEYWORDS:
        if all(kw in norm for kw in keywords):
            return kw_entry.get(language, kw_entry["en"])
    return class_name.replace("_", " ").strip().title()

def _select_species_disease_msg(language: str) -> str:
    return ("Vui lòng chọn loài và bệnh trước khi hỏi." if language == "vi"
            else "Please select both a species and a disease before asking.")

def _fallback_reply(species: str, disease: str, language: str) -> str:
    if language == "vi":
        return (f"[{species}/{disease}] Đây là thông tin tham khảo, không thay thế chuyên gia. "
                f"Hãy xác nhận triệu chứng trên nhiều lá và điều kiện canh tác trước khi xử lý.")
    return (f"[{species}/{disease}] This is reference information only, not a substitute for "
            f"an expert. Please confirm the symptoms across multiple leaves and your growing "
            f"conditions before taking any action.")

def _disclaimer_text(language: str) -> str:
    return ("Chỉ tham khảo, không thay thế chuyên gia nông nghiệp." if language == "vi"
            else "For reference only — not a substitute for an agricultural expert.")

def _confidence_level(p):
    return "High" if p >= 0.85 else ("Medium" if p >= 0.6 else "Low")

def _call_llm(system_prompt: str, user_prompt: str):
    if not LLM_API_KEY:
        return None
    try:
        resp = httpx.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "max_tokens": LLM_MAX_TOKENS,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[consult] LLM lỗi ({e}) -> fallback tĩnh")
        return None

# ========== ENDPOINTS ==========

@router.get("/species")
def species_list(language: str = "en"):
    from src.deploy.app import REPO_MAP
    keys = list(REPO_MAP.keys())
    return {
        "species": keys,
        "items": [{"value": k, "label": translate_species(k, language)} for k in keys],
    }

@router.get("/diseases")
def diseases(species: str, language: str = "en"):
    """Trả danh mục bệnh – luôn dùng DISEASE_I18N để dropdown hiển thị ngay."""
    disease_keys = list(DISEASE_I18N.get(species, {}).keys())
    if not disease_keys:
        disease_keys = ["healthy"]
    return {
        "species": species,
        "diseases": disease_keys,
        "items": [{"value": k, "label": translate_disease(species, k, language)} for k in disease_keys],
    }

@router.post("/predict")
async def api_predict(species: str = Form(...), file: UploadFile = File(...),
                      language: str = Form("en")):
    from src.deploy.app import predict
    t0 = time.time()
    try:
        result = await predict(species=species, file=file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if result.get("status") == "ok":
        preds = {d["label"]: d["prob"] for d in result["top3"]}
        return {
            "status": result["prediction"],
            "predictions": preds,
            "confidence_level": _confidence_level(result["confidence"]),
            "latency_ms": round((time.time() - t0) * 1000, 1)
        }
    return {
        "status": result.get("status"),
        "reason": result.get("reason"),
        "predictions": {},
        "confidence_level": "Low"
    }

@router.post("/consult")
async def consult(payload: dict):
    species = payload.get("species", "")
    disease = payload.get("disease", "")
    message = (payload.get("message", "") or "").strip()
    language = payload.get("language", "vi")
    if not species or not disease:
        return {"reply": _select_species_disease_msg(language)}

    msg_lower = message.lower()
    is_fungicide = any(kw in msg_lower for kw in ["fungicide", "thuốc", "diệt nấm", "spray", "phun", "trị", "xử lý"])
    is_prevention = any(kw in msg_lower for kw in ["prevent", "phòng", "tránh", "ngăn", "chặn"])
    is_symptom = any(kw in msg_lower for kw in ["symptom", "triệu chứng", "dấu hiệu", "biểu hiện", "vết"])
    is_spread = any(kw in msg_lower for kw in ["lây lan", "truyền", "lây", "contagious", "spread"])

    if is_fungicide:
        focus = (
            "Chỉ trả lời về thuốc đặc trị, bao gồm: tên hoạt chất, tên thương mại, "
            "liều lượng pha, cách phun và thời điểm phun tốt nhất. "
            "KHÔNG đề cập đến triệu chứng, nguyên nhân hay phòng ngừa trừ khi được hỏi cụ thể."
        )
    elif is_prevention:
        focus = (
            "Chỉ trả lời về biện pháp phòng ngừa: canh tác, vệ sinh đồng ruộng, "
            "chọn giống kháng, quản lý môi trường. "
            "KHÔNG đề cập đến triệu chứng hay thuốc trừ khi được hỏi cụ thể."
        )
    elif is_symptom:
        focus = (
            "Chỉ mô tả triệu chứng điển hình trên lá, thân, quả (màu sắc, hình dạng, vị trí, diễn biến). "
            "KHÔNG đề cập đến nguyên nhân, thuốc hay phòng ngừa trừ khi được hỏi cụ thể."
        )
    elif is_spread:
        focus = (
            "Chỉ trả lời về khả năng lây lan, con đường truyền bệnh, điều kiện lây lan. "
            "KHÔNG đề cập đến triệu chứng hay thuốc trừ khi được hỏi cụ thể."
        )
    else:
        focus = (
            "Trả lời ngắn gọn, đúng trọng tâm câu hỏi. Chỉ đưa thông tin cần thiết, "
            "không liệt kê tất cả các phần (triệu chứng, nguyên nhân, phòng, trị)."
        )

    system = (
        f"Bạn là chuyên gia nông học. Người dùng hỏi về bệnh '{disease}' trên cây {species}. "
        f"{focus} "
        "Luôn nhắc người dùng tham khảo chuyên gia địa phương và đọc kỹ hướng dẫn sử dụng thuốc. "
        "Không đưa ra chẩn đoán y tế hay khẳng định tuyệt đối."
    )

    if language == "vi":
        system += " Hãy trả lời bằng tiếng Việt, dễ hiểu, không quá dài dòng."
    else:
        system += " Answer in English, concise and clear."

    user = f"Câu hỏi: {message}"

    reply = _call_llm(system, user)
    if reply is None:
        reply = _fallback_reply(species, disease, language)
    return {"reply": reply, "disclaimer": _disclaimer_text(language)}