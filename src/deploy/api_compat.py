"""Router /api/* khớp hợp đồng frontend PhytoScan. Bọc quanh inference ở app.py.
Đã include vào app trong app.py.
"""
import os
import re
import time
import httpx
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

# SỬA: import thêm REPO_MAP và get_model
from src.deploy.app import REPO_MAP, get_model, _classes, predict

from dotenv import load_dotenv
load_dotenv()  # tự động đọc .env
router = APIRouter(prefix="/api")

# --- Cấu hình LLM qua ENV (KHÔNG hard-code khoá) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "15"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))


# --- i18n cho dropdown loài & bệnh (Mục 7.9-7.10 — frontend không hard-code,
# nhưng nhãn hiển thị cần dịch theo `language`; khoá nội bộ species/class_name
# vẫn giữ nguyên để tra cứu class_mapping/bundle không đổi). ---
SPECIES_I18N = {
    "apple": {"en": "Apple", "vi": "Táo"},
    "corn": {"en": "Corn", "vi": "Ngô"},
    "grape": {"en": "Grape", "vi": "Nho"},
    "pepper_bell": {"en": "Bell Pepper", "vi": "Ớt chuông"},
    "tomato": {"en": "Tomato", "vi": "Cà chua"},
}

# Khoá đã chuẩn hoá (lowercase, snake_case) khớp taxonomy Appendix A —
# dùng _norm() để không phụ thuộc cách viết hoa/thường trong class_mapping.json thực tế.
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
    """Chuẩn hoá tên lớp về chữ thường, chỉ giữ chữ/số, phần còn lại thành '_'.
    Chịu được mọi kiểu viết: 'Apple_scab', 'apple-scab', 'Apple___Apple_scab' (PlantVillage),
    'Corn_(maize)___Northern_Leaf_Blight', v.v."""
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


# Khớp theo TỪ KHOÁ (all-of) thay vì khớp tuyệt đối — chịu được tiền tố loài,
# định dạng PlantVillage kiểu "Loài___Bệnh", hoặc cách viết hoa/gạch dưới khác với
# những gì bảng DISEASE_I18N giả định. Đây là lớp fallback thứ 2, áp dụng SAU khi
# tra cứu tuyệt đối theo DISEASE_I18N thất bại.
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
    """Tên loài hiển thị theo `language`. Loài lạ (chưa có trong bảng) -> humanize khoá gốc."""
    entry = SPECIES_I18N.get(species)
    if entry is None:
        return species.replace("_", " ").strip().title()
    return entry.get(language, entry["en"])


def translate_disease(species: str, class_name: str, language: str) -> str:
    """Tên lớp bệnh hiển thị theo `language`. Thứ tự tra cứu:
    1) Khớp tuyệt đối theo DISEASE_I18N[species] (tên lớp đúng taxonomy Appendix A).
    2) Khớp theo từ khoá DISEASE_KEYWORDS (chịu được tiền tố loài, định dạng kiểu
       PlantVillage 'Loài___Bệnh', viết hoa/gạch dưới khác nhau).
    3) Không nhận diện được -> humanize tên gốc. Lưu ý: nhánh (3) hiển thị GIỐNG NHAU
       ở cả 2 ngôn ngữ (không có bản dịch) — nếu gặp trường hợp này thường xuyên,
       nghĩa là cần bổ sung entry/keyword mới ở trên cho đúng với dữ liệu thật."""
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
    """Gọi LLM ngoài (chat completions). Trả text hoặc None nếu key trống / lỗi.
    An toàn: chỉ text đi ra ngoài; timeout ngắn; nuốt lỗi để không sập API chính."""
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


# ==================== SỬA ENDPOINTS ====================

@router.get("/species")
def species_list(language: str = "en"):
    """Trả danh sách loài đang phục vụ (frontend đổ vào dropdown).
    - species: khoá nội bộ, GIỮ NGUYÊN để tương thích ngược.
    - items: [{value, label}] — value = khoá nội bộ (dùng để gọi /api/diseases),
      label = tên hiển thị đã dịch theo `language`."""
    # Lấy danh sách loài từ REPO_MAP thay vì _sessions (đang rỗng do lazy loading)
    keys = list(REPO_MAP.keys())
    return {
        "species": keys,
        "items": [{"value": k, "label": translate_species(k, language)} for k in keys],
    }


@router.get("/diseases")
def diseases(species: str, language: str = "en"):
    """Trả danh mục lớp bệnh của 1 loài (cho dropdown tư vấn). Lazy load model nếu chưa có."""
    # Gọi get_model để lazy load và điền _classes
    try:
        get_model(species)
    except Exception as e:
        # Nếu không tải được, trả về danh sách rỗng
        return {"species": species, "diseases": [], "items": []}
    
    raw = _classes.get(species, [])
    return {
        "species": species,
        "diseases": raw,
        "items": [{"value": c, "label": translate_disease(species, c, language)} for c in raw],
    }


# ==================== CÁC ENDPOINTS KHÁC GIỮ NGUYÊN ====================

@router.post("/predict")
async def api_predict(species: str = Form(...), file: UploadFile = File(...),
                      language: str = Form("en")):
    """Bọc core.predict -> shape frontend: {status, predictions:{disease:conf}, confidence_level}."""
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
    """Tư vấn chatbot bằng LLM ngoài. Chỉ gửi TEXT (loài/bệnh/câu hỏi),
    TUYỆT ĐỐI không gửi ảnh/PII. Fallback tĩnh nếu key trống / lỗi mạng."""
    species = payload.get("species", "")
    disease = payload.get("disease", "")
    message = (payload.get("message", "") or "").strip()
    language = payload.get("language", "vi")
    if not species or not disease:
        return {"reply": _select_species_disease_msg(language)}

    # ---- Phân tích câu hỏi để xác định trọng tâm ----
    msg_lower = message.lower()
    is_fungicide = any(kw in msg_lower for kw in ["fungicide", "thuốc", "diệt nấm", "spray", "phun", "trị", "xử lý"])
    is_prevention = any(kw in msg_lower for kw in ["prevent", "phòng", "tránh", "ngăn", "chặn"])
    is_symptom = any(kw in msg_lower for kw in ["symptom", "triệu chứng", "dấu hiệu", "biểu hiện", "vết"])
    is_spread = any(kw in msg_lower for kw in ["lây lan", "truyền", "lây", "contagious", "spread"])

    # ---- Xây dựng system prompt theo trọng tâm ----
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

    # Thêm yêu cầu ngôn ngữ
    if language == "vi":
        system += " Hãy trả lời bằng tiếng Việt, dễ hiểu, không quá dài dòng."
    else:
        system += " Answer in English, concise and clear."

    user = f"Câu hỏi: {message}"

    reply = _call_llm(system, user)
    if reply is None:
        reply = _fallback_reply(species, disease, language)
    return {"reply": reply, "disclaimer": _disclaimer_text(language)}