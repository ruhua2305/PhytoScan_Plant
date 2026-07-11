# -*- coding: utf-8 -*-
"""
Gom dữ liệu field & lab từ nhiều dataset bệnh cây,
tách riêng ảnh quả (nếu có). Đầu ra: ONFIELD_LEAF_DETECTED và LAB_LEAF_DETECTED.

PHÂN LOẠI DATASET:
- ONFIELD: field tuyệt đối → copy thẳng vào ONFIELD_LEAF_DETECTED.
- LAB_SUSPEN: nghi lẫn lab → dùng CLIP lab/field, field → ONFIELD, lab → LAB.
- FRUIT_SUSPEN: nghi lẫn quả → CLIP leaf/fruit trước, nếu leaf thì tiếp tục
  CLIP lab/field; nếu fruit thì bỏ qua (có thể lưu riêng nếu cần).

Các prompt CLIP được giữ nguyên từ phiên bản trước.
"""

import shutil
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import open_clip
except ImportError:
    open_clip = None
    print("Cảnh báo: open_clip_torch chưa được cài. Phân loại bằng CLIP sẽ không hoạt động.")

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

# ======================================================================
# CONFIG
# ======================================================================
DATA_ROOT = Path(r"D:\DAT301m_Project\data")
PV_ROOT = DATA_ROOT / "PlantVillage"

# Thư mục đầu ra
ONFIELD_DIR = DATA_ROOT / "ONFIELD_LEAF_DETECTED"
LAB_DIR = DATA_ROOT / "LAB_LEAF_DETECTED"
# FRUIT_DIR = DATA_ROOT / "FRUIT_DETECTED"   # bỏ comment nếu muốn giữ ảnh quả

CROP_RESIZE_SIZE = (224, 224)
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# ---------- PHÂN NHÓM DATASET (đã cập nhật) ----------
ONFIELD_DATASETS = {
    "PlantDoc",
    "FieldPlant",
    "Plant Pathology 2020",
    "Plant Pathology 2021",
}

LAB_SUSPEN_DATASETS = {
    "Tomato-Village-main",
    "Niphad Grape Leaf Disease Dataset (NGLD)",
    "Master Plant Disease Dataset",
    "Plant Disease Classification Merged Dataset",
    "TomatoDataset",                # chuyển từ ONFIELD sang LAB_SUSPEN
}

FRUIT_SUSPEN_DATASETS = {
    "plantseg",                     # chuyển từ LAB_SUSPEN sang FRUIT_SUSPEN
}

# ======================================================================
# TÊN LOÀI CHUẨN (PlantVillage)
# ======================================================================
SP_APPLE = "Apple"
SP_CHERRY = "Cherry_(including_sour)"
SP_CORN = "Corn_(maize)"
SP_GRAPE = "Grape"
SP_PEACH = "Peach"
SP_PEPPER = "Pepper,_bell"
SP_POTATO = "Potato"
SP_STRAWBERRY = "Strawberry"
SP_TOMATO = "Tomato"

LBL_CORN_GRAY_LEAF_SPOT = f"{SP_CORN}___Cercospora_leaf_spot Gray_leaf_spot"
LBL_TOMATO_SPIDER_MITES = f"{SP_TOMATO}___Spider_mites Two-spotted_spider_mite"

# ======================================================================
# HÀM TIỆN ÍCH
# ======================================================================
def get_pv_classes():
    classes = set()
    for split in ("train", "val"):
        split_dir = PV_ROOT / split
        if not split_dir.exists():
            continue
        for d in split_dir.iterdir():
            if d.is_dir():
                classes.add(d.name)
    return classes

def get_candidate_species(pv_classes):
    has_healthy = defaultdict(bool)
    has_disease = defaultdict(bool)
    for cls in pv_classes:
        if "___" not in cls:
            continue
        species, disease = cls.split("___", 1)
        if disease.strip().lower() == "healthy":
            has_healthy[species] = True
        else:
            has_disease[species] = True
    return sorted(sp for sp in has_healthy if has_healthy[sp] and has_disease.get(sp))

def safe_copy(src: Path, dest_dir: Path, dest_filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / dest_filename
    if dest_path.exists():
        stem, suffix = dest_path.stem, dest_path.suffix
        i = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{stem}_{i}{suffix}"
            i += 1
    shutil.copy2(src, dest_path)
    return dest_path

def add_to_onfield(src_path: Path, pv_label: str, source_tag: str, stats: Counter):
    species = pv_label.split("___")[0]
    dest_dir = ONFIELD_DIR / species / pv_label
    filename = f"{source_tag}_{src_path.name}"
    safe_copy(src_path, dest_dir, filename)
    stats[pv_label] += 1

def add_to_lab(src_path: Path, pv_label: str, source_tag: str, stats: Counter):
    species = pv_label.split("___")[0]
    dest_dir = LAB_DIR / species / pv_label
    filename = f"{source_tag}_{src_path.name}"
    safe_copy(src_path, dest_dir, filename)
    stats[pv_label] += 1

def add_cropped_to_onfield(img: Image.Image, pv_label: str, source_tag: str, idx: int, stats: Counter):
    species = pv_label.split("___")[0]
    dest_dir = ONFIELD_DIR / species / pv_label
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{source_tag}_{idx:06d}.jpg"
    dest_path = dest_dir / filename
    img = img.convert("RGB").resize(CROP_RESIZE_SIZE, RESAMPLE)
    img.save(dest_path, "JPEG", quality=95)
    stats[pv_label] += 1

def find_image(folder: Path, stem: str):
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def print_unmapped(dataset_name: str, counter: Counter, info_only: bool = False):
    if not counter:
        return
    tag = "[BỎ QUA - không thuộc candidate species]" if info_only else "[CẢNH BÁO - KHÔNG MAP ĐƯỢC]"
    print(f"\n  {tag} {dataset_name}: {len(counter)} lớp/nhãn:")
    for name, cnt in counter.most_common():
        print(f"      - {name!r}: {cnt} ảnh")

# ======================================================================
# CLIP ZERO-SHOT (giữ nguyên prompt lab/field, thêm fruit)
# ======================================================================
LAB_PROMPTS = [
    "a photo of a single plant leaf placed on a plain, solid-colored background",
    "a close-up macro photograph of a leaf against a uniform white or gray studio background",
    "a laboratory specimen photo of a detached leaf on a flat, plain surface, no soil or other plants visible",
    "a plant disease dataset image with a clean, monochromatic background and studio lighting",
    "a top-down photo of an isolated leaf on a table, with no natural outdoor scenery in the background",
]
FIELD_PROMPTS = [
    "a photo of a plant growing outdoors in a farm field",
    "a photo of crop leaves still attached to the plant, surrounded by soil, other leaves, and natural vegetation",
    "an outdoor agricultural photo with a complex, cluttered natural background",
    "a field photograph of a crop plant with visible ground, weeds, sky, or other plants in the background",
    "a photo taken in a garden or farm showing foliage among natural surroundings, not a studio background",
]
FRUIT_PROMPTS = [
    "a photo of a fruit",
    "a close-up of a fruit on a plain background in a lab",
    "a single fruit specimen on a neutral background",
    "a fruit with no leaves, shot indoors against a solid color surface",
]

CLIP_MODEL = None
CLIP_PREPROCESS = None
LAB_CENTROID = None
FIELD_CENTROID = None
FRUIT_CENTROID = None
LOGIT_SCALE = 100.0

def init_clip(device: str = "cpu"):
    global CLIP_MODEL, CLIP_PREPROCESS, LAB_CENTROID, FIELD_CENTROID, FRUIT_CENTROID, LOGIT_SCALE
    if open_clip is None:
        raise RuntimeError("Thư viện open_clip_torch chưa được cài. Hãy chạy: pip install open_clip_torch")

    CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    CLIP_MODEL.eval()
    CLIP_MODEL.to(device)

    def build_centroid(prompts):
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            feats = CLIP_MODEL.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        centro = feats.mean(dim=0)
        centro = centro / centro.norm()
        return centro

    LAB_CENTROID = build_centroid(LAB_PROMPTS)
    FIELD_CENTROID = build_centroid(FIELD_PROMPTS)
    FRUIT_CENTROID = build_centroid(FRUIT_PROMPTS)
    LOGIT_SCALE = CLIP_MODEL.logit_scale.exp().item()

def classify_image_type(img_path: Path) -> str:
    """
    Trả về 'field', 'lab' hoặc 'fruit' dựa trên CLIP zero-shot.
    """
    if CLIP_MODEL is None:
        raise RuntimeError("CLIP chưa được khởi tạo. Gọi init_clip() trước.")
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = CLIP_PREPROCESS(img).unsqueeze(0).to(CLIP_MODEL.device)
        with torch.no_grad():
            img_feat = CLIP_MODEL.encode_image(img_tensor).float()
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        centroids = torch.stack([LAB_CENTROID, FIELD_CENTROID, FRUIT_CENTROID], dim=0)
        logits = LOGIT_SCALE * img_feat @ centroids.T
        probs = logits.softmax(dim=-1)

        lab_prob, field_prob, fruit_prob = probs[0, 0].item(), probs[0, 1].item(), probs[0, 2].item()
        # Có thể in debug nếu cần
        # print(f"{img_path.name}: lab={lab_prob:.2f}, field={field_prob:.2f}, fruit={fruit_prob:.2f}")

        if field_prob >= lab_prob and field_prob >= fruit_prob:
            return "field"
        elif fruit_prob >= lab_prob:
            return "fruit"
        else:
            return "lab"
    except Exception:
        return "lab"  # ảnh lỗi -> coi là lab để loại bỏ

# ======================================================================
# MAPPING TỪ-KHÓA
# ======================================================================
SPECIES_NAME_MAP = {
    "apple": SP_APPLE, "cherry": SP_CHERRY, "sour cherry": SP_CHERRY,
    "corn": SP_CORN, "maize": SP_CORN, "grape": SP_GRAPE, "grapes": SP_GRAPE,
    "peach": SP_PEACH, "pepper": SP_PEPPER, "bell pepper": SP_PEPPER,
    "capsicum": SP_PEPPER, "potato": SP_POTATO, "strawberry": SP_STRAWBERRY,
    "tomato": SP_TOMATO,
}

DISEASE_KEYWORDS = {
    SP_APPLE: [
        (["healthy"], "healthy"),
        (["scab"], "Apple_scab"),
        (["black rot"], "Black_rot"),
        (["cedar apple rust", "rust"], "Cedar_apple_rust"),
    ],
    SP_CHERRY: [
        (["healthy"], "healthy"),
        (["powdery"], "Powdery_mildew"),
    ],
    SP_CORN: [
        (["healthy"], "healthy"),
        (["common rust"], "Common_rust_"),
        (["gray leaf spot", "cercospora"], LBL_CORN_GRAY_LEAF_SPOT.split("___", 1)[1]),
        (["northern leaf blight", "leaf blight", "blight"], "Northern_Leaf_Blight"),
    ],
    SP_GRAPE: [
        (["healthy"], "healthy"),
        (["black rot"], "Black_rot"),
        (["esca", "black measles"], "Esca_(Black_Measles)"),
        (["isariopsis", "leaf blight"], "Leaf_blight_(Isariopsis_Leaf_Spot)"),
    ],
    SP_PEACH: [
        (["healthy"], "healthy"),
        (["bacterial spot"], "Bacterial_spot"),
    ],
    SP_PEPPER: [
        (["healthy"], "healthy"),
        (["bacterial spot"], "Bacterial_spot"),
    ],
    SP_POTATO: [
        (["healthy"], "healthy"),
        (["early blight"], "Early_blight"),
        (["late blight"], "Late_blight"),
    ],
    SP_STRAWBERRY: [
        (["healthy"], "healthy"),
        (["leaf scorch", "scorch"], "Leaf_scorch"),
    ],
    SP_TOMATO: [
        (["healthy"], "healthy"),
        (["bacterial spot"], "Bacterial_spot"),
        (["early blight"], "Early_blight"),
        (["late blight"], "Late_blight"),
        (["leaf mold", "mold"], "Leaf_Mold"),
        (["septoria"], "Septoria_leaf_spot"),
        (["spider mite", "two spotted", "two-spotted"], LBL_TOMATO_SPIDER_MITES.split("___", 1)[1]),
        (["target spot"], "Target_Spot"),
        (["yellow leaf curl", "tylcv", "yellow curl"], "Tomato_Yellow_Leaf_Curl_Virus"),
        (["mosaic"], "Tomato_mosaic_virus"),
    ],
}

def normalize_species(raw: str):
    return SPECIES_NAME_MAP.get(str(raw).strip().lower())

def match_disease_keywords(species_pv: str, text: str):
    if species_pv not in DISEASE_KEYWORDS:
        return None
    text_l = str(text).lower().strip()
    for keywords, disease in DISEASE_KEYWORDS[species_pv]:
        for kw in keywords:
            if kw in text_l:
                return f"{species_pv}___{disease}"
    return None

def parse_fieldplant_class(raw_class: str):
    tokens = str(raw_class).strip().split()
    for n in (2, 1):
        if len(tokens) >= n:
            sp = normalize_species(" ".join(tokens[:n]))
            if sp:
                return sp, " ".join(tokens[n:])
    return None, raw_class

# ======================================================================
# MAPPING TƯỜNG MINH
# ======================================================================
PLANTDOC_MAP = {
    "Apple_Scab_Leaf": f"{SP_APPLE}___Apple_scab",
    "Apple_leaf": f"{SP_APPLE}___healthy",
    "Apple_rust_leaf": f"{SP_APPLE}___Cedar_apple_rust",
    "Bell_pepper_leaf": f"{SP_PEPPER}___healthy",
    "Bell_pepper_leaf_spot": f"{SP_PEPPER}___Bacterial_spot",
    "Cherry_leaf": f"{SP_CHERRY}___healthy",
    "Corn_Gray_leaf_spot": LBL_CORN_GRAY_LEAF_SPOT,
    "Corn_leaf_blight": f"{SP_CORN}___Northern_Leaf_Blight",
    "Corn_rust_leaf": f"{SP_CORN}___Common_rust_",
    "Peach_leaf": f"{SP_PEACH}___healthy",
    "Potato_leaf_early_blight": f"{SP_POTATO}___Early_blight",
    "Potato_leaf_late_blight": f"{SP_POTATO}___Late_blight",
    "Strawberry_leaf": f"{SP_STRAWBERRY}___healthy",
    "Tomato_Early_blight_leaf": f"{SP_TOMATO}___Early_blight",
    "Tomato_Septoria_leaf_spot": f"{SP_TOMATO}___Septoria_leaf_spot",
    "Tomato_leaf": f"{SP_TOMATO}___healthy",
    "Tomato_leaf_bacterial_spot": f"{SP_TOMATO}___Bacterial_spot",
    "Tomato_leaf_late_blight": f"{SP_TOMATO}___Late_blight",
    "Tomato_leaf_mosaic_virus": f"{SP_TOMATO}___Tomato_mosaic_virus",
    "Tomato_leaf_yellow_virus": f"{SP_TOMATO}___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mold_leaf": f"{SP_TOMATO}___Leaf_Mold",
    "Tomato_two_spotted_spider_mites_leaf": LBL_TOMATO_SPIDER_MITES,
    "grape_leaf": f"{SP_GRAPE}___healthy",
    "grape_leaf_black_rot": f"{SP_GRAPE}___Black_rot",
}

TOMATO_VILLAGE_MAP = {
    "Early_blight": f"{SP_TOMATO}___Early_blight",
    "Healthy": f"{SP_TOMATO}___healthy",
    "Late_blight": f"{SP_TOMATO}___Late_blight",
}

NGLD_MAP = {
    "Healthy Leaves": f"{SP_GRAPE}___healthy",
}

MERGED_DATASET_MAP = {
    "Apple__black_rot": f"{SP_APPLE}___Black_rot",
    "Apple__healthy": f"{SP_APPLE}___healthy",
    "Apple__rust": f"{SP_APPLE}___Cedar_apple_rust",
    "Apple__scab": f"{SP_APPLE}___Apple_scab",
    "Cherry__healthy": f"{SP_CHERRY}___healthy",
    "Cherry__powdery_mildew": f"{SP_CHERRY}___Powdery_mildew",
    "Corn__common_rust": f"{SP_CORN}___Common_rust_",
    "Corn__gray_leaf_spot": LBL_CORN_GRAY_LEAF_SPOT,
    "Corn__healthy": f"{SP_CORN}___healthy",
    "Corn__northern_leaf_blight": f"{SP_CORN}___Northern_Leaf_Blight",
    "Grape__black_measles": f"{SP_GRAPE}___Esca_(Black_Measles)",
    "Grape__black_rot": f"{SP_GRAPE}___Black_rot",
    "Grape__healthy": f"{SP_GRAPE}___healthy",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": f"{SP_GRAPE}___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Peach__bacterial_spot": f"{SP_PEACH}___Bacterial_spot",
    "Peach__healthy": f"{SP_PEACH}___healthy",
    "Pepper_bell__bacterial_spot": f"{SP_PEPPER}___Bacterial_spot",
    "Pepper_bell__healthy": f"{SP_PEPPER}___healthy",
    "Potato__early_blight": f"{SP_POTATO}___Early_blight",
    "Potato__healthy": f"{SP_POTATO}___healthy",
    "Potato__late_blight": f"{SP_POTATO}___Late_blight",
    "Strawberry___leaf_scorch": f"{SP_STRAWBERRY}___Leaf_scorch",
    "Strawberry__healthy": f"{SP_STRAWBERRY}___healthy",
    "Tomato__bacterial_spot": f"{SP_TOMATO}___Bacterial_spot",
    "Tomato__early_blight": f"{SP_TOMATO}___Early_blight",
    "Tomato__healthy": f"{SP_TOMATO}___healthy",
    "Tomato__late_blight": f"{SP_TOMATO}___Late_blight",
    "Tomato__leaf_mold": f"{SP_TOMATO}___Leaf_Mold",
    "Tomato__mosaic_virus": f"{SP_TOMATO}___Tomato_mosaic_virus",
    "Tomato__septoria_leaf_spot": f"{SP_TOMATO}___Septoria_leaf_spot",
    "Tomato__spider_mites_(two_spotted_spider_mite)": LBL_TOMATO_SPIDER_MITES,
    "Tomato__target_spot": f"{SP_TOMATO}___Target_Spot",
    "Tomato__yellow_leaf_curl_virus": f"{SP_TOMATO}___Tomato_Yellow_Leaf_Curl_Virus",
}

# ======================================================================
# CÁC HÀM HANDLER CỤ THỂ
# ======================================================================

# --- Handler ONFIELD cho PlantDoc (copy thẳng, không CLIP) ---
def handler_plantdoc_onfield(stats_on):
    base = DATA_ROOT / "PlantDoc"
    unmapped = Counter()
    for split in ("train", "test"):
        split_dir = base / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            pv_label = PLANTDOC_MAP.get(class_dir.name)
            if pv_label is None or pv_label not in PV_CLASS_SET:
                unmapped[class_dir.name] += 1
                continue
            for img_path in tqdm(list(class_dir.iterdir()),
                                 desc=f"PlantDoc_onfield/{split}/{class_dir.name}",
                                 leave=False):
                if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                    add_to_onfield(img_path, pv_label, f"PLANTDOC_{split}", stats_on)
    print_unmapped("PlantDoc (onfield)", unmapped, info_only=True)

# --- Handler LAB_SUSPEN cho TomatoDataset (có CLIP) ---
def handler_tomato_dataset_suspen(stats_on, stats_lab):
    base = DATA_ROOT / "TomatoDataset"
    unmapped = Counter()
    if not base.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {base}")
        return
    for class_dir in base.iterdir():
        if not class_dir.is_dir():
            continue
        pv_label = class_dir.name
        if pv_label not in PV_CLASS_SET:
            unmapped[pv_label] += 1
            continue
        for img_path in tqdm(list(class_dir.iterdir()),
                             desc=f"TomatoDataset_suspen/{class_dir.name}",
                             leave=False):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                img_type = classify_image_type(img_path)
                if img_type == "field":
                    add_to_onfield(img_path, pv_label, "TOMATODATASET", stats_on)
                elif img_type == "lab":
                    add_to_lab(img_path, pv_label, "TOMATODATASET", stats_lab)
                # fruit bỏ qua
    print_unmapped("TomatoDataset (suspen)", unmapped, info_only=True)

# --- Handler FRUIT_SUSPEN cho plantseg (có thêm đếm fruit) ---
def handler_plantseg(stats_on, stats_lab, stats_fruit):
    base = DATA_ROOT / "plantseg" / "plantseg"
    csv_path = base / "Metadata.csv"
    unmapped = Counter()
    if not csv_path.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {csv_path}")
        return
    df = pd.read_csv(csv_path)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PlantSeg"):
        plant = str(row.get("Plant", "")).strip()
        disease_raw = str(row.get("Disease", "")).strip()
        split = str(row.get("Training/Test", "")).strip().lower()
        if split not in ("training", "test"):
            continue
        species_pv = normalize_species(plant)
        if species_pv is None or species_pv not in CANDIDATE_SPECIES:
            unmapped[plant] += 1
            continue
        pv_label = match_disease_keywords(species_pv, disease_raw)
        if pv_label is None or pv_label not in PV_CLASS_SET:
            unmapped[f"{plant} | {disease_raw}"] += 1
            continue
        split_folder = "train" if split == "training" else "test"
        img_name = str(row.get("Name", "")).strip()
        src_path = base / "images" / split_folder / img_name
        if not src_path.exists():
            continue
        img_type = classify_image_type(src_path)
        if img_type == "field":
            add_to_onfield(src_path, pv_label, "PLANTSEG", stats_on)
        elif img_type == "lab":
            add_to_lab(src_path, pv_label, "PLANTSEG", stats_lab)
        else:  # fruit
            stats_fruit[pv_label] += 1
    print_unmapped("PlantSeg", unmapped)

# Các handler giữ nguyên: handler_tomato_village, handler_ngld, handler_master_plant_disease, handler_merged_dataset, ...
# (giữ nguyên code đã có ở trên, không thay đổi)

def handler_tomato_village(stats_on, stats_lab):
    base = DATA_ROOT / "Tomato-Village-main" / "Variant-a(Multiclass Classification)"
    unmapped = Counter()
    for split in ("train", "val", "test"):
        split_dir = base / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            pv_label = TOMATO_VILLAGE_MAP.get(class_dir.name)
            if pv_label is None or pv_label not in PV_CLASS_SET:
                unmapped[class_dir.name] += 1
                continue
            for img_path in tqdm(list(class_dir.iterdir()), desc=f"TomatoVillage/{split}/{class_dir.name}", leave=False):
                if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                    img_type = classify_image_type(img_path)
                    if img_type == "field":
                        add_to_onfield(img_path, pv_label, f"TOMVILLAGE_{split}", stats_on)
                    else:
                        add_to_lab(img_path, pv_label, f"TOMVILLAGE_{split}", stats_lab)
    print_unmapped("Tomato-Village (Variant-a)", unmapped, info_only=True)
    print("  [BỎ QUA] Variant-b/c như cũ.")

def handler_fieldplant(stats_on):
    base = DATA_ROOT / "FieldPlant" / "train"
    csv_path = base / "_annotations.csv"
    unmapped = Counter()
    if not csv_path.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {csv_path}")
        return
    df = pd.read_csv(csv_path)
    idx = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="FieldPlant"):
        raw_class = str(row.get("class", "")).strip()
        species_pv, disease_text = parse_fieldplant_class(raw_class)
        if species_pv is None or species_pv not in CANDIDATE_SPECIES:
            unmapped[raw_class] += 1
            continue
        pv_label = match_disease_keywords(species_pv, disease_text)
        if pv_label is None or pv_label not in PV_CLASS_SET:
            unmapped[raw_class] += 1
            continue
        filename = str(row.get("filename", "")).strip()
        src_path = base / filename
        if not src_path.exists():
            continue
        try:
            xmin, ymin = max(0, int(row["xmin"])), max(0, int(row["ymin"]))
            xmax, ymax = int(row["xmax"]), int(row["ymax"])
            with Image.open(src_path) as img:
                xmax, ymax = min(img.width, xmax), min(img.height, ymax)
                if xmax <= xmin or ymax <= ymin:
                    continue
                cropped = img.crop((xmin, ymin, xmax, ymax))
                idx += 1
                add_cropped_to_onfield(cropped, pv_label, "FIELDPLANT", idx, stats_on)
        except Exception as e:
            print(f"  [LỖI] Không đọc/crop được ảnh {src_path}: {e}")
    print_unmapped("FieldPlant (class không map được theo keyword)", unmapped)

def handler_plant_pathology_2020(stats_on):
    base = DATA_ROOT / "Plant Pathology 2020"
    csv_path = base / "train.csv"
    images_dir = base / "images"
    skipped_multi = 0
    if not csv_path.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {csv_path}")
        return
    df = pd.read_csv(csv_path)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PlantPathology2020"):
        if int(row.get("multiple_diseases", 0)) == 1:
            skipped_multi += 1
            continue
        if int(row.get("healthy", 0)) == 1:
            disease = "healthy"
        elif int(row.get("rust", 0)) == 1:
            disease = "Cedar_apple_rust"
        elif int(row.get("scab", 0)) == 1:
            disease = "Apple_scab"
        else:
            continue
        pv_label = f"{SP_APPLE}___{disease}"
        if pv_label not in PV_CLASS_SET:
            continue
        src_path = find_image(images_dir, str(row["image_id"]))
        if src_path is None:
            continue
        add_to_onfield(src_path, pv_label, "PP2020", stats_on)
    print(f"  (Bỏ qua {skipped_multi} ảnh multiple_diseases=1)")

def handler_plant_pathology_2021(stats_on):
    base = DATA_ROOT / "Plant Pathology 2021"
    csv_path = base / "train.csv"
    images_dir = base / "train_images"
    unmapped = Counter()
    skipped_multi = 0
    if not csv_path.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {csv_path}")
        return
    df = pd.read_csv(csv_path)
    label_map = {
        "healthy": f"{SP_APPLE}___healthy",
        "scab": f"{SP_APPLE}___Apple_scab",
        "rust": f"{SP_APPLE}___Cedar_apple_rust",
    }
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PlantPathology2021"):
        tokens = str(row.get("labels", "")).strip().split()
        if len(tokens) != 1:
            skipped_multi += 1
            continue
        pv_label = label_map.get(tokens[0].lower())
        if pv_label is None:
            unmapped[tokens[0]] += 1
            continue
        src_path = images_dir / str(row["image"])
        if not src_path.exists():
            continue
        add_to_onfield(src_path, pv_label, "PP2021", stats_on)
    print(f"  (Bỏ qua {skipped_multi} ảnh multi-label)")
    print_unmapped("Plant Pathology 2021 (nhãn không thuộc PV)", unmapped, info_only=True)

def handler_ngld(stats_on, stats_lab):
    base = (DATA_ROOT / "Niphad Grape Leaf Disease Dataset (NGLD)"
            / "Niphad Grape Leaf Disease Dataset (NGLD)" / "Grapes Disease Dataset")
    unmapped = Counter()
    if not base.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {base}")
        return
    for class_dir in base.iterdir():
        if not class_dir.is_dir():
            continue
        pv_label = NGLD_MAP.get(class_dir.name)
        if pv_label is None or pv_label not in PV_CLASS_SET:
            unmapped[class_dir.name] += 1
            continue
        for img_path in tqdm(list(class_dir.iterdir()), desc=f"NGLD/{class_dir.name}", leave=False):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                img_type = classify_image_type(img_path)
                if img_type == "field":
                    add_to_onfield(img_path, pv_label, "NGLD", stats_on)
                elif img_type == "lab":
                    add_to_lab(img_path, pv_label, "NGLD", stats_lab)
                # fruit bỏ qua
    print_unmapped("NGLD", unmapped, info_only=True)

def handler_master_plant_disease(stats_on, stats_lab):
    base = DATA_ROOT / "Master Plant Disease Dataset"
    images_base = base / "master_images" / "master_images"
    unmapped = Counter()
    for split_file in ("train_split.csv", "val_split.csv"):
        csv_path = base / "outputs" / "outputs" / split_file
        if not csv_path.exists():
            print(f"  [CẢNH BÁO] Không tìm thấy {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Master/{split_file}"):
            crop_raw = str(row.get("crop", "")).strip()
            species_pv = normalize_species(crop_raw)
            if species_pv is None or species_pv not in CANDIDATE_SPECIES:
                continue
            canonical = str(row.get("canonical_class", ""))
            disease_text = canonical.split(" - ", 1)[1] if " - " in canonical else canonical
            pv_label = match_disease_keywords(species_pv, disease_text)
            if pv_label is None or pv_label not in PV_CLASS_SET:
                unmapped[f"{crop_raw} | {canonical}"] += 1
                continue
            src_path = images_base / str(row.get("image_path", "")).strip()
            if not src_path.exists():
                continue
            img_type = classify_image_type(src_path)
            if img_type == "field":
                add_to_onfield(src_path, pv_label, "MASTER", stats_on)
            elif img_type == "lab":
                add_to_lab(src_path, pv_label, "MASTER", stats_lab)
            # fruit bỏ qua
    print_unmapped("Master Plant Disease Dataset (canonical_class không map được theo keyword)", unmapped)

def handler_merged_dataset(stats_on, stats_lab):
    base = DATA_ROOT / "Plant Disease Classification Merged Dataset"
    unmapped = Counter()
    if not base.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {base}")
        return
    for class_dir in base.iterdir():
        if not class_dir.is_dir():
            continue
        pv_label = MERGED_DATASET_MAP.get(class_dir.name)
        if pv_label is None:
            unmapped[class_dir.name] += 1
            continue
        if pv_label not in PV_CLASS_SET:
            print(f"  [CẢNH BÁO] '{class_dir.name}' map ra '{pv_label}' nhưng nhãn này không có trong PlantVillage")
            continue
        for img_path in tqdm(list(class_dir.iterdir()), desc=f"Merged/{class_dir.name}", leave=False):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                img_type = classify_image_type(img_path)
                if img_type == "field":
                    add_to_onfield(img_path, pv_label, "MERGED", stats_on)
                elif img_type == "lab":
                    add_to_lab(img_path, pv_label, "MERGED", stats_lab)
    print_unmapped("Plant Disease Classification Merged Dataset", unmapped, info_only=True)

# ======================================================================
# MAIN
# ======================================================================
def main():
    global PV_CLASS_SET, CANDIDATE_SPECIES

    print("Đang đọc danh sách lớp PlantVillage...")
    PV_CLASS_SET = get_pv_classes()
    if not PV_CLASS_SET:
        print(f"[LỖI] Không đọc được lớp nào từ {PV_ROOT}. Kiểm tra lại đường dẫn DATA_ROOT.")
        return
    CANDIDATE_SPECIES = get_candidate_species(PV_CLASS_SET)
    print(f"\n>>> CANDIDATE SPECIES ({len(CANDIDATE_SPECIES)} loài):")
    for sp in CANDIDATE_SPECIES:
        print(f"    - {sp}")

    ONFIELD_DIR.mkdir(parents=True, exist_ok=True)
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    # FRUIT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nĐang khởi tạo CLIP...")
    device_clip = "cuda" if torch.cuda.is_available() else "cpu"
    init_clip(device_clip)

    # Tổng hợp thống kê
    total_onfield = Counter()
    total_lab = Counter()
    total_fruit = Counter()

    # -------- ONFIELD datasets ----------
    for name in ONFIELD_DATASETS:
        print(f"\n{'='*60}\nONFIELD: {name}\n{'='*60}")
        try:
            if name == "PlantDoc":
                stats = Counter()
                handler_plantdoc_onfield(stats)
                total_onfield.update(stats)
            elif name == "FieldPlant":
                stats = Counter()
                handler_fieldplant(stats)
                total_onfield.update(stats)
            elif name == "Plant Pathology 2020":
                stats = Counter()
                handler_plant_pathology_2020(stats)
                total_onfield.update(stats)
            elif name == "Plant Pathology 2021":
                stats = Counter()
                handler_plant_pathology_2021(stats)
                total_onfield.update(stats)
        except Exception as e:
            print(f"  [LỖI] {e}")

    # -------- LAB_SUSPEN datasets ----------
    for name in LAB_SUSPEN_DATASETS:
        print(f"\n{'='*60}\nLAB_SUSPEN: {name}\n{'='*60}")
        try:
            if name == "Tomato-Village-main":
                s_on, s_lab = Counter(), Counter()
                handler_tomato_village(s_on, s_lab)
                total_onfield.update(s_on)
                total_lab.update(s_lab)
            elif name == "Niphad Grape Leaf Disease Dataset (NGLD)":
                s_on, s_lab = Counter(), Counter()
                handler_ngld(s_on, s_lab)
                total_onfield.update(s_on)
                total_lab.update(s_lab)
            elif name == "Master Plant Disease Dataset":
                s_on, s_lab = Counter(), Counter()
                handler_master_plant_disease(s_on, s_lab)
                total_onfield.update(s_on)
                total_lab.update(s_lab)
            elif name == "Plant Disease Classification Merged Dataset":
                s_on, s_lab = Counter(), Counter()
                handler_merged_dataset(s_on, s_lab)
                total_onfield.update(s_on)
                total_lab.update(s_lab)
            elif name == "TomatoDataset":
                s_on, s_lab = Counter(), Counter()
                handler_tomato_dataset_suspen(s_on, s_lab)
                total_onfield.update(s_on)
                total_lab.update(s_lab)
        except Exception as e:
            print(f"  [LỖI] {e}")

    # -------- FRUIT_SUSPEN datasets ----------
    for name in FRUIT_SUSPEN_DATASETS:
        print(f"\n{'='*60}\nFRUIT_SUSPEN: {name}\n{'='*60}")
        try:
            if name == "plantseg":
                s_on, s_lab, s_fruit = Counter(), Counter(), Counter()
                handler_plantseg(s_on, s_lab, s_fruit)
                total_onfield.update(s_on)
                total_lab.update(s_lab)
                total_fruit.update(s_fruit)
        except Exception as e:
            print(f"  [LỖI] {e}")

    # In thống kê cuối cùng
    print(f"\n{'='*70}\nTHỐNG KÊ CUỐI CÙNG\n{'='*70}")
    print("\n>>> ONFIELD_LEAF_DETECTED:")
    for label, cnt in sorted(total_onfield.items()):
        print(f"    {label}: {cnt}")
    print(f"Tổng ảnh field: {sum(total_onfield.values())}")

    print("\n>>> LAB_LEAF_DETECTED:")
    for label, cnt in sorted(total_lab.items()):
        print(f"    {label}: {cnt}")
    print(f"Tổng ảnh lab: {sum(total_lab.values())}")

    if total_fruit:
        print("\n>>> ẢNH QUẢ BỊ BỎ QUA:")
        for label, cnt in sorted(total_fruit.items()):
            print(f"    {label}: {cnt}")
        print(f"Tổng ảnh quả: {sum(total_fruit.values())}")

    print("\n===== HOÀN TẤT =====")

if __name__ == "__main__":
    main()