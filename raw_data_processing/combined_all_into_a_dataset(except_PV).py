# -*- coding: utf-8 -*-
"""
GIAI ĐOẠN 1 — Gom dữ liệu "field" từ nhiều dataset bệnh cây về Field_Raw,
dùng định dạng nhãn của PlantVillage: "Loài___Bệnh".

Cách chạy:
    pip install pandas pillow tqdm
    python stage1_build_field_raw.py

Yêu cầu: các dataset gốc nằm trong DATA_ROOT (xem CONFIG bên dưới), giữ
nguyên cấu trúc thư mục như đã liệt kê. Script CHỈ COPY ảnh (đối với
FieldPlant thì crop + resize trước khi lưu), KHÔNG xóa/sửa dữ liệu gốc.

Sau khi script chạy xong, thư mục Field_Raw sẽ chứa ảnh tổ chức theo:
    Field_Raw/<Loài>/<Loài___Bệnh>/<tên_file>.jpg

LƯU Ý QUAN TRỌNG VỀ ĐỘ TIN CẬY CỦA MAPPING:
- PlantDoc, Tomato-Village, TomatoDataset, NGLD, Plant Disease Classification
  Merged Dataset: mapping dùng dictionary tường minh (độ tin cậy cao, dựa
  trên tên thư mục/lớp đã biết).
- Plant Pathology 2020/2021: mapping dùng cột nhãn CSV đã biết cấu trúc
  (độ tin cậy cao, đây là 2 dataset chỉ chứa lá Apple).
- FieldPlant và Master Plant Disease Dataset: vì không có danh sách đầy đủ
  toàn bộ tên lớp gốc trong cây thư mục được cung cấp, script dùng một bộ
  match theo TỪ KHÓA (keyword-based) để suy ra "Loài___Bệnh". Những lớp
  không khớp được sẽ in CẢNH BÁO và bị bỏ qua — hãy xem lại danh sách này
  sau khi chạy xong để bổ sung mapping thủ công nếu cần.
"""

import shutil
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow cũ
    RESAMPLE = Image.LANCZOS

# ======================================================================
# CONFIG
# ======================================================================
DATA_ROOT = Path(r"D:\DAT301m_Project\data")
PV_ROOT = DATA_ROOT / "PlantVillage"
FIELD_RAW = DATA_ROOT / "Field_Raw"

CROP_RESIZE_SIZE = (224, 224)
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Sẽ được điền sau khi đọc PlantVillage
PV_CLASS_SET = set()
CANDIDATE_SPECIES = []

# ======================================================================
# TÊN LOÀI CHUẨN (đúng như PlantVillage đặt tên thư mục)
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

# Hai nhãn PV có dấu cách "lạ" bên trong, cần ghi đúng tuyệt đối:
LBL_CORN_GRAY_LEAF_SPOT = f"{SP_CORN}___Cercospora_leaf_spot Gray_leaf_spot"
LBL_TOMATO_SPIDER_MITES = f"{SP_TOMATO}___Spider_mites Two-spotted_spider_mite"


# ======================================================================
# HÀM TIỆN ÍCH CHUNG
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
    """Loài 'ứng viên' = có ít nhất 1 lớp healthy và ít nhất 1 lớp bệnh."""
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


def add_to_field_raw(src_path: Path, pv_label: str, source_tag: str, stats: Counter):
    species = pv_label.split("___")[0]
    dest_dir = FIELD_RAW / species / pv_label
    filename = f"{source_tag}_{src_path.name}"
    safe_copy(src_path, dest_dir, filename)
    stats[pv_label] += 1


def add_cropped_to_field_raw(img: Image.Image, pv_label: str, source_tag: str, idx: int, stats: Counter):
    species = pv_label.split("___")[0]
    dest_dir = FIELD_RAW / species / pv_label
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
# MAPPING TỪ-KHÓA (dùng cho FieldPlant & Master Plant Disease Dataset,
# vì tên lớp gốc của 2 dataset này không đầy đủ/khó đoán chính xác 100%)
# ======================================================================
SPECIES_NAME_MAP = {
    "apple": SP_APPLE,
    "cherry": SP_CHERRY,
    "sour cherry": SP_CHERRY,
    "corn": SP_CORN,
    "maize": SP_CORN,
    "grape": SP_GRAPE,
    "grapes": SP_GRAPE,
    "peach": SP_PEACH,
    "pepper": SP_PEPPER,
    "bell pepper": SP_PEPPER,
    "capsicum": SP_PEPPER,
    "potato": SP_POTATO,
    "strawberry": SP_STRAWBERRY,
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
    """'Cassava Mosaic' -> ('Cassava', 'Mosaic') ; thử ghép 2 từ rồi 1 từ đầu làm tên loài."""
    tokens = str(raw_class).strip().split()
    for n in (2, 1):
        if len(tokens) >= n:
            sp = normalize_species(" ".join(tokens[:n]))
            if sp:
                return sp, " ".join(tokens[n:])
    return None, raw_class


# ======================================================================
# MAPPING TƯỜNG MINH (độ tin cậy cao) CHO TỪNG DATASET
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
    # Blueberry_leaf, Raspberry_leaf, Soyabean_leaf, Squash_Powdery_mildew_leaf:
    # loài không phải candidate (PlantVillage không có lớp bệnh tương ứng) -> không map
}

TOMATO_VILLAGE_MAP = {
    "Early_blight": f"{SP_TOMATO}___Early_blight",
    "Healthy": f"{SP_TOMATO}___healthy",
    "Late_blight": f"{SP_TOMATO}___Late_blight",
    # Leaf Miner, Magnesium/Nitrogen/Pottassium Deficiency, Spotted Wilt Virus:
    # không tồn tại trong PlantVillage -> không map
}

NGLD_MAP = {
    "Healthy Leaves": f"{SP_GRAPE}___healthy",
    # Bacterial Leaf Spot, Downy Mildew, Powdery Mildew: không khớp với
    # 3 lớp bệnh grape của PlantVillage (Black_rot/Esca/Leaf_blight) -> không map
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
    "Strawberry___leaf_scorch": f"{SP_STRAWBERRY}___Leaf_scorch",  # tên thư mục gốc lỗi gõ (3 dấu _)
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
    # Blueberry, Cassava, Chili, Coffee, Cucumber, Gauva, Jamun, Lemon, Mango,
    # Pomegranate, Rice, Soybean, Sugarcane, Tea, Wheat: không phải candidate -> không map
}


# ======================================================================
# XỬ LÝ TỪNG DATASET
# ======================================================================
def process_plantdoc():
    base = DATA_ROOT / "PlantDoc"
    stats, unmapped = Counter(), Counter()
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
            for img_path in tqdm(list(class_dir.iterdir()), desc=f"PlantDoc/{split}/{class_dir.name}", leave=False):
                if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                    add_to_field_raw(img_path, pv_label, f"PLANTDOC_{split}", stats)
    print_unmapped("PlantDoc", unmapped, info_only=True)
    return stats


def process_tomato_village():
    base = DATA_ROOT / "Tomato-Village-main" / "Variant-a(Multiclass Classification)"
    stats, unmapped = Counter(), Counter()
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
                    add_to_field_raw(img_path, pv_label, f"TOMVILLAGE_{split}", stats)
    print_unmapped("Tomato-Village (Variant-a)", unmapped, info_only=True)
    print("  [BỎ QUA] Variant-b: multi-label trùng lặp dữ liệu với Variant-a (đã augment) -> không dùng để tránh trùng lặp.")
    print("  [BỎ QUA] Variant-c: object detection, không rõ danh sách class trong annotation -> không dùng.")
    return stats


def process_tomato_dataset():
    base = DATA_ROOT / "TomatoDataset"
    stats, unmapped = Counter(), Counter()
    if not base.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {base}")
        return stats
    for class_dir in base.iterdir():
        if not class_dir.is_dir():
            continue
        pv_label = class_dir.name  # đã đúng định dạng PV sẵn
        if pv_label not in PV_CLASS_SET:
            unmapped[pv_label] += 1
            continue
        for img_path in tqdm(list(class_dir.iterdir()), desc=f"TomatoDataset/{class_dir.name}", leave=False):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                add_to_field_raw(img_path, pv_label, "TOMATODATASET", stats)
    print_unmapped("TomatoDataset", unmapped)
    return stats


def process_fieldplant():
    base = DATA_ROOT / "FieldPlant" / "train"
    csv_path = base / "_annotations.csv"
    stats, unmapped = Counter(), Counter()
    if not csv_path.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {csv_path}")
        return stats
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
                add_cropped_to_field_raw(cropped, pv_label, "FIELDPLANT", idx, stats)
        except Exception as e:
            print(f"  [LỖI] Không đọc/crop được ảnh {src_path}: {e}")
    print_unmapped("FieldPlant (class không map được theo keyword)", unmapped)
    return stats


def process_plant_pathology_2020():
    base = DATA_ROOT / "Plant Pathology 2020"
    csv_path = base / "train.csv"
    images_dir = base / "images"
    stats = Counter()
    skipped_multi = 0
    if not csv_path.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {csv_path}")
        return stats
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
        add_to_field_raw(src_path, pv_label, "PP2020", stats)
    print(f"  (Bỏ qua {skipped_multi} ảnh multiple_diseases=1; bỏ qua test.csv vì không có nhãn)")
    return stats


def process_plant_pathology_2021():
    base = DATA_ROOT / "Plant Pathology 2021"
    csv_path = base / "train.csv"
    images_dir = base / "train_images"
    stats, unmapped = Counter(), Counter()
    skipped_multi = 0
    if not csv_path.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {csv_path}")
        return stats
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
        add_to_field_raw(src_path, pv_label, "PP2021", stats)
    print(f"  (Bỏ qua {skipped_multi} ảnh multi-label; bỏ qua test_images vì không có nhãn)")
    print_unmapped("Plant Pathology 2021 (nhãn không thuộc PV)", unmapped, info_only=True)
    return stats


def process_ngld():
    base = (DATA_ROOT / "Niphad Grape Leaf Disease Dataset (NGLD)"
            / "Niphad Grape Leaf Disease Dataset (NGLD)" / "Grapes Disease Dataset")
    stats, unmapped = Counter(), Counter()
    if not base.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {base}")
        return stats
    for class_dir in base.iterdir():
        if not class_dir.is_dir():
            continue
        pv_label = NGLD_MAP.get(class_dir.name)
        if pv_label is None or pv_label not in PV_CLASS_SET:
            unmapped[class_dir.name] += 1
            continue
        for img_path in tqdm(list(class_dir.iterdir()), desc=f"NGLD/{class_dir.name}", leave=False):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                add_to_field_raw(img_path, pv_label, "NGLD", stats)
    print_unmapped("NGLD", unmapped, info_only=True)
    return stats


def process_master_plant_disease():
    base = DATA_ROOT / "Master Plant Disease Dataset"
    images_base = base / "master_images" / "master_images"
    stats, unmapped = Counter(), Counter()
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
                continue  # loài không phải candidate -> bỏ qua êm, không cần cảnh báo
            canonical = str(row.get("canonical_class", ""))
            disease_text = canonical.split(" - ", 1)[1] if " - " in canonical else canonical
            pv_label = match_disease_keywords(species_pv, disease_text)
            if pv_label is None or pv_label not in PV_CLASS_SET:
                unmapped[f"{crop_raw} | {canonical}"] += 1
                continue
            src_path = images_base / str(row.get("image_path", "")).strip()
            if not src_path.exists():
                continue
            add_to_field_raw(src_path, pv_label, "MASTER", stats)
    print_unmapped("Master Plant Disease Dataset (canonical_class không map được theo keyword)", unmapped)
    return stats


def process_merged_dataset():
    base = DATA_ROOT / "Plant Disease Classification Merged Dataset"
    stats, unmapped = Counter(), Counter()
    if not base.exists():
        print(f"  [CẢNH BÁO] Không tìm thấy {base}")
        return stats
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
                add_to_field_raw(img_path, pv_label, "MERGED", stats)
    print_unmapped("Plant Disease Classification Merged Dataset", unmapped, info_only=True)
    return stats


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

    print(f"\n>>> CANDIDATE SPECIES ({len(CANDIDATE_SPECIES)} loài có cả healthy & bệnh trong PlantVillage):")
    for sp in CANDIDATE_SPECIES:
        print(f"    - {sp}")

    FIELD_RAW.mkdir(parents=True, exist_ok=True)

    handlers = [
        ("PlantDoc", process_plantdoc),
        ("Tomato-Village-main", process_tomato_village),
        ("TomatoDataset", process_tomato_dataset),
        ("FieldPlant", process_fieldplant),
        ("Plant Pathology 2020", process_plant_pathology_2020),
        ("Plant Pathology 2021", process_plant_pathology_2021),
        ("Niphad Grape Leaf Disease Dataset (NGLD)", process_ngld),
        ("Master Plant Disease Dataset", process_master_plant_disease),
        ("Plant Disease Classification Merged Dataset", process_merged_dataset),
    ]

    all_stats = Counter()
    for name, fn in handlers:
        print(f"\n{'=' * 70}\nĐANG XỬ LÝ: {name}\n{'=' * 70}")
        try:
            stats = fn()
            all_stats.update(stats)
        except Exception as e:
            print(f"  [LỖI] Xử lý '{name}' thất bại: {e}")

    print(f"\n{'=' * 70}\nTHỐNG KÊ SỐ ẢNH MỖI LỚP TRONG Field_Raw (sau Giai đoạn 1)\n{'=' * 70}")
    for label, cnt in sorted(all_stats.items()):
        print(f"    {label}: {cnt} ảnh")
    print(f"\nTổng số ảnh trong Field_Raw: {sum(all_stats.values())}")

    print("\n===== GIAI ĐOẠN 1 HOÀN TẤT =====")
    print("Đã gom dữ liệu vào Field_Raw. Vui lòng lọc thủ công ảnh lab trước khi chạy tiếp.")


if __name__ == "__main__":
    main()
