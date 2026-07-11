"""Ánh xạ nhãn cho Cross-Test Chiều 2 (model công khai train trên PlantVillage -> test
holdout FIELD của mình). Map 20 lớp nội bộ (per-species) sang chỉ số output của model công khai
THEO TÊN CHUẨN HOÁ (không hard-code số) nên đúng cho cả backbone-38 lẫn DeiT-39.

Nguồn nhãn (đã đối chiếu 2026-07-12):
- PlantVillage-38 (backbone của mình): thứ tự = sorted() thư mục PRETRAIN DATASET/train
  (split_builder/PretrainDataset đều sort tên lớp). Đây chính là 38 lớp PlantVillage.
  Dataset gốc: Hughes & Salathé 2015, https://arxiv.org/abs/1511.08060
- DeiT-39 (model ngoài): rescu/deit-base-patch16-224-finetuned-plantvillage
  Model card:  https://huggingface.co/rescu/deit-base-patch16-224-finetuned-plantvillage
  id2label:    https://huggingface.co/rescu/deit-base-patch16-224-finetuned-plantvillage/raw/main/config.json
  (38 lớp PlantVillage + 1 lớp 'Background_without_leaves' -> tổng 39)
Cơ sở khoa học chiều 2: Mohanty et al. 2016, https://doi.org/10.3389/fpls.2016.01419
  (99.35% trên lab -> ~31% trên ảnh ngoài phân phối).
"""
import os
import re

# PlantVillage-38: thứ tự sorted thư mục PRETRAIN (khớp backbone_pretrained.pt).
PRETRAIN_TRAIN_DIR = r"D:/DAT301m_Project/Final_Pipeline/data/1. PRETRAIN DATASET/train"

# id2label của DeiT-39 (chép nguyên từ config.json ở link trên; index 0..38).
DEIT39_LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Background_without_leaves", "Blueberry___healthy", "Cherry___Powdery_mildew",
    "Cherry___healthy", "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight", "Corn___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def canon(label: str) -> str:
    """Chuẩn hoá nhãn để so khớp giữa các bộ tên khác nhau.
    Bỏ '(maize)', dấu ngoặc/phẩy/gạch, hạ thường, gộp khoảng trắng.
    Ví dụ: 'Corn_(maize)___Northern_Leaf_Blight' == 'Corn___Northern_Leaf_Blight'."""
    s = label.lower().replace("(maize)", "")
    s = s.replace("___", " ").replace("_", " ").replace(",", " ")
    s = re.sub(r"[()]", " ", s)
    return " ".join(s.split())


def load_pv38_labels(pretrain_dir: str = PRETRAIN_TRAIN_DIR):
    """Trả list 38 nhãn PlantVillage theo đúng thứ tự backbone (sorted thư mục)."""
    return sorted(c for c in os.listdir(pretrain_dir)
                  if os.path.isdir(os.path.join(pretrain_dir, c)))


def build_species_index_map(my_classes_ordered, public_labels):
    """my_classes_ordered: list nhãn nội bộ của 1 loài (theo class_idx 0..K-1 trong manifest).
    public_labels: list nhãn output của model công khai (index 0..N-1).
    Trả (local_to_public, unmatched):
      - local_to_public[k] = index trong public_labels khớp my_classes_ordered[k], hoặc None.
      - unmatched = list nhãn nội bộ không tìm được lớp tương ứng.
    Khớp theo canon(); nếu >1 public label cùng canon thì lấy cái đầu tiên (ổn định)."""
    pub_canon = {}
    for i, lab in enumerate(public_labels):
        pub_canon.setdefault(canon(lab), i)
    local_to_public, unmatched = [], []
    for c in my_classes_ordered:
        idx = pub_canon.get(canon(c))
        local_to_public.append(idx)
        if idx is None:
            unmatched.append(c)
    return local_to_public, unmatched
