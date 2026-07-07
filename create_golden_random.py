import os
import json
import random

PROJECT_ROOT = "D:/DAT301m_Project/Final_Pipeline"
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
GOLDEN_DIR = os.path.join(PROJECT_ROOT, "data", "golden")

# Cấu hình
MIN_IMAGES_PER_CLASS = 3
MAX_IMAGES_PER_SPECIES = 30   # Giới hạn tổng, nhưng ưu tiên lấy đủ mỗi lớp >= MIN
SEED = 42

random.seed(SEED)
os.makedirs(GOLDEN_DIR, exist_ok=True)

def get_train_val_images(species):
    manifest_path = os.path.join(SPLITS_DIR, f"{species}_split_v1.json")
    if not os.path.exists(manifest_path):
        print(f"❌ Không tìm thấy manifest cho {species}")
        return []

    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = []
    # Lấy đường dẫn tuyệt đối từ relative_path + dataset_root
    dataset_root = data.get("dataset_root", "")
    for item in data.get("items", []):
        split = item.get("split")
        if split in ("train", "val"):
            rel_path = item.get("relative_path")
            if rel_path:
                # Nếu dataset_root là tuyệt đối, nối với relative_path
                abs_path = os.path.join(dataset_root, rel_path)
                # Nếu file tồn tại thì thêm
                if os.path.exists(abs_path):
                    images.append(abs_path)
                else:
                    print(f"⚠️ Ảnh không tồn tại: {abs_path}")
    return images

species_list = ["apple", "corn", "grape", "pepper_bell", "tomato"]

for species in species_list:
    all_images = get_train_val_images(species)
    if not all_images:
        print(f"⚠️ Không có ảnh hợp lệ cho {species}")
        continue

    # Nhóm theo tên lớp (từ thư mục cha)
    class_groups = {}
    for img_path in all_images:
        class_name = os.path.basename(os.path.dirname(img_path))
        class_groups.setdefault(class_name, []).append(img_path)

    selected = []
    # 1. Lấy MIN_IMAGES_PER_CLASS ảnh từ mỗi lớp (nếu có đủ)
    for class_name, imgs in class_groups.items():
        if len(imgs) >= MIN_IMAGES_PER_CLASS:
            selected.extend(random.sample(imgs, MIN_IMAGES_PER_CLASS))
        else:
            # Nếu ít hơn, lấy tất cả
            selected.extend(imgs)
            print(f"⚠️ {species}/{class_name} chỉ có {len(imgs)} ảnh, < {MIN_IMAGES_PER_CLASS}")

    # 2. Nếu tổng số đã chọn vẫn ít hơn MAX_IMAGES_PER_SPECIES, lấy thêm ngẫu nhiên
    remaining = MAX_IMAGES_PER_SPECIES - len(selected)
    if remaining > 0:
        pool = [img for img in all_images if img not in selected]
        random.shuffle(pool)
        selected.extend(pool[:remaining])

    # Ghi file
    golden_path = os.path.join(GOLDEN_DIR, f"{species}.json")
    with open(golden_path, 'w', encoding='utf-8') as f:
        json.dump({"images": selected}, f, indent=2, ensure_ascii=False)

    print(f"✅ {species}: đã chọn {len(selected)} ảnh -> {golden_path}")

print("🎉 Hoàn tất!")