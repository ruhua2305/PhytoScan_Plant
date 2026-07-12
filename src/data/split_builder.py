"""
Xây split manifest DETERMINISTIC cho FINETUNE DATASET (Phase 0, Mục 2.4-2.5).
- Lớp thường (>= rare_class_threshold): stratified 70/10/10/10 theo seed cố định.
- Lớp hiếm (< threshold): stratified K-fold vào khối rare_folds (Chế độ 2, Mục 2.3).
Mọi ảnh được hash SHA256 để khoá holdout & phát hiện rò rỉ (Mục 2.5).

Chạy: python -m src.data.split_builder --species all --seed 42
"""
import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone, timedelta

IMG_EXT = (".jpg", ".jpeg", ".png")
SPECIES = ["apple", "corn", "grape", "pepper_bell", "tomato"]
DATASET_ROOT = r"D:/DAT301m_Project/Final_Pipeline/data/2. FINETUNE DATASET"
SPLITS_DIR = r"D:/DAT301m_Project/Final_Pipeline/data/splits"
RATIO = {"train": 0.70, "calib": 0.10, "val": 0.10, "holdout": 0.10}

# Key chuẩn nội bộ (dùng cho tên file split, config) -> tên thư mục THẬT trên đĩa.
# Tên thư mục thật có thể chứa dấu ngoặc/phẩy nên không dùng trực tiếp làm key.
SPECIES_DIR_NAME = {
    "apple": "Apple",
    "corn": "Corn_(maize)",
    "grape": "Grape",
    "pepper_bell": "Pepper,_bell",
    "tomato": "Tomato",
}


def resolve_species_dir(species: str) -> str:
    """Trả đường dẫn thư mục loài. Ưu tiên tên map sẵn; nếu không có thì dò
    không phân biệt hoa/thường trong DATASET_ROOT (bền với khác biệt tên)."""
    candidate = SPECIES_DIR_NAME.get(species, species)
    path = os.path.join(DATASET_ROOT, candidate)
    if os.path.isdir(path):
        return path

    def norm(s):
        return (s.lower().replace("_", "").replace(",", "")
                .replace("(", "").replace(")", "").replace(" ", ""))
    target = norm(species)
    for name in os.listdir(DATASET_ROOT):
        full = os.path.join(DATASET_ROOT, name)
        if os.path.isdir(full) and (norm(name) == target or norm(name).startswith(target)):
            return full
    raise FileNotFoundError(
        f"Không tìm thấy thư mục loài '{species}' trong {DATASET_ROOT}. "
        f"Có: {os.listdir(DATASET_ROOT)}. Cập nhật SPECIES_DIR_NAME nếu cần.")


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def list_class_images(species_dir: str):
    """Trả {class_name: [abs_path, ...]} sắp xếp ổn định để tái lập."""
    classes = sorted(c for c in os.listdir(species_dir)
                     if os.path.isdir(os.path.join(species_dir, c)))
    out = {}
    for cls in classes:
        cdir = os.path.join(species_dir, cls)
        paths = sorted(os.path.join(cdir, f) for f in os.listdir(cdir)
                       if f.lower().endswith(IMG_EXT))
        out[cls] = paths
    return out


def stratified_split(paths, rng):
    """Chia 1 lớp thành train/calib/val/holdout theo RATIO, số lẻ dồn train trước."""
    paths = paths[:]
    rng.shuffle(paths)
    n = len(paths)
    n_calib = int(round(n * RATIO["calib"]))
    n_val = int(round(n * RATIO["val"]))
    n_hold = int(round(n * RATIO["holdout"]))
    n_train = n - n_calib - n_val - n_hold
    idx = 0
    out = {}
    for name, k in [("train", n_train), ("calib", n_calib), ("val", n_val), ("holdout", n_hold)]:
        out[name] = paths[idx:idx + k]
        idx += k
    return out


def kfold_split(paths, rng, k=5):
    """Lớp hiếm: chia đều vào k fold (mỗi ảnh 1 fold) để đánh giá prototype."""
    paths = paths[:]
    rng.shuffle(paths)
    folds = []
    for i, p in enumerate(paths):
        folds.append([i % max(1, min(k, len(paths))), p])
    return folds


def build_species(species: str, seed: int, rare_threshold: int = 30):
    rng = random.Random(seed)
    species_dir = resolve_species_dir(species)
    class_images = list_class_images(species_dir)
    class_to_idx = {c: i for i, c in enumerate(sorted(class_images.keys()))}

    items, rare_folds = [], {}
    for cls, paths in class_images.items():
        if len(paths) < rare_threshold:
            rare_folds[cls] = kfold_split(paths, rng)
            continue
        split_map = stratified_split(paths, rng)
        for split_name, split_paths in split_map.items():
            for p in split_paths:
                items.append({
                    "image_id": os.path.splitext(os.path.basename(p))[0],
                    "relative_path": os.path.relpath(p, DATASET_ROOT).replace("\\", "/"),
                    "sha256": sha256_file(p),
                    "class_name": cls,
                    "class_idx": class_to_idx[cls],
                    "split": split_name,
                    "source_dataset": "FINETUNE",
                    "source_group_id": "[Cần verify]",
                })

    manifest = {
        "schema_version": "split_manifest_v1",
        "created_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "seed": seed,
        "dataset_root": DATASET_ROOT,
        "species": species,
        "species_dir": os.path.relpath(species_dir, DATASET_ROOT).replace("\\", "/"),
        "split_ratio": RATIO,
        "class_to_idx": class_to_idx,
        "rare_class_threshold": rare_threshold,
        "items": items,
        "rare_folds": rare_folds,
    }
    os.makedirs(SPLITS_DIR, exist_ok=True)
    out_path = os.path.join(SPLITS_DIR, f"{species}_split_v1.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[{species}] {len(items)} ảnh thường + {len(rare_folds)} lớp hiếm -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Xây split manifest deterministic (Phase 0)")
    ap.add_argument("--species", default="all", help="'all' hoặc tên 1 loài")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rare-threshold", type=int, default=30,
                    help="lớp có < ngưỡng này -> tách sang rare_folds (few-shot Chế độ 2). "
                         "Đặt thấp (vd 5) để gộp lớp hiếm vào head thường.")
    args = ap.parse_args()
    targets = SPECIES if args.species == "all" else [args.species]
    for sp in targets:
        build_species(sp, args.seed, rare_threshold=args.rare_threshold)
