"""Dựng phần POSITIVE (leaf) cho dataset Leaf Gate từ manifest split (Mục 4.12).

- Chỉ lấy ảnh thuộc split 'train' của từng loài -> KHÔNG rò rỉ holdout/val/calib.
- Cân bằng theo loài (mỗi loài lấy tối đa `per_species`), rải đều lớp trong loài.
- Chia 80/10/10 -> Final_Pipeline/data/leaf_gate/{train,calib,val}/leaf/.
- Phần not_leaf bạn tự thêm vào {train,calib,val}/not_leaf/ (xem README).

Chạy: python -m src.data.build_leaf_gate_dataset --per-species 700 --seed 42
"""
import argparse
import json
import os
import random
import shutil

SPECIES = ["apple", "corn", "grape", "pepper_bell", "tomato"]
SPLITS_DIR = r"D:/DAT301m_Project/Final_Pipeline/data/splits"
LEAF_GATE_ROOT = r"D:/DAT301m_Project/Final_Pipeline/data/leaf_gate"
SUBSPLIT_RATIO = {"train": 0.8, "calib": 0.1, "val": 0.1}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _load_manifest(species):
    path = os.path.join(SPLITS_DIR, f"{species}_split_v1.json")
    if not os.path.exists(path):
        print(f"  [WARN] chưa có {path} — chạy split_builder trước. Bỏ qua {species}.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_train_leaf_images(manifest, per_species, rng):
    """Lấy ảnh split=train, rải đều theo lớp, tối đa per_species ảnh/loài."""
    root = manifest["dataset_root"]
    by_class = {}
    for it in manifest["items"]:
        if it["split"] != "train":
            continue
        by_class.setdefault(it["class_name"], []).append(os.path.join(root, it["relative_path"]))
    for paths in by_class.values():
        rng.shuffle(paths)
    # round-robin qua các lớp để cân bằng
    selected, classes = [], list(by_class.keys())
    idx = {c: 0 for c in classes}
    while len(selected) < per_species and any(idx[c] < len(by_class[c]) for c in classes):
        for c in classes:
            if idx[c] < len(by_class[c]):
                selected.append(by_class[c][idx[c]]); idx[c] += 1
                if len(selected) >= per_species:
                    break
    return selected


def _split_and_copy(paths, rng, dry_run, prefix="leaf"):
    rng.shuffle(paths)
    n = len(paths)
    n_tr = int(n * SUBSPLIT_RATIO["train"])
    n_ca = int(n * SUBSPLIT_RATIO["calib"])
    buckets = {"train": paths[:n_tr], "calib": paths[n_tr:n_tr + n_ca], "val": paths[n_tr + n_ca:]}
    counts = {}
    for split, ps in buckets.items():
        out_dir = os.path.join(LEAF_GATE_ROOT, split, "leaf")
        if not dry_run:
            os.makedirs(out_dir, exist_ok=True)
        for i, p in enumerate(ps):
            # prefix (tên loài) TRÁNH tên file trùng giữa các loài -> không ghi đè
            dst = os.path.join(out_dir, f"leaf_{prefix}_{split}_{i:05d}{os.path.splitext(p)[1].lower()}")
            if not dry_run:
                shutil.copy2(p, dst)
        counts[split] = len(ps)
    return counts


def _split_not_leaf(src_dir, rng, dry_run):
    """Gom MỌI ảnh (đệ quy) trong src_dir -> chia 80/10/10 vào <split>/not_leaf/.
    Cho phép src_dir có nhiều thư mục con theo nhóm (tay, đất, đồ vật...)."""
    paths = []
    for dp, _, fs in os.walk(src_dir):
        paths += [os.path.join(dp, f) for f in fs if f.lower().endswith(IMG_EXT)]
    if not paths:
        print(f"  [WARN] không thấy ảnh not_leaf trong {src_dir}"); return {"train": 0, "calib": 0, "val": 0}
    rng.shuffle(paths)
    n = len(paths)
    n_tr = int(n * SUBSPLIT_RATIO["train"])
    n_ca = int(n * SUBSPLIT_RATIO["calib"])
    buckets = {"train": paths[:n_tr], "calib": paths[n_tr:n_tr + n_ca], "val": paths[n_tr + n_ca:]}
    counts = {}
    for split, ps in buckets.items():
        out_dir = os.path.join(LEAF_GATE_ROOT, split, "not_leaf")
        if not dry_run:
            os.makedirs(out_dir, exist_ok=True)
        for i, p in enumerate(ps):
            dst = os.path.join(out_dir, f"notleaf_{split}_{i:05d}{os.path.splitext(p)[1].lower()}")
            if not dry_run:
                shutil.copy2(p, dst)
        counts[split] = len(ps)
    return counts


def build(per_species, seed, dry_run, not_leaf_src=None):
    rng = random.Random(seed)
    total = {"train": 0, "calib": 0, "val": 0}
    for sp in SPECIES:
        man = _load_manifest(sp)
        if not man:
            continue
        imgs = _collect_train_leaf_images(man, per_species, rng)
        counts = _split_and_copy(imgs, rng, dry_run, prefix=sp)
        for k in total:
            total[k] += counts[k]
        print(f"[{sp}] leaf chọn {sum(counts.values())} ảnh -> {counts}")
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}TỔNG leaf: {total} "
          f"(= {sum(total.values())} ảnh) tại {LEAF_GATE_ROOT}/<split>/leaf/")

    if not_leaf_src:
        nl = _split_not_leaf(not_leaf_src, rng, dry_run)
        print(f"{'[DRY-RUN] ' if dry_run else ''}TỔNG not_leaf: {nl} "
              f"(= {sum(nl.values())} ảnh) tại {LEAF_GATE_ROOT}/<split>/not_leaf/")
        for split in total:
            leaf_n, nl_n = total[split], nl[split]
            ratio = f"1:{nl_n/leaf_n:.2f}" if leaf_n else "n/a"
            print(f"  [{split}] leaf:{leaf_n}  not_leaf:{nl_n}  (tỷ lệ leaf:not_leaf = {ratio})")
    else:
        print("Tiếp theo: tự thêm ảnh not_leaf vào <split>/not_leaf/, "
              "hoặc chạy lại với --not-leaf-src <thư_mục_ảnh_not_leaf>.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dựng dataset Leaf Gate (leaf tự trích + not_leaf tuỳ chọn)")
    ap.add_argument("--per-species", type=int, default=700, help="số ảnh leaf tối đa mỗi loài")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="chỉ đếm, không copy file")
    ap.add_argument("--not-leaf-src", default=None,
                    help="thư mục chứa ảnh not_leaf (đệ quy) để tự chia 80/10/10")
    a = ap.parse_args()
    build(a.per_species, a.seed, a.dry_run, a.not_leaf_src)
