# finalize_candidates.py
"""
Hậu xử lý thư mục ONFIELD_LEAF_DETECTED và LAB_LEAF_DETECTED:
1. Xóa mọi thư mục bệnh trong LAB nếu không tồn tại trong ONFIELD.
2. Sau khi xóa, nếu một loài không có đủ cả healthy và disease
   (tính cả ONFIELD + LAB) thì xóa toàn bộ loài đó khỏi cả hai thư mục.
3. In log chi tiết.
"""

import shutil
from pathlib import Path

# ========== CẤU HÌNH ==========
DATA_ROOT = Path(r"D:\DAT301m_Project\data")
ONFIELD_DIR = DATA_ROOT / "ONFIELD_LEAF_DETECTED"
LAB_DIR = DATA_ROOT / "LAB_LEAF_DETECTED"

def prune_and_validate():
    if not ONFIELD_DIR.exists():
        print(f"Thư mục ONFIELD không tồn tại: {ONFIELD_DIR}")
        return
    if not LAB_DIR.exists():
        print(f"Thư mục LAB không tồn tại: {LAB_DIR}")
        return

    # Thu thập tất cả loài hiện có
    all_species = set()
    for d in [ONFIELD_DIR, LAB_DIR]:
        for sp_dir in d.iterdir():
            if sp_dir.is_dir():
                all_species.add(sp_dir.name)

    removed_label_count = 0

    # Bước 1: Xóa nhãn LAB không có trong ONFIELD
    print(">>> BƯỚC 1: Xóa nhãn LAB không có trong ONFIELD...")
    for sp in sorted(all_species):
        lab_sp = LAB_DIR / sp
        onfield_sp = ONFIELD_DIR / sp
        if not lab_sp.is_dir():
            continue

        onfield_labels = set()
        if onfield_sp.is_dir():
            onfield_labels = {d.name for d in onfield_sp.iterdir() if d.is_dir()}

        for label_dir in sorted(lab_sp.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            if label not in onfield_labels:
                shutil.rmtree(label_dir)
                removed_label_count += 1
                print(f"  [XÓA LAB] {sp}/{label}")

    # Bước 2: Xóa loài nếu không có đủ healthy + disease
    print("\n>>> BƯỚC 2: Xóa loài nếu không có đủ healthy + disease (tính cả ONFIELD + LAB)...")
    removed_species = []
    for sp in sorted(all_species):
        onfield_sp = ONFIELD_DIR / sp
        lab_sp = LAB_DIR / sp
        diseases = set()
        for base in [onfield_sp, lab_sp]:
            if base.is_dir():
                for label_dir in base.iterdir():
                    if label_dir.is_dir():
                        label = label_dir.name
                        if "___" in label:
                            disease = label.split("___", 1)[1].lower()
                            diseases.add(disease)
        has_healthy = "healthy" in diseases
        has_disease = any(d != "healthy" for d in diseases)
        if not (has_healthy and has_disease):
            for p in [onfield_sp, lab_sp]:
                if p.exists():
                    shutil.rmtree(p)
            reason = []
            if not has_healthy:
                reason.append("thiếu healthy")
            if not has_disease:
                reason.append("thiếu disease")
            removed_species.append(f"{sp} ({', '.join(reason)}, nhãn: {diseases})")
            print(f"  [XÓA LOÀI] {sp} - {', '.join(reason)} (nhãn: {diseases})")

    print("\n--- KẾT QUẢ ---")
    print(f"Đã xóa {removed_label_count} nhãn trong LAB.")
    if removed_species:
        print(f"Đã xóa {len(removed_species)} loài:")
        for s in removed_species:
            print(f"  - {s}")
    else:
        print("Tất cả các loài đều có đủ healthy + disease. Không xóa loài nào.")

if __name__ == "__main__":
    prune_and_validate()