# -*- coding: utf-8 -*-
"""Sinh sơ đồ slide: PREPROCESSING - GOM & ÁNH XẠ NHÃN."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'

# ----- palette -----
C_TITLE   = "#1b2a4a"
C_PROBLEM = "#c0392b"
C_STEP1   = "#2e86c1"
C_STEP2   = "#16a085"
C_STRAT   = "#d4efe6"
C_STRAT_E = "#0e6655"
C_SPECIAL = "#e67e22"
C_RESULT  = "#8e44ad"
C_TEXT    = "#f7f9fb"

fig, ax = plt.subplots(figsize=(15, 10.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('white')


def box(x, y, w, h, text, fc, ec=None, tc=C_TEXT, fs=12, weight='bold',
        radius=0.025, align='center'):
    ec = ec or fc
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.4,rounding_size={radius*100}",
                       fc=fc, ec=ec, lw=1.6, mutation_aspect=0.55, zorder=2)
    ax.add_patch(p)
    ha = {'center': 'center', 'left': 'left'}[align]
    tx = x + w / 2 if align == 'center' else x + 2
    ax.text(tx, y + h / 2, text, ha=ha, va='center', color=tc,
            fontsize=fs, fontweight=weight, zorder=3, wrap=True)


def arrow(x1, y1, x2, y2, color="#4a5568", lw=2.2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=22,
                        color=color, lw=lw, zorder=1,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ---------- Title ----------
box(3, 92, 94, 6.5, "PREPROCESSING  —  GOM & ÁNH XẠ NHÃN", C_TITLE, fs=20)

# ---------- Problem ----------
box(3, 82.5, 94, 6.5,
    "Vấn đề: 11 dataset — mỗi nguồn đặt tên lớp riêng, cấu trúc khác nhau (thư mục / CSV / bounding-box)",
    C_PROBLEM, fs=12.5)
arrow(50, 92, 50, 89)

# ---------- Step 1 ----------
box(3, 71.5, 94, 8.5,
    "BƯỚC 1 — Chọn loài mục tiêu\n"
    "Candidate = loài có CẢ lớp healthy VÀ lớp bệnh trong PlantVillage\n"
    "→ 5 loài:  apple · corn · grape · pepper_bell · tomato",
    C_STEP1, fs=12.5)
arrow(50, 82.5, 50, 80)

# ---------- Step 2 header ----------
box(3, 63.5, 94, 5.5,
    "BƯỚC 2 — 3 chiến lược ánh xạ nhãn về  Loài___Bệnh",
    C_STEP2, fs=14)
arrow(50, 71.5, 50, 69)

# ---------- 3 strategy columns ----------
sw = 29.5
gap = 2.75
x0 = 4
ys = 40
sh = 20
titles = [
    ("① Hard-coded dict",
     "PlantDoc · Tomato-Village\nNGLD · Merged\n\nĐộ tin cậy CAO"),
    ("② CSV / annotation",
     "Plant Pathology 2020-21\nPlantSeg · Master\n\nĐọc cột nhãn\nBỏ ảnh multi-label"),
    ("③ Keyword-based",
     "FieldPlant · Master\n\nTách loài (ghép 1-2 từ đầu)\n+ dò từ khóa bệnh\n(scab / rust / blight...)"),
]
for i, (t, body) in enumerate(titles):
    x = x0 + i * (sw + gap)
    box(x, ys, sw, sh, "", C_STRAT, ec=C_STRAT_E, radius=0.02)
    ax.text(x + sw / 2, ys + sh - 2.6, t, ha='center', va='center',
            color=C_STRAT_E, fontsize=13, fontweight='bold', zorder=3)
    ax.plot([x + 3, x + sw - 3], [ys + sh - 5.2, ys + sh - 5.2],
            color=C_STRAT_E, lw=1.1, zorder=3)
    ax.text(x + sw / 2, ys + (sh - 6) / 2, body, ha='center', va='center',
            color="#0b3d34", fontsize=11.5, zorder=3)
    arrow(x + sw / 2, 63.5, x + sw / 2, ys + sh)

# ---------- Special handling ----------
box(3, 27.5, 94, 9.5,
    "XỬ LÝ ĐẶC BIỆT\n"
    "• FieldPlant: crop theo bounding-box trong CSV  →  resize 224×224\n"
    "• Lớp không khớp PlantVillage-38  →  in cảnh báo & bỏ qua",
    C_SPECIAL, fs=12.5, align='center')
for i in range(3):
    x = x0 + i * (sw + gap) + sw / 2
    arrow(x, 40, x, 37)

# ---------- Result ----------
box(3, 15, 94, 9,
    "KẾT QUẢ — gom về 2 nhánh:   Loài   /   Loài___Bệnh\n"
    "(chờ Stage 2 lọc lab / field)",
    C_RESULT, fs=14)
arrow(50, 27.5, 50, 24)

# ---------- footer ----------
ax.text(50, 9, "DAT301m · PhytoScan — Phase 2 Preprocessing pipeline",
        ha='center', va='center', color="#7b8794", fontsize=10.5,
        fontstyle='italic')

plt.tight_layout()
fig.savefig("d:/DAT301m_Project/Final_Pipeline/slide_preprocessing.png",
            dpi=200, bbox_inches='tight', facecolor='white')
print("saved slide_preprocessing.png")
