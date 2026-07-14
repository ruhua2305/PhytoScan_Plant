# -*- coding: utf-8 -*-
"""Bảng thống kê 11 raw dataset cho slide."""
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'

C_HEAD = "#1b2a4a"
C_ROW_A = "#eef2f7"
C_ROW_B = "#dbe4ee"
C_TOTAL = "#16a085"
C_TXT = "#1b2a4a"

# (dataset, số ảnh, kiểu định dạng, nhóm format màu)
rows = [
    ("Plant Disease Classification Merged Dataset", "79,086", "Thư mục theo lớp (ImageFolder)"),
    ("Master Plant Disease Dataset",               "66,953", "Ảnh phẳng + CSV manifest"),
    ("PlantVillage",                               "54,305", "Thư mục theo lớp (train/val)"),
    ("Tomato-Village-main",                        "24,546", "Hỗn hợp: lớp + CSV + bbox"),
    ("plantseg",                                   "22,916", "Ảnh + CSV + mask/JSON (seg)"),
    ("Plant Pathology 2021",                       "18,635", "Ảnh phẳng + CSV nhãn"),
    ("FieldPlant",                                  "5,156", "Ảnh phẳng + CSV bounding-box"),
    ("Plant Pathology 2020",                        "3,642", "Ảnh phẳng + CSV nhãn (one-hot)"),
    ("PlantDoc",                                    "2,922", "Thư mục theo lớp (train/test)"),
    ("Niphad Grape Leaf (NGLD)",                    "2,726", "Thư mục theo lớp"),
    ("TomatoDataset",                               "1,609", "Thư mục theo lớp (phẳng)"),
]
TOTAL = "282,496"

headers = ["#", "Dataset", "Số ảnh", "Kiểu định dạng data"]
col_x = [0.03, 0.09, 0.62, 0.74]   # left edges (fig fraction)
col_w = [0.06, 0.53, 0.12, 0.24]
row_h = 0.058
top = 0.84

fig, ax = plt.subplots(figsize=(14, 8.6))
ax.axis('off')
fig.patch.set_facecolor('white')

# title
ax.text(0.5, 0.93, "RAW DATASET — THỐNG KÊ NGUỒN (11 dataset thô)",
        ha='center', va='center', fontsize=20, fontweight='bold', color=C_HEAD,
        transform=ax.transAxes)

def cell(x, y, w, text, fc, tc, fs, weight, align='left'):
    ax.add_patch(plt.Rectangle((x, y), w, row_h, transform=ax.transAxes,
                               fc=fc, ec='white', lw=1.4, zorder=1))
    if align == 'left':
        tx, ha = x + 0.008, 'left'
    elif align == 'right':
        tx, ha = x + w - 0.012, 'right'
    else:
        tx, ha = x + w / 2, 'center'
    ax.text(tx, y + row_h / 2, text, ha=ha, va='center', color=tc,
            fontsize=fs, fontweight=weight, transform=ax.transAxes, zorder=2)

# header row
for j, h in enumerate(headers):
    al = 'center' if j in (0, 2) else 'left'
    cell(col_x[j], top, col_w[j], h, C_HEAD, 'white', 13, 'bold', al)

# data rows
for i, (name, cnt, fmt) in enumerate(rows):
    y = top - (i + 1) * row_h
    bg = C_ROW_A if i % 2 == 0 else C_ROW_B
    cell(col_x[0], y, col_w[0], str(i + 1), bg, C_TXT, 12, 'bold', 'center')
    cell(col_x[1], y, col_w[1], name,        bg, C_TXT, 12, 'normal', 'left')
    cell(col_x[2], y, col_w[2], cnt,         bg, C_TXT, 12, 'bold', 'right')
    cell(col_x[3], y, col_w[3], fmt,         bg, "#33475b", 11, 'normal', 'left')

# total row
y = top - (len(rows) + 1) * row_h
cell(col_x[0], y, col_w[0], "", C_TOTAL, 'white', 12, 'bold', 'center')
cell(col_x[1], y, col_w[1], "TỔNG", C_TOTAL, 'white', 13, 'bold', 'left')
cell(col_x[2], y, col_w[2], TOTAL, C_TOTAL, 'white', 13, 'bold', 'right')
cell(col_x[3], y, col_w[3], "3 kiểu: folder / CSV / bbox", C_TOTAL, 'white', 11, 'bold', 'left')

ax.text(0.5, y - 0.03,
        "Nguồn: raw_data_processing/raw_datasets_structure.txt  ·  DAT301m · PhytoScan",
        ha='center', va='center', fontsize=10, fontstyle='italic', color="#7b8794",
        transform=ax.transAxes)

fig.savefig("d:/DAT301m_Project/Final_Pipeline/slide_raw_datasets.png",
            dpi=200, bbox_inches='tight', facecolor='white')
print("saved slide_raw_datasets.png")
