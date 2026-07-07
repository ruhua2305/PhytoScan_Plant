import fiftyone as fo
import fiftyone.zoo as foz
import os

# Danh sách class hợp lệ (đã bỏ "Cup")
classes_of_interest = [
    "Flower", "Fruit", "Rose", "Lavender (Plant)", "Pineapple",
    "Watermelon", "Grape", "Pomegranate", "Strawberry", "Pumpkin",
    "Cantaloupe", "Zucchini", "Cucumber", "Bell pepper", "Tomato",
    "Potato", "Carrot", "Apple", "Orange", "Peach", "Pear",
    "Banana", "Mango", "Lemon",
    "Human hand", "Human arm", "Person", "Glove", "Handbag",
    "Table", "Chair", "Book", "Bookcase", "Computer keyboard",
    "Computer monitor", "Laptop", "Mobile phone", "Television",
    "Clock", "Door", "Window", "Bottle", "Coffee cup", "Mug",
    "Plate", "Bowl", "Knife", "Fork", "Spoon",
    "Tree", "Palm tree"
]

print("⏳ Đang tải ảnh...")
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    classes=classes_of_interest,
    max_samples=3500,
    shuffle=True,
    label_types=["classifications"],
)

print(f"✅ Đã tải {len(dataset)} ảnh.")

# --- Chia 80/10/10 bằng cách shuffle và cắt view ---
dataset.shuffle(seed=42)
total = len(dataset)
train_end = int(0.8 * total)
calib_end = int(0.9 * total)

train_view = dataset[:train_end]           # 80%
calib_view = dataset[train_end:calib_end]  # 10%
val_view = dataset[calib_end:]             # 10%

export_root = "D:/DAT301m_Project/Final_Pipeline/data/leaf_gate"
os.makedirs(os.path.join(export_root, "train", "not_leaf"), exist_ok=True)
os.makedirs(os.path.join(export_root, "calib", "not_leaf"), exist_ok=True)
os.makedirs(os.path.join(export_root, "val", "not_leaf"), exist_ok=True)

train_view.export(
    export_dir=os.path.join(export_root, "train", "not_leaf"),
    dataset_type=fo.types.ImageDirectory,
)
calib_view.export(
    export_dir=os.path.join(export_root, "calib", "not_leaf"),
    dataset_type=fo.types.ImageDirectory,
)
val_view.export(
    export_dir=os.path.join(export_root, "val", "not_leaf"),
    dataset_type=fo.types.ImageDirectory,
)

print("✅ Export thành công vào D:/DAT301m_Project/Final_Pipeline/data/leaf_gate")