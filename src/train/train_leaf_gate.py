"""Train Leaf Gate nhị phân (MobileNetV3-Small, Mục 3.2). Chạy 1 lần, độc lập 5 loài.
Mục tiêu: leaf recall >= 98% trên lá field hợp lệ, non-leaf rejection >= 95%.
Chạy: python -m src.train.train_leaf_gate --config configs/leaf_gate.yaml
"""
import argparse
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score

from src.config import TrainConfig
from src.models.backbone import build_leaf_gate
from src.data.augmentation import get_finetune_transforms
from src.train.memory_utils import train_one_epoch_memory_efficient, probe_vram_usage
from src.train.train_pretrain import set_seed

IMG_EXT = (".jpg", ".jpeg", ".png")
CLASS_TO_IDX = {"not_leaf": 0, "leaf": 1}   # index 1 = leaf (khớp app.py: softmax[...,1])


class LeafGateDataset(Dataset):
    """Đọc {split}/{leaf,not_leaf}/*. Nhãn nhị phân: leaf=1, not_leaf=0."""
    def __init__(self, root_split_dir, transform=None):
        self.transform = transform
        self.samples = []
        for cls, idx in CLASS_TO_IDX.items():
            cdir = os.path.join(root_split_dir, cls)
            if not os.path.isdir(cdir):
                continue
            for f in os.listdir(cdir):
                if f.lower().endswith(IMG_EXT):
                    self.samples.append((os.path.join(cdir, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label


@torch.no_grad()
def eval_leaf_gate(model, loader, device):
    """Trả (macro_f1, leaf_recall, notleaf_recall) — theo dõi đúng 2 chỉ tiêu Mục 3.2."""
    model.eval()
    yt, yp = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        yp.extend(logits.argmax(1).cpu().tolist())
        yt.extend(labels.tolist())
    macro = f1_score(yt, yp, average="macro", zero_division=0)
    leaf_rec = recall_score(yt, yp, pos_label=1, zero_division=0)
    notleaf_rec = recall_score(yt, yp, pos_label=0, zero_division=0)
    return macro, leaf_rec, notleaf_rec


def train_leaf_gate(config_path):
    cfg = TrainConfig.load(config_path)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, t = cfg["data"], cfg["train"]
    train_tf = get_finetune_transforms(True, d["img_size"])
    val_tf = get_finetune_transforms(False, d["img_size"])
    train_ds = LeafGateDataset(os.path.join(d["root"], "train"), train_tf)
    val_ds = LeafGateDataset(os.path.join(d["root"], "val"), val_tf)

    def _check(ds, split):
        if len(ds) == 0:
            raise RuntimeError(
                f"Dataset leaf-gate split '{split}' RỖNG tại {os.path.join(d['root'], split)}.\n"
                f"  Bạn cần dựng dataset trước (xem README mục 3):\n"
                f"    1) python -m src.data.build_leaf_gate_dataset --per-species 700 --seed 42\n"
                f"    2) tự thêm ảnh not_leaf vào {os.path.join(d['root'], split, 'not_leaf')}\\\n"
                f"  Cấu trúc cần: {d['root']}\\{{train,calib,val}}\\{{leaf,not_leaf}}\\*.jpg")
        n_leaf = sum(1 for _, y in ds.samples if y == 1)
        n_not = len(ds) - n_leaf
        print(f"[leaf_gate] {split}: {len(ds)} ảnh (leaf={n_leaf}, not_leaf={n_not})")
        if n_leaf == 0 or n_not == 0:
            raise RuntimeError(f"Split '{split}' thiếu MỘT lớp (leaf={n_leaf}, not_leaf={n_not}). "
                               f"Cần đủ cả hai lớp để train nhị phân.")
    _check(train_ds, "train")
    _check(val_ds, "val")

    tl = DataLoader(train_ds, t["batch_size"], shuffle=True,
                    num_workers=d["num_workers"], pin_memory=True)
    vl = DataLoader(val_ds, t["batch_size"], shuffle=False,
                    num_workers=d["num_workers"], pin_memory=True)

    model = build_leaf_gate(pretrained=True).to(device)
    sample, _ = next(iter(tl))
    if probe_vram_usage(model, sample, device, batch_sizes=(t["batch_size"],))[t["batch_size"]] == "OOM":
        raise RuntimeError("OOM leaf-gate — giảm batch_size.")

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    # Cân bằng leaf/not-leaf bằng class weight (thường không quá lệch)
    crit = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.05))
    out = cfg["checkpoint"]["out_path"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    best = 0.0
    for ep in range(t["epochs"]):
        loss = train_one_epoch_memory_efficient(model, tl, opt, crit, device,
                t["accumulation_steps"], t["use_amp"])
        macro, leaf_rec, nl_rec = eval_leaf_gate(model, vl, device)
        print(f"[leaf_gate] ep {ep+1}/{t['epochs']} loss={loss:.4f} "
              f"macro_f1={macro:.4f} leaf_recall={leaf_rec:.4f} notleaf_recall={nl_rec:.4f}")
        if macro > best:
            best = macro
            torch.save({"model_state": model.state_dict(), "arch": "mobilenetv3_small_100",
                        "class_to_idx": CLASS_TO_IDX, "num_classes": 2, "val_macro_f1": best}, out)
            print(f"  -> saved {out} (macro-F1={best:.4f})")
    print(f"[leaf_gate] done — best macro-F1: {best:.4f}. "
          f"Tiếp theo: calibrate (Mục 5.3) trên calib split, chọn ngưỡng từ PR curve.")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/leaf_gate.yaml")
    train_leaf_gate(ap.parse_args().config)
