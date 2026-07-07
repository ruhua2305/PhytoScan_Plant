"""Pretrain backbone 38 lớp (Phase 1). Chọn checkpoint theo macro-F1.
Chạy: python -m src.train.train_pretrain --config configs/pretrain_effv2s.yaml
"""
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.models.backbone import build_backbone
from src.data.datasets import PretrainDataset
from src.data.augmentation import get_pretrain_transforms, CutMixMixupCollate
from src.data.domain_augmentation import FieldBackgroundAugment
from src.train.memory_utils import train_one_epoch_memory_efficient, probe_vram_usage
from src.eval.evaluate import evaluate


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def soft_target_cross_entropy(logits, soft_targets, label_smoothing=0.0):
    """CE cho nhãn phân phối (khi CutMix/Mixup bật). Tự viết để label_smoothing tường minh."""
    n = logits.size(1)
    logp = torch.log_softmax(logits, dim=1)
    if label_smoothing > 0:
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / n
    return -(soft_targets * logp).sum(dim=1).mean()


def list_images_recursive(root):
    out = []
    for dp, _, fs in os.walk(root):
        out += [os.path.join(dp, f) for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return out


def build_dataloaders(cfg):
    d = cfg["data"]
    train_tf = get_pretrain_transforms(True, d["img_size"])
    val_tf = get_pretrain_transforms(False, d["img_size"])
    dom = cfg.raw.get("domain_aug", {})
    field_aug = None
    if dom.get("enabled", False):
        bg = list_images_recursive(dom["field_background_dir"])
        field_aug = FieldBackgroundAugment(bg, p=dom.get("p", 0.4))
        print(f"[domain_aug] {len(bg)} nền field, p={dom.get('p', 0.4)}")
    train_ds = PretrainDataset(d["train_dir"], train_tf, domain_aug=field_aug)
    val_ds = PretrainDataset(d["val_dir"], val_tf, domain_aug=None)
    cm = cfg.raw.get("cutmix_mixup", {})
    collate = CutMixMixupCollate(cfg["model"]["num_classes"], cm.get("alpha", 0.2),
                                 cm.get("cutmix_prob", 0.5), cm.get("p", 0.5)) if cm.get("enabled") else None
    tl = DataLoader(train_ds, cfg["train"]["batch_size"], shuffle=True, num_workers=d["num_workers"],
                    pin_memory=True, persistent_workers=False, collate_fn=collate)
    vl = DataLoader(val_ds, cfg["train"]["batch_size"], shuffle=False,
                    num_workers=d["num_workers"], pin_memory=True)
    return tl, vl, collate is not None


def train_pretrain(config_path):
    cfg = TrainConfig.load(config_path)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tl, vl, soft = build_dataloaders(cfg)
    model = build_backbone(cfg["model"]["num_classes"], cfg["model"]["arch"],
                           drop_path_rate=cfg["model"]["drop_path_rate"]).to(device)

    # BƯỚC BẮT BUỘC — probe VRAM (Mục 4.5)
    sample, _ = next(iter(tl))
    bs = cfg["train"]["batch_size"]
    rep = probe_vram_usage(model, sample, device, batch_sizes=(bs,))
    print(f"[probe_vram] bs={bs}: {rep[bs]}")
    if rep[bs] == "OOM":
        raise RuntimeError(f"OOM ở bs={bs}. Giảm batch_size hoặc tăng accumulation_steps trong {config_path}.")

    t = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr_head"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    if soft:
        crit = lambda lo, ta: soft_target_cross_entropy(lo, ta, t["label_smoothing"])
    else:
        crit = torch.nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])

    out = cfg["checkpoint"]["out_path"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    best = 0.0
    for ep in range(t["epochs"]):
        loss = train_one_epoch_memory_efficient(model, tl, opt, crit, device,
                t["accumulation_steps"], t["use_amp"], t["use_grad_checkpoint"])
        sched.step()
        f1 = evaluate(model, vl, device)
        print(f"[pretrain] ep {ep+1}/{t['epochs']} loss={loss:.4f} val_macro_f1={f1:.4f}")
        if f1 > best:
            best = f1
            torch.save({"model_state": model.state_dict(), "arch": cfg["model"]["arch"],
                        "num_classes": cfg["model"]["num_classes"], "val_macro_f1": best}, out)
            print(f"  -> saved {out} (macro-F1={best:.4f})")
    print(f"[pretrain] done — best val macro-F1: {best:.4f}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_effv2s.yaml")
    train_pretrain(ap.parse_args().config)
