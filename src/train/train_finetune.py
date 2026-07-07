"""Fine-tune 1 loài — Chế độ 1 (Bước 1->2->3) + Chế độ 2 (few-shot).
Chạy: python -m src.train.train_finetune --config configs/finetune_tomato.yaml
"""
import argparse
import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.models.backbone import build_backbone
from src.data.datasets import FinetuneDataset, PretrainDataset
from src.data.augmentation import get_finetune_transforms, get_pretrain_transforms, CutMixMixupCollate
from src.train.losses import effective_num_weights, ClassBalancedFocalLossSoft
from src.train.samplers import build_balanced_sampler
from src.train.peft import set_trainable_params
from src.train.few_shot import compute_prototypes, classify_by_prototype
from src.train.domain_loss import mmd_loss
from src.train.memory_utils import probe_vram_usage
from src.train.train_pretrain import set_seed
from src.eval.evaluate import evaluate


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pretrained_backbone(model, ckpt_path, device):
    """Nạp backbone pretrain 38 lớp, BỎ head (số lớp khác). Lọc theo tên+shape."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state, msd = ckpt["model_state"], model.state_dict()
    filtered = {k: v for k, v in state.items() if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(filtered, strict=False)
    print(f"  [load_pretrained] nạp {len(filtered)}/{len(state)} tensor (head khởi tạo mới)")
    return model


def build_optimizer(model, lr_head, lr_backbone, weight_decay):
    """Discriminative LR: head nhanh hơn backbone vừa mở khoá."""
    head = [p for n, p in model.named_parameters()
            if p.requires_grad and ("classifier" in n or "fc" in n)]
    back = [p for n, p in model.named_parameters()
            if p.requires_grad and not ("classifier" in n or "fc" in n)]
    groups = [{"params": head, "lr": lr_head}]
    if back:
        groups.append({"params": back, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_regular_dataloaders(cfg, manifest):
    d = cfg["data"]
    train_ds = FinetuneDataset(manifest, "train", get_finetune_transforms(True, d["img_size"]))
    val_ds = FinetuneDataset(manifest, "val", get_finetune_transforms(False, d["img_size"]))
    idx_to_class = {v: k for k, v in manifest["class_to_idx"].items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    counts = [sum(1 for _, y in train_ds.samples if y == i) for i in range(len(classes))]
    cm = cfg.raw.get("train", {}).get("cutmix_mixup", {})
    collate = CutMixMixupCollate(len(classes), cm.get("alpha", 0.2),
                                 cm.get("cutmix_prob", 0.5), cm.get("p", 0.5)) if cm.get("enabled") else None
    tl = DataLoader(train_ds, cfg["train"]["batch_size"], shuffle=True, num_workers=d["num_workers"],
                    pin_memory=True, persistent_workers=False, collate_fn=collate)
    vl = DataLoader(val_ds, cfg["train"]["batch_size"], shuffle=False,
                    num_workers=d["num_workers"], pin_memory=True)
    return tl, vl, train_ds, classes, counts


def build_mmd_source_loader(source_dir, img_size, batch_size, num_workers):
    ds = PretrainDataset(source_dir, get_pretrain_transforms(False, img_size), domain_aug=None)
    return DataLoader(ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


def train_one_epoch_finetune(model, loader, opt, crit, device, accumulation_steps=4,
                             use_amp=True, mmd_source_loader=None, lambda_mmd=0.0):
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train(); opt.zero_grad(set_to_none=True)
    run_loss = 0.0
    mmd_iter = iter(mmd_source_loader) if mmd_source_loader is not None else None
    for step, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            logits = model(imgs)
            loss = crit(logits, targets)
            if mmd_iter is not None and lambda_mmd > 0:
                try:
                    src, _ = next(mmd_iter)
                except StopIteration:
                    mmd_iter = iter(mmd_source_loader); src, _ = next(mmd_iter)
                src = src.to(device, non_blocking=True)
                fs = model.forward_head(model.forward_features(src), pre_logits=True)
                ft = model.forward_head(model.forward_features(imgs), pre_logits=True)
                loss = loss + lambda_mmd * mmd_loss(fs, ft)
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        run_loss += loss.item() * accumulation_steps
    return run_loss / len(loader)


def _train_phase(model, tl, vl, opt, crit, device, *, epochs, accumulation_steps, use_amp,
                 best_f1, ckpt_meta, ckpt_path, phase, mmd_loader=None, lambda_max=0.0, mmd_warmup=0,
                 patience=0):
    """Chạy 1 pha, lưu checkpoint tốt nhất XUYÊN SUỐT 3 bước (best_f1 chuyền tiếp).
    patience>0: dừng sớm nếu val F1 không cải thiện sau `patience` epoch liên tiếp (xét trong pha)."""
    best_in_phase = -1.0
    stall = 0
    for ep in range(epochs):
        lam = lambda_max * min(1.0, (ep + 1) / max(1, mmd_warmup)) if (mmd_loader and lambda_max > 0) else 0.0
        loss = train_one_epoch_finetune(model, tl, opt, crit, device, accumulation_steps,
                                        use_amp, mmd_loader, lam)
        f1 = evaluate(model, vl, device)
        msg = f"[{phase}] ep {ep+1}/{epochs} loss={loss:.4f} val_macro_f1={f1:.4f}"
        if lam > 0:
            msg += f" lambda_mmd={lam:.3f}"
        print(msg)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({**ckpt_meta, "model_state": model.state_dict(), "val_macro_f1": best_f1}, ckpt_path)
            print(f"    -> saved ({phase}) {ckpt_path} (macro-F1={best_f1:.4f})")
        # early stopping xét theo cải thiện TRONG pha (dùng epsilon để bỏ qua nhiễu nhỏ)
        if f1 > best_in_phase + 1e-4:
            best_in_phase = f1
            stall = 0
        else:
            stall += 1
            if patience and stall >= patience:
                print(f"    -> early stop ({phase}): {patience} epoch không cải thiện (best pha={best_in_phase:.4f})")
                break
    return best_f1


@torch.no_grad()
def extract_embeddings(model, items, transform, device, batch_size=16):
    """Trích embedding pre-logits theo batch (tránh tràn VRAM).
    items: list (path, label_idx). forward_head(pre_logits=True) -> vector ĐÃ pooled."""
    model.eval()
    feats, labels = [], []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        imgs = []
        for path, _ in chunk:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            imgs.append(transform(image=img)["image"])
        batch = torch.stack(imgs).to(device)
        emb = model.forward_head(model.forward_features(batch), pre_logits=True)
        feats.append(emb.float().cpu())
        labels.extend(y for _, y in chunk)
    return torch.cat(feats), torch.tensor(labels)


def run_few_shot_evaluation(cfg, model, rare_folds, device, ckpt_path):
    """Chế độ 2: dùng backbone TỐT NHẤT (Chế độ 1) để trích embedding, đánh giá K-fold
    (mỗi ảnh làm query đúng 1 lần). Cuối cùng lưu prototype trên TOÀN BỘ ảnh hiếm để suy luận."""
    species = cfg["species"]
    print(f"\n===== [{species}] CHẾ ĐỘ 2: few-shot prototype (Snell et al., 2017) =====")
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"], strict=False); model.eval()

    rare_classes = sorted(rare_folds.keys())
    cls_to_idx = {c: i for i, c in enumerate(rare_classes)}
    num_rare = len(rare_classes)
    val_tf = get_finetune_transforms(False, cfg["data"]["img_size"])

    items = [(fold, path, cls_to_idx[cls])
             for cls, entries in rare_folds.items() for fold, path in entries]
    if not items:
        print(f"[{species}] rare_folds rỗng — bỏ qua."); return
    n_folds = max(f for f, _, _ in items) + 1

    fold_acc, fold_f1 = [], []
    for f in range(n_folds):
        support = [(p, y) for fi, p, y in items if fi != f]
        query = [(p, y) for fi, p, y in items if fi == f]
        if not support or not query:
            continue
        sf, sl = extract_embeddings(model, support, val_tf, device)
        qf, ql = extract_embeddings(model, query, val_tf, device)
        protos = compute_prototypes(sf, sl, num_rare)
        preds = classify_by_prototype(qf, protos.to(qf.device)).argmax(dim=1)
        fold_acc.append((preds == ql).float().mean().item())
        fold_f1.append(f1_score(ql.numpy(), preds.numpy(), average="macro", zero_division=0))
        print(f"  fold {f}: n_query={len(query)} acc={fold_acc[-1]:.4f} macro_f1={fold_f1[-1]:.4f}")
    if fold_acc:
        print(f"[{species}] Chế độ 2 ({len(fold_acc)} fold): "
              f"acc={np.mean(fold_acc):.4f}±{np.std(fold_acc):.4f} "
              f"macro_f1={np.mean(fold_f1):.4f}±{np.std(fold_f1):.4f}")

    # Prototype CUỐI: dùng toàn bộ ảnh hiếm (để suy luận thật)
    all_items = [(p, y) for _, p, y in items]
    af, al = extract_embeddings(model, all_items, val_tf, device)
    final_protos = compute_prototypes(af, al, num_rare)
    proto_path = ckpt_path.replace(".pt", "_prototypes.pt")
    torch.save({"prototypes": final_protos, "rare_classes": rare_classes,
                "class_to_idx": cls_to_idx, "species": species,
                "embed_dim": final_protos.size(1)}, proto_path)
    print(f"[{species}] Lưu {num_rare} prototype -> {proto_path}")


def train_finetune(config_path):
    cfg = TrainConfig.load(config_path)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    species, t = cfg["species"], cfg["train"]
    manifest = load_manifest(cfg["data"]["split_manifest"])

    # ===== CHẾ ĐỘ 1 =====
    tl, vl, train_ds, classes, counts = build_regular_dataloaders(cfg, manifest)
    print(f"[{species}] Chế độ 1 — {len(classes)} lớp thường: {dict(zip(classes, counts))}")
    model = build_backbone(len(classes), cfg["model"]["arch"], drop_path_rate=0.1).to(device)
    model = load_pretrained_backbone(model, cfg["model"]["pretrained_ckpt"], device)

    sample, _ = next(iter(tl)); bs = t["batch_size"]
    rep = probe_vram_usage(model, sample, device, batch_sizes=(bs,))
    print(f"[probe_vram] bs={bs}: {rep[bs]}")
    if rep[bs] == "OOM":
        raise RuntimeError(f"OOM ở bs={bs}. Giảm batch_size / tăng accumulation_steps trong {config_path}.")

    cb = t["class_balanced_loss"]
    crit_cb = ClassBalancedFocalLossSoft(counts, beta=cb["beta"], gamma=cb["gamma"]).to(device)
    ckpt_path = cfg["checkpoint"]["out_path"]
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    meta = {"arch": cfg["model"]["arch"], "num_classes": len(classes), "classes": classes, "species": species}
    best = 0.0

    # Bước 1 — warm-up head
    print(f"\n===== [{species}] BƯỚC 1: head_only =====")
    set_trainable_params(model, "head_only")
    opt = build_optimizer(model, t["lr_head"], t["lr_backbone"], t["weight_decay"])
    best = _train_phase(model, tl, vl, opt, crit_cb, device, epochs=t["epochs_warmup"],
                        accumulation_steps=t["accumulation_steps"], use_amp=t["use_amp"],
                        best_f1=best, ckpt_meta=meta, ckpt_path=ckpt_path, phase="warmup")

    # Bước 2 — head_bn_lastblock + CB-Focal + CutMix + MMD tuỳ chọn
    print(f"\n===== [{species}] BƯỚC 2: head_bn_lastblock =====")
    set_trainable_params(model, "head_bn_lastblock")
    opt = build_optimizer(model, t["lr_head"], t["lr_backbone"], t["weight_decay"])
    mmd_loader, lam_max, warm = None, 0.0, 0
    mcfg = t.get("domain_mmd", {})
    if mcfg.get("enabled"):
        mmd_loader = build_mmd_source_loader(mcfg["source_dir"], cfg["data"]["img_size"],
                                             t["batch_size"], cfg["data"]["num_workers"])
        lam_max, warm = mcfg["lambda_max"], mcfg["warmup_epochs"]
        print(f"  [MMD] bật: lambda_max={lam_max}, warmup={warm}")
    best = _train_phase(model, tl, vl, opt, crit_cb, device, epochs=t["epochs_full"],
                        accumulation_steps=t["accumulation_steps"], use_amp=t["use_amp"],
                        best_f1=best, ckpt_meta=meta, ckpt_path=ckpt_path, phase="full",
                        mmd_loader=mmd_loader, lambda_max=lam_max, mmd_warmup=warm,
                        patience=t.get("early_stop_patience", 0))

    # Bước 3 — decoupled re-training (backbone đóng băng, sampler cân bằng, KHÔNG CutMix)
    print(f"\n===== [{species}] BƯỚC 3: decoupled re-training =====")
    set_trainable_params(model, "head_only")
    opt = build_optimizer(model, t["lr_head"], t["lr_backbone"], t["weight_decay"])
    bal_ds = FinetuneDataset(manifest, "train", get_finetune_transforms(True, cfg["data"]["img_size"]))
    sampler = build_balanced_sampler(bal_ds.samples, counts, beta=cb["beta"])
    bal_loader = DataLoader(bal_ds, t["batch_size"], sampler=sampler,
                            num_workers=cfg["data"]["num_workers"], pin_memory=True)
    # Decoupled (Kang et al., 2019): balanced sampler ĐI VỚI CE THƯỜNG (KHÔNG class weight).
    # Dùng cả hai = cân bằng gấp đôi -> head lệch mạnh về lớp thiểu số, F1 tụt.
    crit_ce = nn.CrossEntropyLoss()
    best = _train_phase(model, bal_loader, vl, opt, crit_ce, device, epochs=t["epochs_decoupled"],
                        accumulation_steps=t["accumulation_steps"], use_amp=t["use_amp"],
                        best_f1=best, ckpt_meta=meta, ckpt_path=ckpt_path, phase="decoupled")
    print(f"\n[{species}] Chế độ 1 DONE — best macro-F1: {best:.4f}")

    # ===== CHẾ ĐỘ 2 =====
    rare_folds = manifest.get("rare_folds", {})
    if rare_folds:
        run_few_shot_evaluation(cfg, model, rare_folds, device, ckpt_path)
    else:
        print(f"[{species}] không có lớp hiếm — bỏ qua Chế độ 2.")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    train_finetune(ap.parse_args().config)
