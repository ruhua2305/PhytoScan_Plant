"""Hiệu chỉnh confidence.
Temperature scaling: Guo et al., 2017 — https://arxiv.org/abs/1706.04599
ECE: Naeini et al., 2015 — https://ojs.aaai.org/index.php/AAAI/article/view/9602
Học DUY NHẤT scalar T>0 tối ưu NLL trên calib split -> KHÔNG đổi accuracy,
chỉ hiệu chỉnh probability -> chọn threshold reject trên xác suất ĐÃ hiệu chỉnh.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, init_T=1.5):
        super().__init__()
        self.log_T = nn.Parameter(torch.log(torch.tensor(init_T)))   # T=exp(log_T)>0 luôn dương

    @property
    def temperature(self):
        return self.log_T.exp().item()

    def forward(self, logits):
        return logits / self.log_T.exp()

    def fit(self, logits, labels, max_iter=50):
        """L-BFGS như đề xuất gốc — hội tụ nhanh với bài toán 1 tham số."""
        logits, labels = logits.detach(), labels.detach()
        opt = torch.optim.LBFGS([self.log_T], lr=0.01, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        opt.step(closure)
        return self.temperature


@torch.no_grad()
def expected_calibration_error(probs, labels, n_bins=15):
    """ECE: chia confidence thành n_bins, |accuracy - confidence| bình quân có trọng số."""
    conf, pred = probs.max(dim=1)
    acc = pred.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=probs.device)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.float().mean() > 0:
            ece += (acc[m].float().mean() - conf[m].mean()).abs() * m.float().mean()
    return ece.item()


@torch.no_grad()
def brier_score(probs, labels):
    """Brier multiclass: MSE giữa probability và one-hot."""
    oh = F.one_hot(labels, probs.size(1)).float()
    return ((probs - oh) ** 2).sum(dim=1).mean().item()


@torch.no_grad()
def collect_logits(model, loader, device, use_amp=True):
    """Trích logits 1 lần rồi tối ưu T trên tensor tĩnh (không forward lại khi L-BFGS lặp)."""
    model.eval()
    all_logits, all_labels = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            logits = model(imgs)
        all_logits.append(logits.float().cpu()); all_labels.append(targets.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def calibrate_head(model, calib_loader, device):
    """Trích logits trên calib -> ECE/Brier trước -> fit T -> ECE/Brier sau.
    Trả dict để ghi calibration.json (Mục 7). Chọn threshold TRÊN xác suất đã chia T."""
    logits, labels = collect_logits(model, calib_loader, device)
    p_before = torch.softmax(logits, dim=1)
    ece_b, brier_b = expected_calibration_error(p_before, labels), brier_score(p_before, labels)
    T = TemperatureScaler().fit(logits, labels)
    p_after = torch.softmax(logits / T, dim=1)
    ece_a, brier_a = expected_calibration_error(p_after, labels), brier_score(p_after, labels)
    print(f"[calibrate] T={T:.4f} | ECE {ece_b:.4f}->{ece_a:.4f} | Brier {brier_b:.4f}->{brier_a:.4f}")
    return {"temperature": T, "ece_before": ece_b, "ece_after": ece_a,
            "brier_before": brier_b, "brier_after": brier_a}


def _run_calibration(config_path, target):
    """target='species' | 'leaf_gate' (giữ để tương thích nhưng không dùng). 
    Trích logits trên calib -> fit T -> ghi calibration.json.
    Chạy: python -m src.eval.calibration --config configs/finetune_tomato.yaml --target species
          python -m src.eval.calibration --config configs/leaf_gate.yaml --target leaf_gate"""
    import json
    import os
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from src.config import TrainConfig
    from src.data.augmentation import get_finetune_transforms
    from src.train.train_finetune import load_manifest
    
    cfg = TrainConfig.load(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint (xử lý weights_only)
    try:
        ckpt = torch.load(cfg["checkpoint"]["out_path"], map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(cfg["checkpoint"]["out_path"], map_location=device)

    # Xây dựng model và dataset dựa trên cấu hình
    if "split_manifest" in cfg["data"]:
        # species: dùng FinetuneDataset
        from src.models.backbone import build_backbone
        from src.data.datasets import FinetuneDataset
        model = build_backbone(ckpt["num_classes"], ckpt["arch"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        manifest = load_manifest(cfg["data"]["split_manifest"])
        ds = FinetuneDataset(manifest, "calib", get_finetune_transforms(False, cfg["data"]["img_size"]))
    else:
        # leaf_gate: dùng ImageFolder (hoặc LeafGateDataset nếu có)
        from src.models.backbone import build_leaf_gate
        model = build_leaf_gate(pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"])
        root_path = os.path.join(cfg["data"]["root"], "calib")
        if not os.path.isdir(root_path):
            raise FileNotFoundError(f"Thư mục calib {root_path} không tồn tại.")
        # Transform eval (torchvision)
        transform_eval = transforms.Compose([
            transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        ds = ImageFolder(root_path, transform_eval)

    loader = DataLoader(ds, cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], pin_memory=True)
    cal = calibrate_head(model, loader, device)
    out = cfg["checkpoint"]["out_path"].replace(".pt", "_calibration.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=2)
    print(f"[calibration] -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Calibration temperature scaling trên calib split")
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", default="species", choices=["species", "leaf_gate"])
    a = ap.parse_args()
    _run_calibration(a.config, a.target)