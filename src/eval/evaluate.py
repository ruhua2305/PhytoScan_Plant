"""Đánh giá — evaluate() là 'đơn vị đo' dùng chung cho train_pretrain/train_finetune
(chọn checkpoint theo macro-F1). Hỗ trợ softmax head + prototype few-shot.
"""
import argparse
import os
import torch
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torchvision import transforms
from torchvision.datasets import ImageFolder


@torch.no_grad()
def collect_predictions(model, loader, device, use_amp=True):
    model.eval()
    y_true, y_pred = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            logits = model(imgs)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


def evaluate(model, loader, device, verbose=False, class_names=None, use_amp=True):
    """Trả macro-F1. verbose=True in thêm classification report + confusion matrix."""
    y_true, y_pred = collect_predictions(model, loader, device, use_amp)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if verbose:
        print(f"\n=== Classification report (macro-F1={macro_f1:.4f}) ===")
        print(classification_report(y_true, y_pred, target_names=class_names,
                                    zero_division=0, digits=4))
        print("=== Confusion matrix (hàng=thật, cột=dự đoán) ===")
        print(confusion_matrix(y_true, y_pred))
    return macro_f1


@torch.no_grad()
def evaluate_prototype(query_features, query_labels, prototypes, verbose=False, class_names=None):
    """Đánh giá nhánh Chế độ 2 — trả macro-F1 để so sánh đồng nhất với softmax."""
    from src.train.few_shot import classify_by_prototype
    preds = classify_by_prototype(query_features, prototypes.to(query_features.device)).argmax(1).cpu()
    yt = query_labels.cpu().tolist()
    macro_f1 = f1_score(yt, preds.tolist(), average="macro", zero_division=0)
    if verbose:
        print(classification_report(yt, preds.tolist(), target_names=class_names,
                                    zero_division=0, digits=4))
    return macro_f1


def _run_standalone(config_path, split):
    from src.config import TrainConfig
    from src.models.backbone import build_backbone
    from src.data.datasets import FinetuneDataset
    from src.data.augmentation import get_finetune_transforms
    from src.train.train_finetune import load_manifest
    from torch.utils.data import DataLoader

    cfg = TrainConfig.load(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint (thử weights_only=True để tránh cảnh báo)
    try:
        ckpt = torch.load(cfg["checkpoint"]["out_path"], map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(cfg["checkpoint"]["out_path"], map_location=device)

    model = build_backbone(ckpt["num_classes"], ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    # --- Tạo dataset dựa trên loại cấu hình ---
    if "split_manifest" in cfg["data"]:
        # Các loài chính: dùng manifest JSON và albumentations
        transforms_eval = get_finetune_transforms(False, cfg["data"]["img_size"])
        manifest = load_manifest(cfg["data"]["split_manifest"])
        ds = FinetuneDataset(manifest, split, transforms_eval)
        class_names = ckpt["classes"]
    else:
        # leaf_gate: dùng thư mục con và torchvision transforms
        root_path = os.path.join(cfg["data"]["root"], split)
        if not os.path.isdir(root_path):
            raise FileNotFoundError(
                f"Thư mục {root_path} không tồn tại. "
                f"Kiểm tra cfg['data']['root'] và split='{split}'."
            )
        transforms_eval = transforms.Compose([
            transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        ds = ImageFolder(root_path, transforms_eval)
        class_names = ds.classes  # ['leaf', 'not_leaf'] theo thứ tự alphabet

    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )

    evaluate(model, loader, device, verbose=True, class_names=class_names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val", choices=["train", "calib", "val", "holdout"])
    args = ap.parse_args()
    _run_standalone(args.config, args.split)