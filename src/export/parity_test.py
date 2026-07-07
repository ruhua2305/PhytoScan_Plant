"""Parity PyTorch vs ONNX: cùng input -> so top-1 + max abs logit diff.
Chạy: python -m src.export.parity_test --bundle model_bundles/tmp/tomato/ --golden-set data/golden/tomato.json
"""
import argparse
import json
import os
import numpy as np
import onnxruntime as ort
import torch
from src.models.backbone import build_backbone


def run_parity(pt_ckpt, onnx_path, sample_batch_np, atol=1e-3):
    """sample_batch_np: [N,3,H,W] float32 đã preprocess. Trả dict parity."""
    device = "cpu"
    ckpt = torch.load(pt_ckpt, map_location=device)
    model = build_backbone(ckpt["num_classes"], ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    with torch.no_grad():
        pt_logits = model(torch.from_numpy(sample_batch_np)).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logits"], {"input": sample_batch_np})[0]

    top1_match = (pt_logits.argmax(1) == onnx_logits.argmax(1)).mean()
    max_diff = float(np.abs(pt_logits - onnx_logits).max())
    passed = bool(top1_match == 1.0 and max_diff < atol)
    return {"top1_match_rate": float(top1_match), "max_abs_diff": max_diff,
            "atol": atol, "passed": passed}


def _load_golden_batch(golden_set_path, img_size=224):
    """golden_set: JSON {"images": ["abs_or_rel_path", ...]}. Preprocess ImageNet-norm."""
    from PIL import Image
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    with open(golden_set_path, encoding="utf-8") as f:
        paths = json.load(f)["images"]
    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((img_size, img_size))
        a = (np.asarray(img, np.float32) / 255.0 - mean) / std
        arrs.append(np.transpose(a, (2, 0, 1)))
    return np.stack(arrs).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="thư mục chứa model.onnx")
    ap.add_argument("--pt-ckpt", required=True, help="checkpoint PyTorch tương ứng")
    ap.add_argument("--golden-set", required=True)
    a = ap.parse_args()
    batch = _load_golden_batch(a.golden_set)
    report = run_parity(a.pt_ckpt, os.path.join(a.bundle, "model.onnx"), batch)
    out = os.path.join(a.bundle, "parity_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[parity] {report} -> {out}")
    if not report["passed"]:
        raise SystemExit("PARITY FAIL — CHẶN release (Mục 8.13).")
