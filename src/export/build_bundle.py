"""Đóng gói bundle bất biến (Mục 7.1). Gom artefact -> manifest.json + checksums.sha256.
Chạy: python -m src.export.build_bundle --species tomato --version tomato_2026_07_04_effv2s_ab12cd \
        --onnx model_bundles/tmp/tomato/model.onnx --git-commit ab12cd34
"""
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone, timedelta

BUNDLES = r"D:/DAT301m_Project/Final_Pipeline/model_bundles"
CKPT = r"D:/DAT301m_Project/Final_Pipeline/checkpoints"


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _copy_if_exists(src, dst_dir, dst_name):
    """Copy artefact vào bundle nếu tồn tại; trả tên file hoặc None (ghi rõ khi thiếu)."""
    if src and os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, dst_name))
        return dst_name
    print(f"  [WARN] thiếu artefact: {src} (bỏ qua {dst_name})")
    return None


def build_bundle(species, version, onnx_path, git_commit, opset=17, quantization="fp32"):
    out = os.path.join(BUNDLES, version)
    os.makedirs(out, exist_ok=True)

    # 1) Bắt buộc: ONNX + class_mapping + preprocessing
    files = {}
    files["onnx"] = _copy_if_exists(onnx_path, out, "model.onnx")
    cmap_src = onnx_path.replace("model.onnx", "class_mapping.json")
    files["class_mapping"] = _copy_if_exists(cmap_src, out, "class_mapping.json")
    with open(os.path.join(out, "preprocessing.json"), "w", encoding="utf-8") as f:
        json.dump({"img_size": 224, "resize": "bilinear",
                   "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                   "channel_order": "RGB", "layout": "NCHW"}, f, indent=2)
    files["preprocessing"] = "preprocessing.json"

    # 2) Tuỳ artefact có sẵn: calibration / thresholds / prototypes / metrics / parity
    base = os.path.join(CKPT, f"model_{species}")
    files["calibration"] = _copy_if_exists(base + "_calibration.json", out, "calibration.json")
    files["thresholds"] = _copy_if_exists(base + "_thresholds.json", out, "thresholds.json")
    files["prototype_file"] = _copy_if_exists(base + "_prototypes.pt", out, "prototypes.pt")
    files["metrics"] = _copy_if_exists(base + "_metrics.json", out, "metrics.json")
    files["parity_report"] = _copy_if_exists(
        onnx_path.replace("model.onnx", "parity_report.json"), out, "parity_report.json")
    files["taxonomy"] = _copy_if_exists(
        r"D:/DAT301m_Project/Final_Pipeline/configs/taxonomy_v1.yaml", out, "taxonomy.yaml")

    # 3) manifest.json
    manifest = {
        "model_version": version, "species": species, "git_commit": git_commit,
        "onnx": {"file": "model.onnx", "opset": opset, "quantization": quantization},
        **{k: v for k, v in files.items() if v},
        "created_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 4) checksums.sha256 cho MỌI file trong bundle (trừ chính nó)
    with open(os.path.join(out, "checksums.sha256"), "w", encoding="utf-8") as f:
        for fn in sorted(os.listdir(out)):
            if fn == "checksums.sha256":
                continue
            f.write(f"{_sha256(os.path.join(out, fn))}  {fn}\n")

    # 5) Gate: bắt buộc phải có ONNX + class_mapping (Mục 7.1, 8.13)
    if not (files["onnx"] and files["class_mapping"]):
        raise RuntimeError("Bundle THIẾU model.onnx hoặc class_mapping.json — CHẶN release.")
    print(f"[bundle] {out} sẵn sàng ({len([v for v in files.values() if v])} artefact).")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Đóng gói bundle bất biến (Mục 7.1)")
    ap.add_argument("--species", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--git-commit", default="unknown")
    ap.add_argument("--quantization", default="fp32", choices=["fp32", "fp16", "int8"])
    a = ap.parse_args()
    build_bundle(a.species, a.version, a.onnx, a.git_commit, quantization=a.quantization)
