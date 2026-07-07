"""PyTorch -> ONNX cho tầng phục vụ. ONNX inference KHÔNG cần PyTorch/CUDA -> Docker nhẹ.
Chạy: python -m src.export.export_onnx --config configs/finetune_tomato.yaml --out model_bundles/tmp/tomato/model.onnx
"""
import argparse
import json
import os
import torch
from src.config import TrainConfig
from src.models.backbone import build_backbone


def export_to_onnx(config_path, out_path, img_size=224, opset=17):
    device = "cpu"        # export CPU đủ nhanh, tránh phụ thuộc CUDA lúc build
    cfg = TrainConfig.load(config_path)
    ckpt = torch.load(cfg["checkpoint"]["out_path"], map_location=device)
    model = build_backbone(ckpt["num_classes"], ckpt["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(model, dummy, out_path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, do_constant_folding=True)
    # class_mapping.json cạnh model -> API map index -> tên lớp (Mục 7.1)
    meta = out_path.replace("model.onnx", "class_mapping.json")
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({"idx_to_class": {i: c for i, c in enumerate(ckpt["classes"])},
                   "arch": ckpt["arch"], "img_size": img_size,
                   "species": ckpt.get("species", "")}, f, ensure_ascii=False, indent=2)
    print(f"[export] {out_path} + {meta} ({ckpt['num_classes']} lớp)")
    return out_path


def quantize_dynamic_int8(onnx_path, out_path):
    """INT8 dynamic — giảm ~75%. BẮT BUỘC đo lại macro-F1 + parity sau bước này."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(onnx_path, out_path, weight_type=QuantType.QInt8)
    print(f"[int8] {os.path.getsize(onnx_path)/1e6:.1f}MB -> {os.path.getsize(out_path)/1e6:.1f}MB")


def convert_fp16(onnx_path, out_path):
    import onnx
    from onnxconverter_common import float16
    onnx.save(float16.convert_float_to_float16(onnx.load(onnx_path)), out_path)
    print(f"[fp16] -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantize", default="none", choices=["none", "int8", "fp16"])
    a = ap.parse_args()
    export_to_onnx(a.config, a.out)
    if a.quantize == "int8":
        quantize_dynamic_int8(a.out, a.out.replace(".onnx", "_int8.onnx"))
    elif a.quantize == "fp16":
        convert_fp16(a.out, a.out.replace(".onnx", "_fp16.onnx"))
