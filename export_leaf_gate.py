import os
import json
import yaml
import torch
import torch.onnx
from src.models.backbone import build_leaf_gate   # nếu file khác, sửa import

def export_leaf_gate():
    # Đọc config
    with open("configs/leaf_gate.yaml", 'r') as f:
        config = yaml.safe_load(f)
    img_size = config['data']['img_size']
    ckpt_path = config['checkpoint']['out_path']

    # Tạo thư mục output
    out_dir = "model_bundles/leaf_gate"
    os.makedirs(out_dir, exist_ok=True)

    # Khởi tạo model (pretrained=False vì sẽ load trọng số)
    model = build_leaf_gate(pretrained=False)

    # Load checkpoint
    device = 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device)

    # Xử lý nhiều dạng checkpoint
    if isinstance(ckpt, dict):
        # Ưu tiên key 'model' hoặc 'model_state'
        if 'model' in ckpt:
            ckpt = ckpt['model']
        elif 'model_state' in ckpt:
            ckpt = ckpt['model_state']
        model.load_state_dict(ckpt)
    else:
        model = ckpt

    model.eval()
    model.to(device)

    # Export ONNX
    dummy_input = torch.randn(1, 3, img_size, img_size)
    onnx_path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=11,
        do_constant_folding=True
    )
    print(f"✅ ONNX exported to {onnx_path}")

    # Copy hoặc tạo calibration.json
    calib_src = "checkpoints/leaf_gate_calibration.json"
    calib_dst = os.path.join(out_dir, "calibration.json")
    if os.path.exists(calib_src):
        import shutil
        shutil.copy2(calib_src, calib_dst)
        print(f"✅ Calibration copied to {calib_dst}")
    else:
        default_calib = {"temperature": 1.0}
        with open(calib_dst, "w") as f:
            json.dump(default_calib, f, indent=2)
        print(f"⚠️ Created default calibration: {calib_dst}")

if __name__ == "__main__":
    export_leaf_gate()