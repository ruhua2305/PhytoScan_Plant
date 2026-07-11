"""Adapter model công khai cho Cross-Test Chiều 2. Mỗi adapter có cùng interface:
    labels: list[str]                       # nhãn output theo index
    predict_logits(batch_np) -> np.ndarray  # [N,3,224,224] float32 (ImageNet-norm) -> [N, num_labels]

Hai adapter:
  - BackbonePVModel  (B): EfficientNetV2-S 38 lớp của mình (checkpoints/backbone_pretrained.pt).
                          Offline, không cần lib mới.
  - DeiTPVModel      (A): rescu/deit-base-patch16-224-finetuned-plantvillage (39 lớp) qua transformers.
                          https://huggingface.co/rescu/deit-base-patch16-224-finetuned-plantvillage
Cả hai đều KHÔNG fine-tune trên field -> dùng để đo domain gap lab->field (Mohanty 2016).
"""
import numpy as np
import torch

from src.cross_test.label_maps import DEIT39_LABELS, load_pv38_labels


class BackbonePVModel:
    """B — backbone EffV2-S 38 lớp mình đã pretrain (chỉ biết ảnh lab PlantVillage)."""
    name = "backbone_pv38"

    def __init__(self, ckpt_path="checkpoints/backbone_pretrained.pt", device=None):
        from src.models.backbone import build_backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        n = ckpt.get("num_classes", 38)
        self.model = build_backbone(n, ckpt.get("arch", "tf_efficientnetv2_s.in1k"),
                                    pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.labels = load_pv38_labels()   # 38 nhãn theo đúng thứ tự sorted (khớp lúc train)
        assert len(self.labels) == n, f"PV labels {len(self.labels)} != num_classes {n}"

    @torch.no_grad()
    def predict_logits(self, batch_np):
        x = torch.from_numpy(batch_np).to(self.device)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            out = self.model(x)
        return out.float().cpu().numpy()


class DeiTPVModel:
    """A — model NGOÀI: DeiT-base fine-tune PlantVillage (39 lớp, gồm Background_without_leaves)."""
    name = "deit_pv39"
    repo_id = "rescu/deit-base-patch16-224-finetuned-plantvillage"

    def __init__(self, device=None):
        from transformers import AutoModelForImageClassification
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageClassification.from_pretrained(self.repo_id).to(self.device)
        self.model.eval()
        # Ưu tiên id2label từ chính model; fallback bảng đã chép trong label_maps nếu thiếu.
        id2label = getattr(self.model.config, "id2label", None)
        if id2label and len(id2label) == self.model.config.num_labels:
            self.labels = [id2label[i] for i in range(len(id2label))]
        else:
            self.labels = list(DEIT39_LABELS)

    @torch.no_grad()
    def predict_logits(self, batch_np):
        x = torch.from_numpy(batch_np).to(self.device)
        return self.model(pixel_values=x).logits.float().cpu().numpy()


def build_public_model(name, device=None):
    if name in ("backbone", "backbone_pv38", "b"):
        return BackbonePVModel(device=device)
    if name in ("deit", "deit_pv39", "a"):
        return DeiTPVModel(device=device)
    raise ValueError(f"model công khai không hỗ trợ: {name} (chọn 'backbone' | 'deit')")
