"""Factory backbone qua timm. Primary: EfficientNetV2-S (D1). Fallback: B0/MobileNetV3."""
import timm
import torch.nn as nn


def build_backbone(num_classes: int, arch: str = "tf_efficientnetv2_s.in1k",
                   pretrained: bool = True, drop_path_rate: float = 0.1) -> nn.Module:
    """
    arch khuyến nghị theo ưu tiên:
      - "tf_efficientnetv2_s.in1k"       : MẶC ĐỊNH — accuracy tốt nhất trong ngân sách 4GB
      - "tf_efficientnetv2_b0.in1k"      : fallback khi OOM / iterate nhanh
      - "mobilenetv3_large_100.ra_in1k"  : fallback nhẹ hơn nữa
      - "mobilevit_xs.cvnets_in1k"       : ablation hybrid CNN-Transformer (cần AMP)
    drop_path_rate: stochastic depth chống overfit khi data ít.
    """
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes,
                             drop_path_rate=drop_path_rate)


def build_leaf_gate(pretrained: bool = True) -> nn.Module:
    # MobileNetV3-Small — nhẹ, đủ cho bài toán nhị phân leaf/not-leaf (Mục 3.2)
    return timm.create_model("mobilenetv3_small_100.lamb_in1k",
                             pretrained=pretrained, num_classes=2)
