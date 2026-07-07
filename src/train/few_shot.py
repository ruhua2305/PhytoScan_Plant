"""Prototypical classification cho lớp <30 ảnh (Snell, Swersky, Zemel, 2017
   — https://arxiv.org/abs/1703.05175). Không train tham số riêng cho lớp hiếm.
"""
import torch
import torch.nn.functional as F


def compute_prototypes(support_features: torch.Tensor, support_labels: torch.Tensor,
                       num_classes: int) -> torch.Tensor:
    """support_features [N, D] -> prototypes [num_classes, D] (mean đã L2-normalize)."""
    feats = F.normalize(support_features, dim=1)
    protos = torch.zeros(num_classes, feats.size(1))
    for c in range(num_classes):
        mask = support_labels == c
        if mask.sum() > 0:
            protos[c] = F.normalize(feats[mask].mean(dim=0), dim=0)
    return protos


def classify_by_prototype(query_features: torch.Tensor, prototypes: torch.Tensor):
    """Cosine similarity tới prototype gần nhất -> softmax score."""
    q = F.normalize(query_features, dim=1)
    sim = q @ prototypes.t()          # [Nq, num_classes], cosine vì đã normalize
    return sim.softmax(dim=-1)
