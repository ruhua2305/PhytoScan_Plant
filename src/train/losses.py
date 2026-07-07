"""Class-Balanced Focal Loss.
Class-Balanced: Cui et al., 2019 — https://arxiv.org/abs/1901.05555
Focal:          Lin et al., 2017 — https://arxiv.org/abs/1708.02002
Label smoothing: Müller et al., 2019 — https://arxiv.org/abs/1906.02629
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_num_weights(class_counts, beta=0.9999):
    """(1 - beta^n) / (1 - beta), chuẩn hoá về trung bình 1.0.
    Lớp có count=0 (đã tách sang few-shot Chế độ 2) -> weight=0 để tránh inf/nan
    và bỏ qua lớp đó trong loss (nó không có mẫu train)."""
    counts = torch.tensor(class_counts, dtype=torch.float)
    eff_num = 1.0 - torch.pow(beta, counts)
    weights = torch.where(counts > 0, (1.0 - beta) / eff_num, torch.zeros_like(counts))
    s = weights.sum()
    if s <= 0:
        raise ValueError("Tất cả lớp đều count=0 — không có dữ liệu train.")
    return weights / s * (counts > 0).sum()   # chuẩn hoá về TB 1.0 trên các lớp có mẫu


class ClassBalancedFocalLoss(nn.Module):
    """Cho targets là CHỈ SỐ LỚP nguyên (val, hoặc train không CutMix)."""
    def __init__(self, class_counts, beta=0.9999, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.register_buffer("weights", effective_num_weights(class_counts, beta))
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        w = self.weights.to(logits.device)
        ce = F.cross_entropy(logits, targets, weight=w,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class ClassBalancedFocalLossSoft(nn.Module):
    """Tương thích SOFT-LABEL từ CutMix/Mixup (tổ hợp lồi 2 one-hot).
    Khi soft_targets suy biến về one-hot thuần -> thu gọn ĐÚNG về bản gốc ở trên."""
    def __init__(self, class_counts, beta=0.9999, gamma=2.0):
        super().__init__()
        self.register_buffer("weights", effective_num_weights(class_counts, beta))
        self.gamma = gamma

    def forward(self, logits, soft_targets):
        w = self.weights.to(logits.device)
        # Chấp nhận cả nhãn nguyên [B] (khi tắt CutMix/Mixup) lẫn soft-label [B, C].
        if soft_targets.dim() == 1 or soft_targets.shape != logits.shape:
            soft_targets = torch.zeros_like(logits).scatter_(
                1, soft_targets.view(-1, 1).long(), 1.0)
        log_probs = torch.log_softmax(logits, dim=1)
        raw_ce = -(soft_targets * log_probs).sum(dim=1)         # soft-CE, chưa nhân weight
        sample_weight = (soft_targets * w).sum(dim=1)            # tổ hợp lồi trọng số lớp
        pt = torch.exp(-raw_ce)
        return ((1 - pt) ** self.gamma * raw_ce * sample_weight).mean()
