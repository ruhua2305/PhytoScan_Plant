"""Energy-based OOD (Liu, Wang, Owens, Li, 2020 — https://arxiv.org/abs/2010.03759)."""
import torch


def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Điểm năng lượng: THẤP (âm nhiều) = giống in-distribution (lá quen thuộc)."""
    return -T * torch.logsumexp(logits / T, dim=1)


def leaf_gate_decision(binary_logits, species_logits, energy_threshold,
                       binary_threshold=0.5, leaf_T=1.0):
    """Kết hợp binary gate VÀ energy score từ model loài (lớp bảo vệ thứ hai).
    binary_threshold & energy_threshold chọn từ calib split (Mục 5), KHÔNG cảm tính."""
    p_leaf = torch.softmax(binary_logits / leaf_T, dim=1)[:, 1]
    is_leaf = p_leaf > binary_threshold
    is_familiar = energy_score(species_logits) < energy_threshold
    return is_leaf & is_familiar
