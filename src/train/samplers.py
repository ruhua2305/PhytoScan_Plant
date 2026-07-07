"""WeightedRandomSampler dùng effective_num_weights (nhất quán công thức toàn dự án)."""
from torch.utils.data import WeightedRandomSampler
from src.train.losses import effective_num_weights


def build_balanced_sampler(samples, class_counts, beta=0.9999):
    """samples: list (path, label_idx). Dùng cho Bước 3 decoupled re-training (Mục 4.9)."""
    class_w = effective_num_weights(class_counts, beta=beta)
    sample_weights = [class_w[label].item() for _, label in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
