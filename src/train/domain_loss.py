"""MMD loss (cảm hứng Wu et al., 2023 — https://doi.org/10.34133/plantphenomics.0038)."""
import torch


def mmd_loss(source_features, target_features, sigma=1.0):
    """Maximum Mean Discrepancy (RBF) — kéo gần phân phối đặc trưng 2 miền."""
    def rbf(x, y):
        return torch.exp(-(torch.cdist(x, y) ** 2) / (2 * sigma ** 2))
    return rbf(source_features, source_features).mean() \
        + rbf(target_features, target_features).mean() \
        - 2 * rbf(source_features, target_features).mean()
# Trong train loop: loss = ce + lambda_mmd * mmd_loss(feat_src, feat_tgt)
