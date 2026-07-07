"""Đánh giá OOD/reject: AUROC + FPR@95%TPR trên điểm phân tách in-dist vs OOD."""
import numpy as np
from sklearn.metrics import roc_auc_score


def fpr_at_tpr(scores_in, scores_out, tpr_target=0.95):
    """scores: CAO = giống in-dist. Tìm ngưỡng giữ tpr_target của in-dist, đo FPR trên OOD."""
    thr = np.quantile(scores_in, 1 - tpr_target)
    fpr = (np.array(scores_out) >= thr).mean()
    return float(fpr), float(thr)


def ood_auroc(scores_in, scores_out):
    y = np.concatenate([np.ones_like(scores_in), np.zeros_like(scores_out)])
    s = np.concatenate([scores_in, scores_out])
    return float(roc_auc_score(y, s))
