"""Prototype few-shot: shape + hành vi cơ bản (Mục 3.5, 8.10). Cần torch."""
import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not _HAS_TORCH, reason="cần torch")
def test_prototype_shape_and_classify():
    import torch
    from src.train.few_shot import compute_prototypes, classify_by_prototype
    torch.manual_seed(0)
    D, C = 16, 3
    support = torch.randn(30, D)
    labels = torch.randint(0, C, (30,))
    protos = compute_prototypes(support, labels, C)
    assert protos.shape == (C, D)
    # prototype đã L2-normalize -> norm ~1 (trừ lớp rỗng)
    for c in range(C):
        if (labels == c).sum() > 0:
            assert abs(protos[c].norm().item() - 1.0) < 1e-4
    query = torch.randn(5, D)
    scores = classify_by_prototype(query, protos)
    assert scores.shape == (5, C)
    assert torch.allclose(scores.sum(1), torch.ones(5), atol=1e-5)


@pytest.mark.skipif(not _HAS_TORCH, reason="cần torch")
def test_prototype_recovers_own_class():
    """Query = chính prototype -> phải phân về đúng lớp đó."""
    import torch
    from src.train.few_shot import compute_prototypes, classify_by_prototype
    D = 8
    support = torch.eye(3, D)               # 3 vector trực giao rõ ràng
    labels = torch.tensor([0, 1, 2])
    protos = compute_prototypes(support, labels, 3)
    preds = classify_by_prototype(support, protos).argmax(1)
    assert preds.tolist() == [0, 1, 2]
