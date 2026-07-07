"""Calibration: TemperatureScaler + ECE (Mục 5.3, 8.10). Cần torch."""
import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not _HAS_TORCH, reason="cần torch")
def test_temperature_positive_and_reduces_nll_when_overconfident():
    import torch
    import torch.nn.functional as F
    from src.eval.calibration import TemperatureScaler
    torch.manual_seed(0)
    # logits overconfident: scale lớn -> T tối ưu > 1 để "làm mềm"
    labels = torch.randint(0, 5, (200,))
    base = F.one_hot(labels, 5).float()
    logits = base * 8.0 + torch.randn(200, 5) * 0.5
    ts = TemperatureScaler()
    nll_before = F.cross_entropy(logits, labels).item()
    T = ts.fit(logits, labels)
    nll_after = F.cross_entropy(logits / T, labels).item()
    assert T > 0
    assert nll_after <= nll_before + 1e-4


@pytest.mark.skipif(not _HAS_TORCH, reason="cần torch")
def test_ece_perfect_calibration_low():
    import torch
    from src.eval.calibration import expected_calibration_error
    # dự đoán đúng với confidence ~1 -> ECE thấp
    probs = torch.zeros(100, 3)
    labels = torch.randint(0, 3, (100,))
    probs[torch.arange(100), labels] = 0.99
    probs += 0.005
    ece = expected_calibration_error(probs, labels)
    assert ece < 0.1
