"""ONNX parity logic (Mục 7.4, 8.10). Cần torch+onnxruntime -> thường chạy trên máy đích.
Test độc lập: hàm so sánh top-1/max-diff hoạt động đúng trên mảng giả."""
import importlib.util

import numpy as np
import pytest


def _parity_metrics(pt_logits, onnx_logits, atol=1e-3):
    top1 = (pt_logits.argmax(1) == onnx_logits.argmax(1)).mean()
    max_diff = float(np.abs(pt_logits - onnx_logits).max())
    return {"top1_match_rate": float(top1), "max_abs_diff": max_diff,
            "passed": bool(top1 == 1.0 and max_diff < atol)}


def test_parity_pass_when_identical():
    a = np.random.randn(8, 20).astype(np.float32)
    m = _parity_metrics(a, a.copy())
    assert m["passed"] and m["top1_match_rate"] == 1.0


def test_parity_fail_on_large_diff():
    a = np.random.randn(8, 20).astype(np.float32)
    b = a.copy(); b[0, 0] += 5.0    # đổi argmax hàng 0
    m = _parity_metrics(a, b)
    assert not m["passed"]


def test_parity_tolerates_tiny_noise():
    a = np.random.randn(8, 20).astype(np.float32)
    b = a + 1e-5                    # nhiễu nhỏ, không đổi argmax
    m = _parity_metrics(a, b)
    assert m["top1_match_rate"] == 1.0


_HAS = all(importlib.util.find_spec(x) for x in ("torch", "onnxruntime"))


@pytest.mark.skipif(not _HAS, reason="cần torch+onnxruntime (máy đích)")
def test_run_parity_importable():
    from src.export.parity_test import run_parity  # noqa: F401
