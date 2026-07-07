"""Preprocessing train vs API khớp (Mục 8.10).
So numpy-preprocess của app.py với công thức ImageNet-norm tham chiếu."""
import importlib.util

import numpy as np
import pytest

_HAS = all(importlib.util.find_spec(m) for m in ("onnxruntime", "PIL", "fastapi"))

PP = {"img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def _reference_preprocess(arr_uint8):
    mean = np.array(PP["mean"], np.float32)
    std = np.array(PP["std"], np.float32)
    a = (arr_uint8.astype(np.float32) / 255.0 - mean) / std
    return np.transpose(a, (2, 0, 1))[None, ...].astype(np.float32)


@pytest.mark.skipif(not _HAS, reason="cần PIL/onnxruntime/fastapi")
def test_app_preprocess_matches_reference():
    from PIL import Image
    from src.deploy.app import _preprocess
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    out_app = _preprocess(img, PP)
    out_ref = _reference_preprocess(arr)
    assert out_app.shape == (1, 3, 224, 224)
    assert np.allclose(out_app, out_ref, atol=1e-5)


def test_reference_shape_and_range():
    """Chạy được không cần dep — bảo vệ công thức tham chiếu."""
    arr = np.zeros((224, 224, 3), np.uint8)
    out = _reference_preprocess(arr)
    assert out.shape == (1, 3, 224, 224)
    # pixel 0 -> (-mean/std), kiểm kênh R
    assert abs(out[0, 0, 0, 0] - (-0.485 / 0.229)) < 1e-4
