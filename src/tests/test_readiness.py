"""Readiness/rollback: đọc con trỏ active_version + tính có mặt bundle (Mục 8.2-8.4, 8.10).
Test thuần filesystem — không cần torch/onnx."""
import os
import tempfile


def _read_active_bundle(bundles_dir, species):
    """Bản sao logic app._read_active_bundle (test không import onnxruntime)."""
    ptr = os.path.join(bundles_dir, f"active_version_{species}.txt")
    if not os.path.exists(ptr):
        return None
    with open(ptr, encoding="utf-8") as f:
        return os.path.join(bundles_dir, f.read().strip())


def test_active_pointer_missing_returns_none():
    with tempfile.TemporaryDirectory() as d:
        assert _read_active_bundle(d, "tomato") is None


def test_active_pointer_resolves_bundle_dir():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "tomato_v1"))
        with open(os.path.join(d, "active_version_tomato.txt"), "w") as f:
            f.write("tomato_v1\n")
        resolved = _read_active_bundle(d, "tomato")
        assert resolved.endswith("tomato_v1") and os.path.isdir(resolved)


def test_rollback_switch_pointer():
    """Đổi con trỏ sang bundle cũ = rollback (Mục 8.2)."""
    with tempfile.TemporaryDirectory() as d:
        for v in ("tomato_v1", "tomato_v2"):
            os.makedirs(os.path.join(d, v))
        ptr = os.path.join(d, "active_version_tomato.txt")
        with open(ptr, "w") as f:
            f.write("tomato_v2")
        assert _read_active_bundle(d, "tomato").endswith("tomato_v2")
        with open(ptr, "w") as f:      # rollback
            f.write("tomato_v1")
        assert _read_active_bundle(d, "tomato").endswith("tomato_v1")
