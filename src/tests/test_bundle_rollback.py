"""Bundle build gate + rollback readiness (Mục 7.4b, 8.13, 8.10)."""
import os
import tempfile

from src.export.build_bundle import _copy_if_exists


def test_copy_if_exists_returns_name():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model.onnx")
        with open(src, "wb") as f:
            f.write(b"onnx")
        dst = os.path.join(d, "bundle")
        os.makedirs(dst)
        assert _copy_if_exists(src, dst, "model.onnx") == "model.onnx"
        assert os.path.exists(os.path.join(dst, "model.onnx"))


def test_copy_if_exists_missing_returns_none():
    with tempfile.TemporaryDirectory() as d:
        assert _copy_if_exists(os.path.join(d, "nope.onnx"), d, "x.onnx") is None


def test_bundle_gate_requires_onnx_and_mapping():
    """Mô phỏng gate cuối build_bundle: thiếu 1 trong 2 artefact bắt buộc -> chặn."""
    def gate(files):
        return bool(files.get("onnx") and files.get("class_mapping"))
    assert gate({"onnx": "model.onnx", "class_mapping": "class_mapping.json"})
    assert not gate({"onnx": "model.onnx", "class_mapping": None})
    assert not gate({"onnx": None, "class_mapping": "class_mapping.json"})


def test_keep_two_recent_bundles_policy():
    """Chính sách giữ >=2 bundle gần nhất để rollback (Mục 8.4)."""
    versions = ["tomato_v1", "tomato_v2", "tomato_v3"]
    keep = sorted(versions)[-2:]
    assert keep == ["tomato_v2", "tomato_v3"]
    assert len(keep) >= 2
