"""Class mapping ổn định giữa train/bundle/API (Mục 8.10).
Kiểm cách app.py dựng danh sách lớp từ class_mapping.json — index liên tục 0..N-1."""
import json
import os
import tempfile


def _classes_from_mapping(cm):
    """Mô phỏng logic app._startup: idx_to_class[str(i)] cho i=0..N-1."""
    return [cm["idx_to_class"][str(i)] for i in range(len(cm["idx_to_class"]))]


def test_index_contiguous_and_ordered():
    cm = {"idx_to_class": {"0": "Apple_scab", "1": "Cedar_apple_rust", "2": "healthy"}}
    classes = _classes_from_mapping(cm)
    assert classes == ["Apple_scab", "Cedar_apple_rust", "healthy"]


def test_mapping_json_roundtrip():
    cm = {"idx_to_class": {i: f"class_{i}" for i in range(9)}}  # export ghi key int
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "class_mapping.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cm, f)
        loaded = json.load(open(p, encoding="utf-8"))   # JSON ép key -> str
        classes = _classes_from_mapping(loaded)
        assert len(classes) == 9
        assert classes[5] == "class_5"


def test_missing_index_raises():
    """Thiếu index -> KeyError (bắt lỗi mapping hỏng thay vì im lặng)."""
    cm = {"idx_to_class": {"0": "a", "2": "c"}}  # thiếu "1"
    try:
        _classes_from_mapping(cm)
        assert False, "phải raise KeyError khi mapping không liên tục"
    except KeyError:
        pass
