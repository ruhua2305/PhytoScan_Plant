"""Kiểm checksums.sha256 khớp file trong bundle (Mục 8.10)."""
import hashlib
import os
import tempfile

from src.export.build_bundle import _sha256


def test_sha256_matches_hashlib():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"phytoscan-bundle-test")
        path = f.name
    try:
        expected = hashlib.sha256(b"phytoscan-bundle-test").hexdigest()
        assert _sha256(path) == expected
    finally:
        os.unlink(path)


def test_checksum_file_format_roundtrip():
    """Mô phỏng ghi + đọc checksums.sha256 rồi re-verify."""
    with tempfile.TemporaryDirectory() as d:
        files = {"a.txt": b"aaa", "b.json": b'{"k":1}'}
        for name, content in files.items():
            with open(os.path.join(d, name), "wb") as f:
                f.write(content)
        lines = []
        for name in sorted(files):
            lines.append(f"{_sha256(os.path.join(d, name))}  {name}")
        with open(os.path.join(d, "checksums.sha256"), "w") as f:
            f.write("\n".join(lines) + "\n")
        # re-verify
        with open(os.path.join(d, "checksums.sha256")) as f:
            for line in f:
                digest, name = line.strip().split("  ")
                assert _sha256(os.path.join(d, name)) == digest
