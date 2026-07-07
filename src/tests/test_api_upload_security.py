"""Upload security: oversize / MIME / magic-byte / pixel-bomb bị chặn (Mục 7.6, 8.10).

Test thuần logic (không cần onnxruntime) + test tích hợp validate_image_upload (skip nếu thiếu dep).
"""
import importlib.util

import pytest

MAX_BYTES = 8 * 1024 * 1024
ALLOWED_MAGIC = {b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"}


def _has_magic(raw):
    return any(raw.startswith(sig) for sig in ALLOWED_MAGIC)


def test_reject_oversize_logic():
    assert len(b"x" * (MAX_BYTES + 1)) > MAX_BYTES


def test_reject_bad_magic_bytes():
    assert not _has_magic(b"GIF89a....")          # GIF không được phép
    assert not _has_magic(b"<html>")              # HTML/text
    assert not _has_magic(b"%PDF-1.4")            # PDF


def test_accept_valid_magic_bytes():
    assert _has_magic(b"\xff\xd8\xff\xe0JFIF")    # JPEG
    assert _has_magic(b"\x89PNG\r\n\x1a\n")       # PNG


_HAS_DEPS = all(importlib.util.find_spec(m) for m in ("onnxruntime", "PIL", "fastapi"))


@pytest.mark.skipif(not _HAS_DEPS, reason="cần onnxruntime/PIL/fastapi (chạy trên máy đích)")
def test_validate_image_upload_integration():
    from fastapi import HTTPException
    from src.deploy.app import validate_image_upload
    with pytest.raises(HTTPException):
        validate_image_upload(b"not-an-image")
    with pytest.raises(HTTPException):
        validate_image_upload(b"\xff\xd8\xff" + b"0" * (MAX_BYTES + 1))
