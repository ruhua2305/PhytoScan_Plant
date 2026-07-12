"""Field-background recomposition (ý tưởng Jeon et al., 2025 — https://dx.doi.org/10.2139/ssrn.5217081).

Cắt lá khỏi nền lab đơn sắc rồi dán lên nền field ngẫu nhiên -> giảm domain gap lab->field.
Dùng như augmentation khi PRETRAIN (Mục 4.6, Cách 1).
"""
import cv2
import numpy as np
import random


def extract_leaf_mask(lab_image_bgr):
    """Tách lá khỏi nền đơn sắc bằng Otsu (ảnh PRETRAIN nền sạch)."""
    gray = cv2.cvtColor(lab_image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def recompose_on_field_background(lab_image_bgr, field_bg_bgr):
    mask = extract_leaf_mask(lab_image_bgr)
    field_bg = cv2.resize(field_bg_bgr, (lab_image_bgr.shape[1], lab_image_bgr.shape[0]))
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    return (lab_image_bgr * m3 + field_bg * (1 - m3)).astype(np.uint8)


class FieldBackgroundAugment:
    """Dùng trong Dataset.__getitem__ khi train PRETRAIN, p ~ 0.3-0.5."""
    def __init__(self, field_background_paths, p=0.4):
        self.bg_paths, self.p = field_background_paths, p

    def __call__(self, lab_image_bgr):
        if random.random() > self.p or not self.bg_paths:
            return lab_image_bgr
        bg = cv2.imread(random.choice(self.bg_paths))
        return recompose_on_field_background(lab_image_bgr, bg) if bg is not None else lab_image_bgr
