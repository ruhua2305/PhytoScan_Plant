"""Benchmark robustness: áp corruption tăng dần, đo macro-F1 drop. KHÔNG dùng holdout.

Với mỗi (loại, mức): build val loader kèm corruption -> evaluate() -> ghi macro-F1 drop.
Fairness note (Mục 8.13): nếu có metadata nguồn/thiết bị, báo cáo drop theo nhóm;
nếu thiếu metadata, nêu giới hạn thay vì suy diễn.
"""
import albumentations as A


CORRUPTIONS = {
    "gauss_blur": [A.GaussianBlur(blur_limit=(3, 3), p=1), A.GaussianBlur(blur_limit=(7, 7), p=1),
                   A.GaussianBlur(blur_limit=(13, 13), p=1)],
    "brightness": [A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=0, p=1),
                   A.RandomBrightnessContrast(brightness_limit=(0.4, 0.4), contrast_limit=0, p=1)],
    "noise": [A.GaussNoise(std_range=(0.04, 0.04), p=1), A.GaussNoise(std_range=(0.1, 0.1), p=1),
              A.GaussNoise(std_range=(0.2, 0.2), p=1)],
    "jpeg": [A.ImageCompression(quality_range=(50, 50), p=1),
             A.ImageCompression(quality_range=(30, 30), p=1),
             A.ImageCompression(quality_range=(15, 15), p=1)],
    "rotate": [A.Rotate(limit=(15, 15), p=1), A.Rotate(limit=(30, 30), p=1),
               A.Rotate(limit=(45, 45), p=1)],
}


def build_corruption_transform(corruption_op, img_size=224):
    """Ghép 1 corruption trước resize + normalize để tái dùng cùng pipeline eval."""
    from albumentations.pytorch import ToTensorV2
    from src.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
    return A.Compose([corruption_op, A.Resize(height=img_size, width=img_size),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
