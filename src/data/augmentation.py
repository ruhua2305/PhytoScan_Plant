"""Augmentation FINETUNE nâng cấp: RandAugment + CutMix/Mixup.
RandAugment: Cubuk et al., 2019 — https://arxiv.org/abs/1909.13719
CutMix: Yun et al., 2019 — https://arxiv.org/abs/1905.04899
Mixup:  Zhang et al., 2017 — https://arxiv.org/abs/1710.09412
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_finetune_transforms(train=True, img_size=224, strong=False):
    if train:
        aug = [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            # Thay RandAugment (không có trong albumentations) bằng tổ hợp tương đương:
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.06),
                     rotate=(-25, 25), shear=(-8, 8), p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(std_range=(0.03, 0.12), p=0.15),
            A.ImageCompression(quality_range=(50, 95), p=0.2),
        ]
        if strong:   # lớp biên <30 ảnh — augment mạnh hơn, KHÔNG thay thế few-shot (Mục 3.5)
            aug += [A.CoarseDropout(num_holes_range=(1, 4),
                                    hole_height_range=(0.05, 0.12),
                                    hole_width_range=(0.05, 0.12), p=0.4)]
        aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
        return A.Compose(aug)
    return A.Compose([A.Resize(height=img_size, width=img_size),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def get_pretrain_transforms(train=True, img_size=224):
    """Ảnh lab nền sạch — augment nhẹ hơn."""
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
    return A.Compose([A.Resize(height=img_size, width=img_size),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


class CutMixMixupCollate:
    """CutMix HOẶC Mixup ngẫu nhiên mỗi batch — dùng làm collate_fn.
    LƯU Ý: chỉ trộn trong head của MỘT loài (mỗi loài num_classes riêng),
    KHÔNG trộn ảnh xuyên loài. Ở pretrain thì hợp lệ vì 38 lớp chung 1 head."""
    def __init__(self, num_classes, alpha=0.2, cutmix_prob=0.5, p=0.5):
        self.num_classes, self.alpha, self.cutmix_prob, self.p = num_classes, alpha, cutmix_prob, p

    def _one_hot(self, y):
        oh = torch.zeros(len(y), self.num_classes)
        oh[torch.arange(len(y)), y] = 1.0
        return oh

    def __call__(self, batch):
        imgs = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch])
        if np.random.rand() > self.p:
            return imgs, self._one_hot(labels)
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(imgs.size(0))
        y1, y2 = self._one_hot(labels), self._one_hot(labels[idx])
        if np.random.rand() < self.cutmix_prob:
            H, W = imgs.shape[-2:]
            r = np.sqrt(1 - lam)
            cw, ch = int(W * r), int(H * r)
            cx, cy = np.random.randint(W), np.random.randint(H)
            x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
            y1_, y2_ = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
            imgs[:, :, y1_:y2_, x1:x2] = imgs[idx][:, :, y1_:y2_, x1:x2]
            lam = 1 - ((x2 - x1) * (y2_ - y1_) / (W * H))
        else:
            imgs = lam * imgs + (1 - lam) * imgs[idx]
        return imgs, lam * y1 + (1 - lam) * y2
