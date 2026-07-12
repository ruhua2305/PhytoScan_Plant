"""PretrainDataset (đọc theo folder, chèn field-background) +
   FinetuneDataset (đọc từ manifest split_v1.json, Mục 2.5).
"""
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

IMG_EXT = (".jpg", ".jpeg", ".png")


def imread_bgr(path):
    """Đọc ảnh BGR an toàn với đường dẫn Unicode trên Windows.
    cv2.imread không đọc được path chứa ký tự non-ASCII (vd tên file tiếng Nhật)
    -> dùng np.fromfile + cv2.imdecode. Raise lỗi rõ nếu file hỏng/không đọc được."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, IOError) as e:
        raise FileNotFoundError(f"Không đọc được file ảnh: {path} ({e})")
    if data.size == 0:
        raise ValueError(f"File ảnh rỗng: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không giải mã được ảnh (định dạng lỗi/không hỗ trợ): {path}")
    return img


def imread_rgb(path):
    """Như imread_bgr nhưng trả RGB (cho pipeline dùng RGB)."""
    return cv2.cvtColor(imread_bgr(path), cv2.COLOR_BGR2RGB)


class PretrainDataset(Dataset):
    def __init__(self, root_dir, transform=None, domain_aug=None):
        """domain_aug: FieldBackgroundAugment hoặc None. CHỈ truyền cho train;
        val giữ nguyên phân phối lab gốc để macro-F1 khách quan."""
        self.transform, self.domain_aug = transform, domain_aug
        self.classes = sorted(c for c in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, c)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cdir = os.path.join(root_dir, cls)
            for f in os.listdir(cdir):
                if f.lower().endswith(IMG_EXT):
                    self.samples.append((os.path.join(cdir, f), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = imread_bgr(path)          # hỗ trợ Unicode path (Windows)
        if self.domain_aug is not None:
            img = self.domain_aug(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label


class FinetuneDataset(Dataset):
    """Đọc split 'train'|'calib'|'val'|'holdout' từ manifest (Mục 2.5).
    class_to_idx LẤY TỪ manifest (bất biến), KHÔNG suy từ folder name."""
    def __init__(self, manifest, split, transform=None):
        self.transform = transform
        self.class_to_idx = manifest["class_to_idx"]
        root = manifest["dataset_root"]
        self.samples = [
            (os.path.join(root, it["relative_path"]), it["class_idx"])
            for it in manifest["items"] if it["split"] == split
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = imread_rgb(path)          # hỗ trợ Unicode path (Windows)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label