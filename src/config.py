"""Loader YAML tối giản — tránh phụ thuộc Hydra/OmegaConf cho quy mô dự án cá nhân.

Dùng:
    from src.config import TrainConfig
    cfg = TrainConfig.load("configs/finetune_tomato.yaml")
    cfg["train"]["batch_size"]           # truy cập kiểu dict
    cfg.raw.get("domain_aug", {})        # truy cập an toàn khối tuỳ chọn
"""
from dataclasses import dataclass

import yaml


@dataclass
class TrainConfig:
    raw: dict

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(raw=yaml.safe_load(f))

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)
