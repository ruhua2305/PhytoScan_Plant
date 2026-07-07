"""Checkpoint đủ để resume chính xác + giải thích metric sau này (Mục 4.10)."""
import torch


def save_resume_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric, cfg,
                           class_to_idx, split_sha256, config_hash):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch, "best_metric": best_metric,
        "class_to_idx": class_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()},
        "config": cfg.raw, "config_hash": config_hash,
        "split_manifest_sha256": split_sha256,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)
