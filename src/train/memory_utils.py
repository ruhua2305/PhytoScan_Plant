"""AMP + gradient accumulation + probe VRAM.
Mixed precision: Micikevicius et al., 2017 — https://arxiv.org/abs/1710.03740
Grad checkpointing: Chen et al., 2016 — https://arxiv.org/abs/1604.06174
"""
import torch


def train_one_epoch_memory_efficient(model, loader, optimizer, criterion, device,
                                     accumulation_steps=4, use_amp=True,
                                     use_grad_checkpoint=False):
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_grad_checkpoint and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)   # timm hỗ trợ sẵn nhiều kiến trúc
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    for step, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            logits = model(imgs)
            loss = criterion(logits, labels) / accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * accumulation_steps
    return running_loss / len(loader)


def probe_vram_usage(model, sample_batch, device, batch_sizes=(4, 8, 16, 32)):
    """BƯỚC BẮT BUỘC trước train full: đo VRAM thật trên chính máy thay vì đoán.
    Trả {bs: '<peak> GB' hoặc 'OOM'}. Chạy 1 lần trước mỗi kiến trúc mới."""
    results = {}
    for bs in batch_sizes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            imgs = sample_batch[:1].repeat(bs, 1, 1, 1).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs)
                loss = out.sum()
            loss.backward()
            results[bs] = f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB"
            model.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError:
            results[bs] = "OOM"
            torch.cuda.empty_cache()
            break
    return results
