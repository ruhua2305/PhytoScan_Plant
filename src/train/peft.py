"""PEFT-lite: mở khoá dần head -> BN -> block cuối. Rẻ VRAM hơn full-unfreeze.
LoRA (Hu et al., 2021 — https://arxiv.org/abs/2106.09685) là mức nâng cao cho backbone ViT-based.
"""


def set_trainable_params(model, mode="head_bn_lastblock"):
    for name, p in model.named_parameters():
        n = name.lower()
        if mode == "head_only":
            p.requires_grad = "classifier" in n or "fc" in n
        elif mode == "head_bn_only":
            p.requires_grad = ("classifier" in n or "fc" in n or "bn" in n or "norm" in n)
        elif mode == "head_bn_lastblock":
            p.requires_grad = ("classifier" in n or "fc" in n or "bn" in n or "norm" in n
                               or "blocks.6" in name or "blocks.5" in name)  # tuỳ kiến trúc timm
        else:
            p.requires_grad = True
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  [PEFT {mode}] trainable {n_tr:,}/{n_all:,} ({100*n_tr/n_all:.2f}%)")
    return model
