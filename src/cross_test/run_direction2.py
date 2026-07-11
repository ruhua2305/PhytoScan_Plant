"""Cross-Test Chiều 2: model công khai (train trên PlantVillage lab) -> đánh giá trên HOLDOUT
FIELD của mình, so với model per-species đã fine-tune. Định lượng domain gap lab->field.

Ba lựa chọn --model:
  backbone : B — EffV2-S 38 lớp mình pretrain (offline).       https://arxiv.org/abs/1511.08060
  deit     : A — DeiT-39 ngoài (transformers, tải HF ~330MB).  https://huggingface.co/rescu/deit-base-patch16-224-finetuned-plantvillage
  mine     : baseline — model per-species đã fine-tune (checkpoints/model_<species>.pt).
Cơ sở: Mohanty et al. 2016, https://doi.org/10.3389/fpls.2016.01419

Đánh giá CLOSED-SET per-species: mỗi loài chỉ xét các lớp thuộc loài đó (công bằng khi so với
head per-species của mình). Với model công khai cũng log open-set top-1 (argmax toàn bộ nhãn) để
biết tỉ lệ đoán "lạc loài".

Chạy: python -m src.cross_test.run_direction2 --model backbone
      python -m src.cross_test.run_direction2 --model deit
      python -m src.cross_test.run_direction2 --model mine
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

# Console Windows (cp1258) không in được ký tự tiếng Việt/em-dash -> ép UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.cross_test.label_maps import build_species_index_map

SPECIES = ["apple", "corn", "grape", "pepper_bell", "tomato"]
SPLITS_DIR = "data/splits"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
OUT_DIR = "reports/cross_test"


def load_manifest(species):
    with open(os.path.join(SPLITS_DIR, f"{species}_split_v1.json"), encoding="utf-8") as f:
        return json.load(f)


def species_classes(manifest):
    """Trả list nhãn nội bộ theo class_idx 0..K-1."""
    idx_to_class = {v: k for k, v in manifest["class_to_idx"].items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]


def preprocess(path, img_size=224):
    """ImageNet-norm 224 NCHW — khớp preprocessing.json của bundle + input DeiT/backbone."""
    from src.data.datasets import imread_rgb
    import cv2
    img = imread_rgb(path)
    img = cv2.resize(img, (img_size, img_size))
    arr = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))


def holdout_items(manifest):
    """Trả list (abs_path, local_idx) của split holdout."""
    root = manifest["dataset_root"]
    return [(os.path.join(root, it["relative_path"]), it["class_idx"])
            for it in manifest["items"] if it["split"] == "holdout"]


def batched(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i + bs]


def eval_public_model(model, batch_size=32):
    """Chạy model công khai trên holdout 5 loài. Trả (summary_rows, per_image_rows)."""
    summary, all_rows = [], []
    for sp in SPECIES:
        man = load_manifest(sp)
        my_classes = species_classes(man)
        local_to_pub, unmatched = build_species_index_map(my_classes, model.labels)
        if unmatched:
            print(f"[{sp}] ⚠️ lớp không map được sang model công khai: {unmatched}")
        pub_idx = [j for j in local_to_pub if j is not None]     # cột public dùng cho closed-set
        # local index chỉ giữ lớp map được
        keep_local = [k for k, j in enumerate(local_to_pub) if j is not None]

        items = holdout_items(man)
        y_true, y_pred_closed, y_pred_open_correct_sp = [], [], []
        n_open_offspecies = 0
        for chunk in batched(items, batch_size):
            batch = np.stack([preprocess(p) for p, _ in chunk]).astype(np.float32)
            logits = model.predict_logits(batch)                 # [n, num_pub_labels]
            for row, (path, local) in zip(logits, chunk):
                if local not in keep_local:
                    continue                                     # lớp loài này model công khai không có
                # closed-set: chỉ xét các cột public thuộc loài -> map ngược về local idx
                sub = row[pub_idx]
                pred_pub_pos = int(np.argmax(sub))               # vị trí trong pub_idx
                pred_local = keep_local[pred_pub_pos]
                y_true.append(local)
                y_pred_closed.append(pred_local)
                # open-set: argmax toàn bộ; đoán có rơi đúng vào lớp thuộc loài không
                open_top1 = int(np.argmax(row))
                if open_top1 not in pub_idx:
                    n_open_offspecies += 1
                all_rows.append({"species": sp, "image": os.path.basename(path),
                                 "true": my_classes[local], "pred_closed": my_classes[pred_local],
                                 "open_top1_label": model.labels[open_top1],
                                 "correct": local == pred_local})
        if y_true:
            mf1 = f1_score(y_true, y_pred_closed, average="macro", zero_division=0)
            acc = accuracy_score(y_true, y_pred_closed)
            off = n_open_offspecies / len(y_true)
            summary.append({"species": sp, "n": len(y_true), "macro_f1": round(mf1, 4),
                            "accuracy": round(acc, 4), "open_offspecies_rate": round(off, 4)})
            print(f"[{sp}] n={len(y_true)} closed macro-F1={mf1:.4f} acc={acc:.4f} "
                  f"open off-species={off:.1%}")
    return summary, all_rows


def eval_mine(batch_size=32):
    """Baseline: model per-species đã fine-tune, trên cùng holdout. Closed-set theo bản chất."""
    from src.models.backbone import build_backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary, all_rows = [], []
    for sp in SPECIES:
        man = load_manifest(sp)
        my_classes = species_classes(man)
        ckpt = torch.load(f"checkpoints/model_{sp}.pt", map_location=device, weights_only=False)
        model = build_backbone(ckpt["num_classes"], ckpt["arch"], pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"]); model.eval()
        items = holdout_items(man)
        y_true, y_pred = [], []
        with torch.no_grad():
            for chunk in batched(items, batch_size):
                batch = torch.from_numpy(
                    np.stack([preprocess(p) for p, _ in chunk]).astype(np.float32)).to(device)
                with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(batch)
                preds = logits.argmax(1).cpu().numpy()
                for pr, (path, local) in zip(preds, chunk):
                    y_true.append(local); y_pred.append(int(pr))
                    all_rows.append({"species": sp, "image": os.path.basename(path),
                                     "true": my_classes[local], "pred_closed": my_classes[int(pr)],
                                     "open_top1_label": my_classes[int(pr)],
                                     "correct": local == int(pr)})
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        summary.append({"species": sp, "n": len(y_true), "macro_f1": round(mf1, 4),
                        "accuracy": round(acc, 4), "open_offspecies_rate": 0.0})
        print(f"[{sp}] n={len(y_true)} macro-F1={mf1:.4f} acc={acc:.4f}")
    return summary, all_rows


def main():
    ap = argparse.ArgumentParser(description="Cross-Test Chiều 2 (public model -> holdout field)")
    ap.add_argument("--model", required=True, choices=["backbone", "deit", "mine"])
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if args.model == "mine":
        summary, rows = eval_mine(args.batch_size)
        tag = "mine_finetuned"
    else:
        from src.cross_test.public_models import build_public_model
        model = build_public_model(args.model)
        print(f"[model] {model.name} — {len(model.labels)} nhãn output")
        summary, rows = eval_public_model(model, args.batch_size)
        tag = model.name

    # tổng hợp macro-F1 trung bình (không trọng số theo loài) + accuracy gộp
    if summary:
        overall_mf1 = round(float(np.mean([s["macro_f1"] for s in summary])), 4)
        total_n = sum(s["n"] for s in summary)
        overall_acc = round(sum(s["accuracy"] * s["n"] for s in summary) / total_n, 4)
    else:
        overall_mf1 = overall_acc = None

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"direction2_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = {"model": tag, "per_species": summary,
           "overall_macro_f1_mean": overall_mf1, "overall_accuracy": overall_acc}
    with open(os.path.join(OUT_DIR, f"direction2_{tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[SUMMARY {tag}] overall macro-F1 (mean per-species)={overall_mf1} "
          f"| accuracy={overall_acc}\n-> {OUT_DIR}/direction2_{tag}_summary.json")


if __name__ == "__main__":
    main()
