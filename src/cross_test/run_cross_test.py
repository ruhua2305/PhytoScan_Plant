"""Chạy cross-test theo mapping nhãn giao + log prediction để audit (Mục 6).
Mỗi prediction ghi: image_id, source_dataset, mapped_label, pred_label, confidence, correct.
"""
import json
import os
from sklearn.metrics import f1_score, accuracy_score


def evaluate_with_label_map(preds, label_map, intersection_classes):
    """preds: list dict {image_id, true_raw, pred_label, confidence}.
    label_map: raw_label -> internal_label. Chỉ tính trên intersection_classes."""
    rows, y_true, y_pred = [], [], []
    for p in preds:
        mapped = label_map.get(p["true_raw"])
        if mapped not in intersection_classes:
            continue                       # bỏ lớp ngoài giao
        y_true.append(mapped)
        y_pred.append(p["pred_label"])
        rows.append({**p, "mapped_label": mapped, "correct": mapped == p["pred_label"]})
    return rows, y_true, y_pred


def summarize(rows, y_true, y_pred, scenario, out_dir="reports/cross_test"):
    """Ghi rows -> jsonl + trả macro-F1/accuracy để điền bảng 6.5."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{scenario}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not y_true:
        return {"scenario": scenario, "n": 0, "macro_f1": None, "accuracy": None}
    return {"scenario": scenario, "n": len(y_true),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred)}
