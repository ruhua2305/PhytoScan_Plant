"""Log inference dạng JSON một dòng để monitor + drift (Mục 8.6). KHÔNG ghi ảnh."""
import json
import time
import uuid


def log_prediction(logger_file, species, model_version, result, latency_ms, input_bytes):
    rec = {
        "timestamp": time.time(), "request_id": str(uuid.uuid4()),
        "model_version": model_version, "species": species,
        "predicted_class": result.get("prediction"),
        "top1_confidence": result.get("confidence"),
        "reject_reason": result.get("reason") or result.get("status"),
        "latency_ms": round(latency_ms, 1), "input_bytes": input_bytes,
        "status_code": 200,
    }
    with open(logger_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
