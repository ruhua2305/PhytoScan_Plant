# DAT301m — Phân loại Bệnh Cây Trồng qua Ảnh Lá — KẾ HOẠCH FINAL

| Trường | Giá trị |
|---|---|
| Phiên bản | Final v1.1 — đồng bộ với code đã triển khai (đối chiếu 2026-07-12) |
| Ngày | 2026-07-12 (bản gốc 2026-07-04) |
| Thư mục gốc | `D:\DAT301m_Project\Final_Pipeline` |
| Phần cứng | NVIDIA RTX 3050 Ti Laptop GPU, 4GB VRAM — 16GB RAM — Windows |
| Loại dự án | Computer vision, ML nông nghiệp, triển khai web FastAPI |
| Trạng thái | Đã có code chạy đầu-cuối + checkpoint 5 loài + leaf-gate + bundle + web demo; một số thành phần nâng cao chưa kích hoạt (xem Mục 1.8) |

> **Tài liệu này đã được đồng bộ lại với code thực tế trong repo (rà 2026-07-12).** Nó giữ khung **quản trị khoa học** (split 70/10/10/10 có manifest + hash, holdout khoá, calibration split riêng, golden set, ONNX parity, bundle bất biến) và phần **kỹ thuật + code chạy được** (Class-Balanced Focal Loss, PEFT 3 bước, AMP/gradient-accumulation, FastAPI + ONNX Runtime, leaf-gate + energy-OOD). So với bản 2026-07-04, bản này **cập nhật theo dữ liệu và code thật**: dữ liệu FINETUNE nay có **18,402 ảnh** (lớp `Spider_mites` tăng từ 7 lên **55 ảnh**), nên **không còn lớp nào < 30 ảnh** — nhánh few-shot/prototype (Chế độ 2) trở thành **hướng mở rộng, không phải thành phần đang chạy** (Mục 3.5, 1.8); tầng serving load bundle từ **Hugging Face Hub** (Mục 7.5); và các thành phần chưa hoàn tất được **đánh dấu trạng thái trung thực** thay vì trình bày như đã xong (Mục 1.8). Chi tiết thay đổi ở Mục 1.5.

---

## Các quyết định đã chốt (đọc trước tiên)

| # | Quyết định | Chọn | Lý do |
|---|---|---|---|
| D1 | Backbone chính | **EfficientNetV2-S** (~21M), fallback **EfficientNetV2-B0** (~7M) | Ưu tiên accuracy tối đa; khả thi trên 4GB nhờ AMP + gradient accumulation + PEFT (không full-unfreeze) + freeze backbone cho loài nhỏ |
| D2 | ~~Embedding cho lớp hiếm~~ | **Không dùng ở bản hiện tại** | Dữ liệu mới không còn lớp < 30 ảnh (Spider_mites nay 55) → few-shot/prototype chuyển sang hướng mở rộng (Mục 1.8, 3.5) |
| D3 | Ngôn ngữ tài liệu | **Tiếng Việt**, giữ thuật ngữ kỹ thuật + tên bảng bằng tiếng Anh | Theo yêu cầu người dùng |
| D4 | Mức độ code | **Kế hoạch chi tiết + đầy đủ code chạy được** | Ghép code V2 + governance V1 |
| D5 | Chia dữ liệu | **70/10/10/10** = train/calib/val/holdout, có manifest + SHA256, holdout khoá | Theo V1 — nghiêm ngặt nhất, chống rò rỉ, tốt cho báo cáo học thuật |
| D6 | Định tuyến loài | **Per-species heads** (5 model riêng), người dùng chọn loài | Giảm nhầm lẫn xuyên loài, đơn giản mapping |

---

## 1. Executive Summary

### 1.1. Mục tiêu dự án

Dự án xây dựng hệ thống phân loại bệnh cây từ ảnh lá cho **5 loài**: Apple, Corn, Grape, Pepper bell và Tomato. Sản phẩm cuối là một **web demo** (frontend PhytoScan + FastAPI backend) nơi người dùng chọn loài cây, upload ảnh lá; hệ thống kiểm tra ảnh hợp lệ bằng **Leaf Gate**, sau đó model theo loài trả về **top-k bệnh** kèm **confidence đã calibration**, và có khung **từ chối** (reject) khi ảnh không phải lá / độ tin cậy thấp / ngoài phân phối. Backend hiện load bundle model (ONNX) từ **Hugging Face Hub** và phục vụ inference bằng ONNX Runtime CPU (Mục 7.5).

Mục tiêu không chỉ là accuracy cao trên validation, mà là một **pipeline có thể audit**: split có manifest bất biến, class mapping cố định, calibration riêng từng thành phần, ONNX parity, golden set smoke test, và policy reject rõ ràng. Tài liệu ưu tiên **bằng chứng field holdout + cross-test** hơn là con số accuracy kiểu PlantVillage.

### 1.2. Luồng xử lý end-to-end

```text
Người dùng chọn loài -> upload ảnh
  -> API kiểm tra: dung lượng, MIME, magic bytes, kích thước pixel, EXIF
  -> Leaf Gate kiểm tra ảnh có phải lá hợp lệ không
       -> từ chối nếu: không phải lá / ảnh hỏng / loài không hỗ trợ / quá mập mờ
       -> tiếp tục nếu là lá hợp lệ
  -> Model ONNX theo loài (per-species classifier)
  -> temperature scaling + thresholds + margin + prototype/OOD check + top-k
  -> trả về: ok / low_confidence / ambiguous / unknown_or_ood / not_a_leaf / invalid_file
```

Chi tiết reject taxonomy và fallback UX ở Mục 3 (Leaf Gate) và Mục 7 (API).

### 1.3. Chiến lược theo giai đoạn

| Phase | Dữ liệu | Hành động model | Output | Vì sao quan trọng |
|---|---|---|---|---|
| Phase 0 | Thư mục dữ liệu hiện có | Xây taxonomy + manifest khoá | `taxonomy_v1.json`, split manifests | Chặn lỗi class mapping & rò rỉ |
| Phase 1 | PRETRAIN kiểu PlantVillage, 53,599 ảnh, 38 lớp | Pretrain EfficientNetV2-S | Checkpoint backbone chung | Học đặc trưng lá/bệnh rẻ |
| Phase 2 | FINETUNE ảnh field, 18,402 ảnh, 20 lớp | Fine-tune per-species (PEFT 3 bước) | 5 model theo loài | Thu hẹp domain gap lab→field |
| Phase 3 | calib split | Temperature scaling (thresholds là việc còn lại) | Calibration.json mỗi loài | Confidence có ý nghĩa hơn |
| Phase 4 | golden + holdout | ONNX export + parity, holdout report | Bundle + báo cáo | Tái lập & so sánh được |
| Phase 5 | FastAPI + PhytoScan frontend | Serving qua HF Hub bundle | Web demo | Sản phẩm dùng được |

### 1.4. Chiến lược phần cứng (4GB VRAM là ràng buộc hạng nhất)

GPU mục tiêu là RTX 3050 Ti Laptop 4GB VRAM — đây là **ràng buộc thiết kế trung tâm**. Bản Final chọn **EfficientNetV2-S** làm backbone chính để tối đa accuracy (D1), và khả thi trên 4GB nhờ tổ hợp bắt buộc: **AMP FP16 + gradient accumulation + PEFT** (mở khoá dần head → BN → 1–2 block cuối thay vì full-unfreeze). Với loài dữ liệu nhỏ (Grape/Pepper), thay vì đóng băng hoàn toàn, code dùng **backbone LR rất thấp** (`5e-6`) + augment bảo thủ + early stopping để tránh overfit (xem `configs/finetune_grape.yaml`). B0 là fallback khi cần iterate nhanh hoặc nếu S vẫn OOM ở batch nhỏ nhất.

Quy tắc vận hành:

- Kích thước ảnh mặc định: `224x224` cho mọi baseline; `256x256` chỉ sau khi probe VRAM đạt.
- Batch mặc định: physical `16` + AMP; effective batch `32–64` qua gradient accumulation (không dùng batch thật 32/64).
- DataLoader workers: bắt đầu `2` trên 16GB RAM; không dùng 8.
- Mọi experiment phải log `torch.cuda.max_memory_allocated()` và `max_memory_reserved()`.
- Một backbone bị **loại khỏi production** nếu dry-run 20 step vượt `3.6GB` reserved VRAM.
- **Bắt buộc chạy `probe_vram_usage()` trước mỗi vòng train full** (tích hợp sẵn trong script train, Mục 4).

### 1.5. Các thay đổi so với bản 2026-07-04

Bản này đồng bộ tài liệu với code + dữ liệu thực tế. Các sửa đổi chính:

| Khu vực | Bản 2026-07-04 nói | Thực tế trong repo (2026-07-12) | Xử lý |
|---|---|---|---|
| Dữ liệu Spider_mites | 7 ảnh (Chế độ 2 rare) | **55 ảnh** | Cập nhật mọi số; bỏ khỏi diện "lớp hiếm" |
| Tổng FINETUNE | 18,354 | **18,402** | Cập nhật bảng 2.1–2.4 |
| Lớp < 30 ảnh | Có (Spider_mites) | **Không còn lớp nào < 30** | `rare_folds` rỗng ở mọi manifest; không có file prototype |
| Few-shot/prototype (Chế độ 2) | Thành phần chính | Code còn nhưng **không kích hoạt** với dữ liệu hiện tại | Hạ xuống mục "hướng mở rộng" (Mục 3.5, 1.8) |
| Serving | Registry cục bộ `active_version_*.txt`, load-all lúc startup | Load bundle từ **Hugging Face Hub** (private repos), **lazy load per-species** | Viết lại Mục 7.5 theo HF Hub; registry cục bộ là phương án offline |
| Thresholds reject | Đã tune trên calib | **Chưa có `thresholds.json`** — API chạy default | Đánh dấu là việc còn lại (Mục 5.4, 1.8) |
| Bước 3 decoupled | CE có `weight=class_w` | Code dùng **CE thường** (tránh cân bằng gấp đôi với sampler) | Sửa Mục 4.9 theo code |
| Grape/Pepper | "freeze backbone" | Backbone LR thấp `5e-6` + early stopping | Sửa Mục 1.4, 3.1 |

**Lỗi kế thừa từ bản gốc đã sửa từ trước (giữ lại để tham chiếu):** loài `potato` sai → 5 loài đúng; đường dẫn `src//Main_Pipeline_V2/...` → `src/...`; thống nhất root `D:\DAT301m_Project\Final_Pipeline`; áp split 70/10/10/10 + holdout lock; backbone chốt EffV2-S + B0 fallback.

### 1.6. Mục tiêu metric & deployment gates

| Khu vực | Mục tiêu | Gate |
|---|---|---|
| Apple/Tomato field holdout | Macro-F1 >= 0.90 | Bắt buộc đạt để tuyên bố demo cuối |
| Corn field holdout | Macro-F1 >= 0.80 | Phải báo per-class F1 vì imbalance cao |
| Grape/Pepper field holdout | Macro-F1 >= 0.70 | Phải kèm confidence interval + ghi chú giới hạn |
| Tomato Spider_mites | 55 ảnh (không còn critical-rare) | Vẫn là lớp ít nhất của Tomato — báo per-class recall, không thổi phồng |
| Calibration | ECE <= 0.05 mỗi loài nếu có thể | **Mục tiêu chưa đạt** ở hầu hết loài (Mục 5.3) → cần đặt reject threshold chặt hơn khi build thresholds |
| ONNX parity | 100% top-1 parity trên golden/parity set | Bắt buộc trước khi release bundle |
| API security | Test oversize/MIME/magic-byte pass | Bắt buộc trước demo |
| Latency | Local p95 <= 3s | Nếu không, fallback backbone nhỏ hơn / preload session |

### 1.7. Non-Goals

- Không chẩn đoán mọi bệnh cây, sâu hại, thiếu dinh dưỡng, bệnh quả/thân hay tổn thương cơ học.
- Không thay thế tư vấn chuyên gia nông nghiệp.
- Không tuyên bố so sánh trực tiếp với các paper trừ khi dataset/split/mapping đã căn chỉnh.
- Không deploy một model 20-lớp nguyên khối; luôn định tuyến per-species theo lựa chọn người dùng (D6).

### 1.8. Trạng thái triển khai (Đã làm / Chưa làm / Tùy chọn)

Để tránh trình bày quá mức, bảng sau phân biệt rõ cái gì đã chạy và cái gì còn là thiết kế/việc còn lại:

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| Split manifest 5 loài + hash SHA256 | ✅ Đã làm | `data/splits/*_split_v1.json`, seed 42 |
| Pretrain backbone 38 lớp | ✅ Đã làm | `checkpoints/backbone_pretrained.pt` |
| Fine-tune 5 loài (PEFT 3 bước) | ✅ Đã làm | `checkpoints/model_<species>.pt` |
| Leaf-gate (MobileNetV3-Small) | ✅ Đã làm | `checkpoints/leaf_gate.pt` |
| Temperature scaling mỗi loài + leaf-gate | ✅ Đã làm | `*_calibration.json` (nhưng **ECE chưa đạt ≤0.05**, Mục 5.3) |
| ONNX export + parity test | ✅ Đã làm | parity pass (top-1 100%, max_diff ~6e-6) |
| Bundle bất biến + checksum | ✅ Đã làm | `model_bundles/<species>_.../` |
| FastAPI + PhytoScan frontend + /api/consult (LLM) | ✅ Đã làm | Serving qua HF Hub (Mục 7.5, 7.10) |
| **Thresholds reject** (`T_low_confidence`, `T_margin`, `T_energy`, `binary_leaf`) | ❌ Chưa làm | Chưa có `thresholds.json` → energy-OOD & ambiguous/low-confidence hiện dùng default, thực tế **chưa kích hoạt** |
| **Calibration đạt ECE ≤ 0.05** | ⚠️ Chưa đạt | Xem số thực tế Mục 5.3; cần siết threshold để bù |
| **Golden set đầy đủ metadata** | ⚠️ Tối giản | Hiện chỉ là danh sách `{"images": [...]}`, chưa có `expected_top1`/`min_confidence` (Mục 2.7) |
| Few-shot / prototype (Chế độ 2) | 🔵 Tùy chọn / mở rộng | Code còn nhưng **không kích hoạt** vì không còn lớp < 30 (Mục 3.5) |
| Domain adaptation (field-bg recompose, MMD) | 🔵 Tùy chọn | `domain_mmd.enabled=false` mặc định; field-bg recompose không bật ở pretrain hiện tại |
| Cross-test Chiều 2 (public→field) | ✅ Đã chạy | `run_direction2.py`; DeiT-39 + backbone-38 vs model mình (Mục 6.5) |
| Cross-test Chiều 1 (mình→PlantDoc) | 🔵 Tạm gác | Cần de-dup hash (FINETUNE đã chứa 742 ảnh PlantDoc) |
| Robustness / OOD eval suite (Mục 5.5–5.6) | 🔵 Khung | Chỉ có định nghĩa corruption/metric, chưa có runner đầu-cuối |
| Ablation B0 / Swin-T / DINOv2 (E7–E9) | 🔵 Tùy chọn | Chưa chạy |
| Resume training (`--resume`) | ❌ Chưa làm | `checkpoints.py` có helper nhưng train script chưa dùng |

---

## 2. Dataset Analysis

### 2.0. Pipeline tiền xử lý dữ liệu thô (upstream — trước Phase 0)

> **Sơ bộ (đối chiếu code trong `raw_data_processing/`, 2026-07-12).** Đây là bước **trước** khi có `1. PRETRAIN DATASET` và `2. FINETUNE DATASET`. Mục tiêu: từ **11 dataset thô** gộp lại, tách ảnh field-leaf sạch để fine-tune, giữ PlantVillage làm pretrain. Có **hai thế hệ code** trong thư mục; V2 (grouping theo dataset) thay thế V1 (dump-all rồi CLIP).

**11 dataset nguồn** (`raw_datasets_structure.txt`):

| Dataset | Ảnh thô | Nhóm xử lý |
|---|---:|---|
| PlantVillage | 54,305 | **PRETRAIN** (không qua pipeline field) |
| Master Plant Disease Dataset | 66,953 | LAB_SUSPEN |
| Plant Disease Classification Merged Dataset | 79,086 | LAB_SUSPEN |
| Tomato-Village-main | 24,546 | LAB_SUSPEN |
| TomatoDataset | 1,609 | LAB_SUSPEN |
| Niphad Grape Leaf Disease Dataset (NGLD) | 2,726 | LAB_SUSPEN |
| plantseg | 22,916 | FRUIT_SUSPEN |
| PlantDoc | 2,922 | ONFIELD |
| FieldPlant | 5,156 | ONFIELD |
| Plant Pathology 2020 | 3,642 | ONFIELD |
| Plant Pathology 2021 | 18,635 | ONFIELD |

**Ý tưởng pipeline (đúng thứ tự):**

| # | Bước | File (V2) | Vào → Ra |
|---|---|---|---|
| **1** | Map label + chọn candidate species (loài PV có **≥1 nhãn healthy + ≥1 nhãn disease**) + phân 3 nhóm + CLIP theo nhóm | `datasets_grouping(for_classify_images).py` | 11 dataset thô → `ONFIELD_LEAF_DETECTED` + `LAB_LEAF_DETECTED` |
| **2** | Prune: xóa nhãn LAB không có trong ONFIELD; xóa loài thiếu healthy/disease | `del_unrelevant_label.py` (`finalize_candidates`) | prune tại chỗ 2 thư mục |
| **3** | Đổi tên | *(thủ công)* | `ONFIELD_LEAF_DETECTED` → **`2. FINETUNE DATASET`**; `PlantVillage` → **`1. PRETRAIN DATASET`** |
| **4** | Dedup exact (cả 2 tập) + near-duplicate (chỉ FINETUNE) — **đã chạy** | *(script ngoài `raw_data_processing/`)* | tập sạch cuối |

**Logic phân nhóm ở Bước 1** (quyết định cách lọc từng dataset):

- **ONFIELD** (field chắc chắn → copy thẳng, **không** CLIP): PlantDoc, FieldPlant, Plant Pathology 2020, Plant Pathology 2021.
- **LAB_SUSPEN** (nghi lẫn ảnh lab → CLIP nhị phân lab/field: field → ONFIELD, lab → LAB): Master Plant Disease Dataset, Plant Disease Classification Merged Dataset, Tomato-Village-main, TomatoDataset, NGLD.
- **FRUIT_SUSPEN** (nghi lẫn ảnh quả → CLIP fruit/leaf trước; nếu leaf mới CLIP lab/field; nếu fruit thì bỏ): plantseg.
- **PRETRAIN**: PlantVillage giữ nguyên (chỉ đổi tên, dùng cho Phase 1).

> **Code V1 đã bị thay thế (giữ để tham chiếu):** `combined_all_into_a_dataset(except_PV).py` (dump toàn bộ trừ PV vào `Field_Raw`, không tách theo dataset), `classify_lab_field_clip.py` (CLIP lab/field mù trên `Field_Raw` → `Field_Clean`/`Lab_Detected`), `classification_log.csv` (log chạy V1). V1 **không** có bước lọc quả và CLIP chạy trên tập đã gộp; V2 lọc theo từng dataset nên chính xác hơn. File `lab_field_fruit_leaf_split.py` gần trùng 1:1 với `datasets_grouping(...)` → nên archive một bản.

**Kết quả cuối** (`processed_datasets_structure.txt`): `1. PRETRAIN DATASET` = **53,599 ảnh / 38 lớp** (train+val kiểu PlantVillage); `2. FINETUNE DATASET` = **18,354 ảnh / 20 lớp field-leaf** trên 5 loài (Apple, Corn, Grape, Pepper bell, Tomato).

> **Lưu ý số liệu:** `processed_datasets_structure.txt` là **ảnh chụp trước khi bổ sung Spider_mites** (khi đó `Tomato/Spider_mites` = 7 ảnh, tổng FINETUNE = 18,354). Bản dữ liệu hiện dùng cho training đã tăng Spider_mites → **55 ảnh**, tổng **18,402** (Mục 2.2). Số 18,402 là con số chuẩn cho mọi bảng split/metric bên dưới.

### 2.1. Phạm vi dataset & ranh giới nguồn

Dự án dùng **hai lớp dữ liệu** với vai trò khác nhau. Lớp PRETRAIN hữu ích để học đặc trưng texture lá/bệnh chung; lớp FINETUNE mới là **miền đích thật** cho triển khai field. Hai lớp không được trộn tuỳ tiện: ảnh kiểu PlantVillage có thể thổi phồng validation accuracy nếu lọt vào protocol test cuối.

| Lớp | Đường dẫn local | Ảnh | Lớp | Loài | Miền | Được dùng | Không được dùng |
|---|---|---:|---:|---:|---|---|---|
| PRETRAIN | `D:\DAT301m_Project\Final_Pipeline\data\1. PRETRAIN DATASET` | 53,599 | 38 | 14 | Studio/lab, nền kiểm soát | Pretrain đặc trưng, sanity check backbone | Tuyên bố sẵn sàng field |
| FINETUNE | `D:\DAT301m_Project\Final_Pipeline\data\2. FINETUNE DATASET` | 18,402 | 20 | 5 | Ảnh field thực tế | Fine-tune per-species, calibration, validation, holdout khoá | Tái dùng holdout để tuning |

> Lưu ý đường dẫn: dữ liệu nguồn đặt tại `D:\DAT301m_Project\Final_Pipeline\data\...`; toàn bộ code, manifest, checkpoint, model và report của bản Final cũng nằm dưới root `D:\DAT301m_Project\Final_Pipeline`. Report cuối phải nêu rõ **target production là hiệu năng FINETUNE-field**; metric PRETRAIN chỉ là chẩn đoán phụ trợ.

### 2.2. Phân tích phân phối lớp & mất cân bằng

FINETUNE có 18,402 ảnh nhưng phân phối rất không đều: nhẹ với Apple/Grape/Pepper, nặng với Corn, và **mất cân bằng cao với Tomato** vì lớp ít nhất `Spider_mites` chỉ có 55 ảnh field (so với 596 của lớp lớn nhất). Sau khi dữ liệu Spider_mites tăng từ 7 → 55 ảnh, **không còn lớp nào dưới 30 ảnh** — mọi lớp đều đủ để train softmax head chuẩn.

| Loài | Tổng ảnh | #Lớp | Min class | Max class | Mean | Median | Imbalance ratio | Lớp hiếm `<30` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Apple | 12,778 | 3 | 2,008 | 5,555 | 4,259.3 | 5,215 | 2.77 | Không |
| Corn | 2,461 | 4 | 116 | 1,971 | 615.3 | 187.0 | 16.99 | Không |
| Grape | 204 | 2 | 71 | 133 | 102.0 | 102.0 | 1.87 | Không |
| Pepper bell | 268 | 2 | 97 | 171 | 134.0 | 134.0 | 1.76 | Không |
| Tomato | 2,691 | 9 | 55 | 596 | 299.0 | 304 | 10.84 | Không |
| **Tổng** | **18,402** | **20** | **55** | **5,555** | **920.1** | **211.0** | **101.0 global** | **0 lớp** |

Diễn giải:

- **Apple**: đủ dữ liệu cho model supervised chuẩn, nên là loài production-quality đầu tiên.
- **Corn**: cần xử lý imbalance vì `Northern_Blight` chiếm 80% loài.
- **Grape/Pepper**: dữ liệu nhỏ — backbone LR thấp + augment bảo thủ, nhạy với photographer/source bias.
- **Tomato**: các lớp bệnh dễ nhầm (`Early_blight`/`Late_blight`/`Septoria`/`Leaf_Mold`) và lớp ít nhất `Spider_mites` (55 ảnh) — vẫn cần theo dõi per-class recall.

Bảng chi tiết từng lớp:

| Loài | Lớp | Ảnh | Tỷ lệ trong loài | Rarity | Cách xử lý khuyến nghị |
|---|---|---:|---:|---|---|
| Apple | Apple_scab | 5,555 | 43.47% | Common | CE chuẩn / weighted CE ablation |
| Apple | Cedar_apple_rust | 2,008 | 15.72% | Medium | CE chuẩn / theo dõi recall |
| Apple | healthy | 5,215 | 40.81% | Common | CE chuẩn |
| Corn | Cercospora | 222 | 9.02% | Low | Class weights, oversampling nhẹ |
| Corn | Common_rust | 116 | 4.71% | Low | Class weights, focal-loss ablation |
| Corn | Northern_Blight | 1,971 | 80.09% | Dominant | Tránh majority collapse |
| Corn | healthy | 152 | 6.18% | Low | Class weights, theo dõi false disease rate |
| Grape | Black_rot | 133 | 65.20% | Low | Backbone LR thấp, augment bảo thủ |
| Grape | healthy | 71 | 34.80% | Low | Theo dõi recall, threshold thận trọng |
| Pepper bell | Bacterial_spot | 171 | 63.81% | Low | Backbone LR thấp, augment bảo thủ |
| Pepper bell | healthy | 97 | 36.19% | Low | Theo dõi recall, threshold thận trọng |
| Tomato | Bacterial_spot | 141 | 5.24% | Low | Weighted CE / focal ablation |
| Tomato | Early_blight | 497 | 18.47% | Medium | Lớp bệnh chuẩn |
| Tomato | Late_blight | 220 | 8.18% | Low | Theo dõi nhầm với Early_blight |
| Tomato | Leaf_Mold | 596 | 22.15% | Common | Lớp bệnh chuẩn |
| Tomato | Septoria | 304 | 11.30% | Medium | Theo dõi nhầm với Leaf_Mold |
| Tomato | Spider_mites | 55 | 2.04% | Low | Lớp ít nhất — theo dõi recall riêng |
| Tomato | TYLCV | 363 | 13.49% | Medium | Lớp bệnh chuẩn |
| Tomato | mosaic_virus | 135 | 5.02% | Low | Weighted CE, theo dõi nhầm với TYLCV |
| Tomato | healthy | 380 | 14.12% | Medium | Lớp bệnh chuẩn |

### 2.3. Ranh giới lớp thường / lớp hiếm (hiện không có lớp hiếm)

Bản gốc thiết kế hai chế độ với ngưỡng `rare_class_threshold = 30`: lớp `>= 30` ảnh train bằng softmax head, lớp `< 30` dùng prototype few-shot. **Với dữ liệu hiện tại, không lớp nào rơi vào Chế độ 2** — lớp ít nhất (`Tomato/Spider_mites`) đã có 55 ảnh (> 30). Vì vậy:

| Chế độ | Điều kiện | Phương pháp | Lớp thuộc chế độ (hiện tại) |
|---|---|---|---|
| **Chế độ 1 — lớp thường** | `>= rare_threshold` | Fine-tune softmax head + Class-Balanced Focal Loss (Mục 4) | **Toàn bộ 20 lớp / 5 loài** |
| **Chế độ 2 — lớp hiếm** | `< rare_threshold` | Prototype few-shot (Snell 2017) | **Không có lớp nào** |

Hệ quả cụ thể trong code: mọi manifest có khối `rare_folds` **rỗng**; `train_finetune.py` khi thấy `rare_folds` rỗng sẽ in "không có lớp hiếm — bỏ qua Chế độ 2" và **không sinh file prototype nào**. Nhánh few-shot vẫn còn trong code như **hướng mở rộng** cho tương lai (nếu thêm bệnh mới rất ít ảnh) — chi tiết ở Mục 3.5.

> Ghi chú ngưỡng: config `tomato` đặt `rare_class_threshold: 30` nhưng manifest tomato hiện lưu `20` (chạy trước đó); cả hai đều cho cùng kết quả vì lớp nhỏ nhất là 55 — không lớp nào dưới ngưỡng. `split_builder.py` nhận `--rare-threshold` để điều chỉnh nếu cần.

### 2.4. Chiến lược split & khoá holdout

Split phải **deterministic, stratified theo lớp, và khoá bằng manifest**. Áp dụng tỷ lệ **70/10/10/10**:

- `train`: chỉ dùng để fit model.
- `calib`: temperature scaling, confidence thresholds, OOD threshold tuning.
- `val`: chọn model, early stopping, so sánh ablation.
- `holdout`: field test cuối, **chỉ mở sau khi đã đóng băng mọi quyết định** model/bundle.

Dữ liệu đã được làm sạch ở nguồn (theo bối cảnh dự án), nên bản Final **không** thêm phase cleaning mới. Nhưng split vẫn cần manifest lock vì dữ liệu sạch vẫn có thể rò rỉ qua việc re-split cẩu thả hoặc chồng lấn nguồn.

Bảng phân bổ split (số thực tế từ manifest `*_split_v1.json`, seed 42):

| Loài | Lớp | Train 70% | Calib 10% | Val 10% | Holdout 10% | Tổng |
|---|---|---:|---:|---:|---:|---:|
| Apple | Apple_scab | 3,887 | 556 | 556 | 556 | 5,555 |
| Apple | Cedar_apple_rust | 1,405 | 201 | 201 | 201 | 2,008 |
| Apple | healthy | 3,649 | 522 | 522 | 522 | 5,215 |
| Corn | Cercospora | 156 | 22 | 22 | 22 | 222 |
| Corn | Common_rust | 80 | 12 | 12 | 12 | 116 |
| Corn | Northern_Blight | 1,380 | 197 | 197 | 197 | 1,971 |
| Corn | healthy | 107 | 15 | 15 | 15 | 152 |
| Grape | Black_rot | 94 | 13 | 13 | 13 | 133 |
| Grape | healthy | 50 | 7 | 7 | 7 | 71 |
| Pepper bell | Bacterial_spot | 120 | 17 | 17 | 17 | 171 |
| Pepper bell | healthy | 67 | 10 | 10 | 10 | 97 |
| Tomato | Bacterial_spot | 99 | 14 | 14 | 14 | 141 |
| Tomato | Early_blight | 347 | 50 | 50 | 50 | 497 |
| Tomato | Late_blight | 154 | 22 | 22 | 22 | 220 |
| Tomato | Leaf_Mold | 416 | 60 | 60 | 60 | 596 |
| Tomato | Septoria | 214 | 30 | 30 | 30 | 304 |
| Tomato | Spider_mites | 37 | 6 | 6 | 6 | 55 |
| Tomato | TYLCV | 255 | 36 | 36 | 36 | 363 |
| Tomato | mosaic_virus | 93 | 14 | 14 | 14 | 135 |
| Tomato | healthy | 266 | 38 | 38 | 38 | 380 |
| **Tổng** | **All** | **12,876** | **1,842** | **1,842** | **1,842** | **18,402** |

> **Không còn khối `rare_folds`:** với dữ liệu hiện tại mọi lớp đều `>= 30` ảnh nên tất cả đi theo split 70/10/10/10 chuẩn; `rare_folds` rỗng ở cả 5 manifest. Nếu về sau thêm lớp bệnh mới rất ít ảnh, `split_builder.py` sẽ tự tách chúng sang `rare_folds` (K-fold) và nhánh few-shot (Mục 3.5) sẽ được kích hoạt lại.

Quy tắc xây split:

| Quy tắc | Quyết định |
|---|---|
| Seed | Cố định `42` cho manifest v1; không tự tái sinh |
| Stratification | Theo loài và theo lớp |
| Grouping | Nếu có metadata nhóm ảnh/nguồn, group-split theo nhóm trước; nếu không, ghi `[Cần verify source grouping]` |
| Augmentation | Chỉ áp dụng **sau** split; không bao giờ vật chất hoá ảnh augment trước khi split |
| Tách calibration | `calib` không dùng cho early stopping hay chọn model |
| Truy cập holdout | Holdout read-only tới tận đánh giá cuối + golden smoke test |
| Tái sinh | Split mới phải đặt tên `split_v2`, kèm lý do và diff với `split_v1` |

### 2.5. Manifest schema & chống rò rỉ holdout

Manifest là "hợp đồng dữ liệu" bất biến, đảm bảo training / evaluation / export / serving dùng **cùng** class mapping, taxonomy và preprocessing version.

Trường bắt buộc của manifest:

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `schema_version` | string | Phải là `split_manifest_v1` |
| `created_at` | ISO datetime | Ví dụ `2026-07-04T00:00:00+07:00` |
| `seed` | integer | Cố định `42` cho v1 |
| `dataset_root` | string | `D:/DAT301m_Project/Final_Pipeline/data/2. FINETUNE DATASET` |
| `species` | string | `apple`, `corn`, `grape`, `pepper_bell`, `tomato` |
| `split_ratio` | object | `{train:0.70, calib:0.10, val:0.10, holdout:0.10}` |
| `class_to_idx` | object | Class mapping bất biến cho loài này. **Key là tên thư mục đầy đủ** (vd `Tomato___Bacterial_spot`), sinh bằng `sorted()` để ổn định |
| `species_dir` | string | Tên thư mục loài thật trên đĩa (vd `Corn_(maize)`, `Pepper,_bell`) |
| `rare_class_threshold` | integer | `30` mặc định (`--rare-threshold`) — lớp `<` ngưỡng vào `rare_folds` (Mục 2.3). Hiện không lớp nào dưới ngưỡng |
| `items[]` | array | Mỗi ảnh một dòng: path, hash, label, split, source trace |
| `rare_folds` | object | K-fold cho lớp hiếm (Chế độ 2), `{class: [[fold_idx, path], ...]}` |

Mỗi dòng `items[]` gồm: `image_id`, `relative_path`, `sha256`, `class_name`, `class_idx`, `split`, `source_dataset`, `source_group_id`. Trường source thiếu chỉ được phép khi ghi rõ `[Cần verify]`.

Policy chống rò rỉ holdout:

1. Không train / tuning threshold / calibration / chọn kiến trúc / chọn augmentation policy nào được dùng nhãn holdout.
2. Dữ liệu cross-test public phải đối chiếu với FINETUNE bằng **file hash** khi có thể; dữ liệu bị resize/đổi tên dùng **perceptual hash**.
3. Nếu phát hiện một ảnh holdout nằm trong train/calib/val sau khi train, experiment đó **bị vô hiệu** và chạy lại với manifest mới.
4. Nếu metadata nguồn không đầy đủ, report phải nói rõ thay vì tuyên bố độc lập nguồn nghiêm ngặt.

```python
# Final_Pipeline/src/data/split_builder.py
"""
Xây split manifest DETERMINISTIC cho FINETUNE DATASET (Phase 0, Mục 2.4-2.5).
- Lớp thường (>= rare_class_threshold): stratified 70/10/10/10 theo seed cố định.
- Lớp hiếm (< threshold): stratified K-fold vào khối rare_folds (Chế độ 2, Mục 2.3).
  (Hiện KHÔNG lớp nào < 30 nên rare_folds rỗng ở mọi manifest.)
Mọi ảnh được hash SHA256 để khoá holdout & phát hiện rò rỉ (Mục 2.5).

Chạy: python -m src.data.split_builder --species all --seed 42
"""
import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone, timedelta

IMG_EXT = (".jpg", ".jpeg", ".png")
SPECIES = ["apple", "corn", "grape", "pepper_bell", "tomato"]
DATASET_ROOT = r"D:/DAT301m_Project/Final_Pipeline/data/2. FINETUNE DATASET"
SPLITS_DIR = r"D:/DAT301m_Project/Final_Pipeline/data/splits"
RATIO = {"train": 0.70, "calib": 0.10, "val": 0.10, "holdout": 0.10}

# Key chuẩn nội bộ -> tên thư mục THẬT trên đĩa (có dấu ngoặc/phẩy).
SPECIES_DIR_NAME = {
    "apple": "Apple", "corn": "Corn_(maize)", "grape": "Grape",
    "pepper_bell": "Pepper,_bell", "tomato": "Tomato",
}


def resolve_species_dir(species: str) -> str:
    """Trả đường dẫn thư mục loài; nếu tên map sẵn không có thì dò không phân biệt hoa/thường."""
    candidate = SPECIES_DIR_NAME.get(species, species)
    path = os.path.join(DATASET_ROOT, candidate)
    if os.path.isdir(path):
        return path
    def norm(s):
        return (s.lower().replace("_", "").replace(",", "")
                .replace("(", "").replace(")", "").replace(" ", ""))
    target = norm(species)
    for name in os.listdir(DATASET_ROOT):
        full = os.path.join(DATASET_ROOT, name)
        if os.path.isdir(full) and (norm(name) == target or norm(name).startswith(target)):
            return full
    raise FileNotFoundError(f"Không tìm thấy thư mục loài '{species}' trong {DATASET_ROOT}.")


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def list_class_images(species_dir: str):
    """Trả {class_name: [abs_path, ...]} sắp xếp ổn định để tái lập. class_name = tên thư mục đầy đủ."""
    classes = sorted(c for c in os.listdir(species_dir)
                     if os.path.isdir(os.path.join(species_dir, c)))
    out = {}
    for cls in classes:
        cdir = os.path.join(species_dir, cls)
        out[cls] = sorted(os.path.join(cdir, f) for f in os.listdir(cdir)
                          if f.lower().endswith(IMG_EXT))
    return out


def stratified_split(paths, rng):
    """Chia 1 lớp thành train/calib/val/holdout theo RATIO, số lẻ dồn train trước."""
    paths = paths[:]; rng.shuffle(paths); n = len(paths)
    n_calib = int(round(n * RATIO["calib"])); n_val = int(round(n * RATIO["val"]))
    n_hold = int(round(n * RATIO["holdout"])); n_train = n - n_calib - n_val - n_hold
    idx, out = 0, {}
    for name, k in [("train", n_train), ("calib", n_calib), ("val", n_val), ("holdout", n_hold)]:
        out[name] = paths[idx:idx + k]; idx += k
    return out


def kfold_split(paths, rng, k=5):
    """Lớp hiếm: chia đều vào k fold (mỗi ảnh 1 fold) để đánh giá prototype (hiện không dùng)."""
    paths = paths[:]; rng.shuffle(paths)
    return [[i % max(1, min(k, len(paths))), p] for i, p in enumerate(paths)]


def build_species(species: str, seed: int, rare_threshold: int = 30):
    rng = random.Random(seed)
    species_dir = resolve_species_dir(species)
    class_images = list_class_images(species_dir)
    class_to_idx = {c: i for i, c in enumerate(sorted(class_images.keys()))}

    items, rare_folds = [], {}
    for cls, paths in class_images.items():
        if len(paths) < rare_threshold:
            rare_folds[cls] = kfold_split(paths, rng)
            continue
        for split_name, split_paths in stratified_split(paths, rng).items():
            for p in split_paths:
                items.append({
                    "image_id": os.path.splitext(os.path.basename(p))[0],
                    "relative_path": os.path.relpath(p, DATASET_ROOT).replace("\\", "/"),
                    "sha256": sha256_file(p),
                    "class_name": cls, "class_idx": class_to_idx[cls],
                    "split": split_name, "source_dataset": "FINETUNE",
                    "source_group_id": "[Cần verify]",
                })

    manifest = {
        "schema_version": "split_manifest_v1",
        "created_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "seed": seed, "dataset_root": DATASET_ROOT, "species": species,
        "species_dir": os.path.relpath(species_dir, DATASET_ROOT).replace("\\", "/"),
        "split_ratio": RATIO, "class_to_idx": class_to_idx,
        "rare_class_threshold": rare_threshold, "items": items, "rare_folds": rare_folds,
    }
    os.makedirs(SPLITS_DIR, exist_ok=True)
    out_path = os.path.join(SPLITS_DIR, f"{species}_split_v1.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[{species}] {len(items)} ảnh thường + {len(rare_folds)} lớp hiếm -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Xây split manifest deterministic (Phase 0)")
    ap.add_argument("--species", default="all", help="'all' hoặc tên 1 loài")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rare-threshold", type=int, default=30,
                    help="lớp < ngưỡng này -> rare_folds (few-shot). Đặt thấp để gộp vào head thường.")
    args = ap.parse_args()
    targets = SPECIES if args.species == "all" else [args.species]
    for sp in targets:
        build_species(sp, args.seed, rare_threshold=args.rare_threshold)
```

### 2.6. Phân tích domain gap (lab → field)

| Chiều | PRETRAIN lab/studio | FINETUNE field | Rủi ro |
|---|---|---|---|
| Nền | Trơn, thường đồng nhất | Đất, trời, tay, lá khác, lộn xộn | Model học shortcut theo nền khi pretrain |
| Ánh sáng | Đều, kiểm soát | Nắng gắt, bóng, thiếu sáng, ngược sáng | Confidence bất ổn khi đổi độ sáng |
| Vị trí lá | Một lá ở giữa | Lá một phần, nhiều lá, xoay, che khuất | Vùng bệnh có thể nhỏ / lệch tâm |
| Chất lượng ảnh | Sắc nét | Mờ, nén, biến thiên camera điện thoại | False low-confidence, texture sai |
| Mức độ bệnh | Triệu chứng rõ | Triệu chứng sớm, hỗn hợp, gần giống healthy | Nhầm giữa các lớp tương tự |
| Prior loài | 14 loài nhưng nhãn lab | 5 loài đích do user chọn | Vẫn có thể upload sai loài |

Loài rủi ro cao: **Corn** (lá dài, texture, chụp trong ruộng dày; `Northern_Blight` áp đảo nên shortcut nền/hình dạng lá dễ che khuyết điểm lớp nhỏ); **Grape/Pepper** (tập field rất nhỏ, nhạy source bias); **Tomato** (nhãn bệnh gần nhau về thị giác).

Biện pháp giảm domain gap — **đang dùng** vs **tùy chọn** (code có sẵn, chưa bật):

- ✅ Dùng PRETRAIN **chỉ** để khởi tạo backbone; dùng val/holdout **chỉ từ field** cho mọi production metric.
- ✅ Augmentation photometric (RandAugment, ColorJitter, JPEG, noise) khi fine-tune (Mục 4.3).
- ✅ Calibration + khung OOD/reject để lỗi field **hiện ra** thay vì bị giấu sau top-1 accuracy.
- 🔵 **Field-background recomposition** (Jeon et al., 2025): code có ở `domain_augmentation.py` nhưng **không bật** ở pretrain hiện tại (`domain_aug` chưa kích hoạt). Hướng cải thiện: bằng chứng gốc +9.9–23.6 điểm % trên 4/5 loài.
- 🔵 **MMD loss** (Wu et al., 2023): code có, `domain_mmd.enabled=false` mặc định.
- ✅ **Đo domain gap định lượng (Chiều 2 — ĐÃ CHẠY):** model chỉ-biết-lab (DeiT-39 ngoài, backbone-38 của mình) chạy trên holdout field → macro-F1 0.479 và 0.312 so với **0.595** của model fine-tune field. Xác nhận domain gap (Mục 6.5).
- 🔵 **Chiều 1 trên PlantDoc (lab→field zero-shot/+adaptation):** là kế hoạch, chưa chạy (cần de-dup hash).

### 2.7. Định nghĩa Golden Set

Golden set là **tập smoke-test nhỏ, cố định** cho an toàn triển khai — **khác** với holdout. Holdout đo hiệu năng khoa học; golden set bắt lỗi triển khai: preprocessing hỏng, class mapping bị hoán đổi, ONNX export sai, load nhầm bundle.

| Loại lớp | Ảnh mỗi lớp | Ghi chú |
|---|---:|---|
| Common `>=300` ảnh | 20–30 | Trộn ca dễ và khó |
| Low `100–299` ảnh | 10–15 | Đủ ca để kiểm mapping |
| Rare `<100` ảnh | 3–7 nếu có | Không tuyên bố ý nghĩa thống kê |
| Tomato Spider_mites | 1–3 | Chỉ dùng làm canary, không phải bằng chứng metric |

> **Trạng thái hiện tại:** golden set đang ở dạng **tối giản** — mỗi loài một file `data/golden/<species>.json` chỉ chứa `{"images": [đường dẫn, ...]}` (30 ảnh/loài, dùng cho parity test top-1 PyTorch↔ONNX ở Mục 7.4). Schema đầy đủ dưới đây (với `expected_top1`, `min_confidence`, `must_pass_leaf_gate`...) là **mục tiêu hoàn thiện**, chưa được điền.

Trường golden item (mục tiêu): `id`, `species`, `image_path`, `sha256`, `expected_top1`, `allowed_top3`, `min_confidence`, `must_pass_leaf_gate`, `notes`. Ca canonical yêu cầu **top-1 chính xác**; ca mập mờ chỉ chấp nhận top-3 khi đánh dấu rõ.

Quy tắc pass golden:

- Leaf gate phải chấp nhận mọi ảnh lá hợp lệ trong golden.
- Top-1 phải khớp cho ảnh canonical/dễ.
- Bất kỳ sai lệch class mapping nào là **release blocker**.
- Golden set fail **chặn deploy** ngay cả khi metric validation tổng thể đẹp.

> Golden set được version cùng bundle (Mục 7). Ảnh golden **không** trùng holdout; chọn từ train/val để tránh tiêu tốn holdout.

---

## 3. Model Architecture

### 3.1. VRAM budget & lựa chọn backbone

GPU chỉ 4GB VRAM, nên batch size là tài nguyên được kiểm soát chặt: AMP FP16 + gradient accumulation + partial freezing trước khi thử kiến trúc nặng hơn. Các số dưới đây là **ước lượng lập kế hoạch**; usage thật phải verify bằng `torch.cuda.max_memory_allocated()` trên chính laptop (Mục 4.4, Appendix C).

| Backbone | Params | Input | Full FT BS16 AMP | Full FT BS32 AMP | Khuyến nghị 4GB | Ghi chú |
|---|---:|---:|---|---|---|---|
| EfficientNetV2-B0 | ~7M | 224 | Feasible | Feasible | **Fallback / iterate nhanh** | Nhẹ, ít OOM, accuracy thấp hơn S |
| MobileNetV3-Large | ~5.4M | 224 | Feasible | Feasible | Fallback nhẹ hơn nữa | Rất nhanh |
| MobileViT-XS/S | ~2–6M | 224 | Feasible | Possible | Ablation hybrid | Cần AMP bắt buộc |
| ResNet50 | 25.6M | 224 | Feasible | Risky | Baseline only | Optimizer state lớn hơn EffV2-S |
| **EfficientNetV2-S** | **~21–22M** | **224** | **Feasible** | **BS16 + accum** | **PRIMARY** | **Accuracy tốt nhất trong ngân sách 4GB** |
| DINOv2-S/14 frozen | 22M backbone | 224 | Feasible (frozen) | Possible (frozen) | Embedding lớp hiếm | Không backprop backbone → rẻ VRAM |
| ConvNeXt-Tiny | ~28M | 224 | Feasible nhưng tight | Risky | Secondary | Áp lực memory cao hơn |
| Swin-T | 28M | 224 | Tight | Not recommended | Ablation only | Attention memory spike |
| EfficientNetV2-M / Swin-S / DINOv2-B (full FT) | 49–86M | 224+ | OOM likely | No | **Tránh** | Để cho phần cứng mạnh hơn |

**Vì sao EfficientNetV2-S là backbone chính (D1 — ưu tiên max accuracy):**

1. Trong nhóm khả thi trên 4GB, EffV2-S cho accuracy/macro-F1 downstream cao nhất — mạnh hơn rõ B0/MobileNetV3 trên các task fine-grained như phân biệt bệnh lá.
2. Inductive bias CNN mạnh cho đốm, vết, texture cục bộ và biên lá.
3. Export ONNX + parity inference dễ hơn transformer.
4. Đường đi sạch từ minimal baseline → production candidate mà không đổi serving stack.
5. So sánh được với literature PlantVillage và public CNN baselines.

**Khả thi trên 4GB nhờ:** EffV2-S (~21M params ≈ 86MB FP32 weights, ~172MB AdamW state) — params **không** phải bottleneck; **activation** mới là biến lớn. Kiểm soát activation bằng: batch physical 16 + AMP FP16 (giảm ~½ activation) + gradient accumulation (effective 32–64) + **PEFT** (chỉ mở head+BN+block cuối → optimizer state giảm ~20×). Với Grape/Pepper (data nhỏ) dùng **backbone LR thấp `5e-6`** + augment bảo thủ + early stopping thay vì đóng băng cứng. Nếu probe VRAM vẫn OOM ở BS8, hạ về **B0 fallback** (D1).

Operational VRAM target:
- Training peak phải `< 3.6GB` để chừa cho Windows display + driver.
- Mặc định: `batch_size=16`, AMP FP16, `accumulation_steps=2–4`, effective batch `32–64`.
- OOM → hạ `batch_size=8`, `accumulation=4–8`, freeze thêm block sớm.
- Grape/Pepper: backbone LR thấp + early stopping (xem `configs/finetune_grape.yaml`, `finetune_pepper_bell.yaml`).

Bảng backbone tóm tắt:

| Kiến trúc | Params | Vai trò trong bản Final |
|---|---:|---|
| EfficientNetV2-S | ~21M | **Primary** cho pretrain + fine-tune 5 loài |
| EfficientNetV2-B0 | ~7M | Fallback khi OOM / cần iterate nhanh |
| MobileNetV3-Small | ~2.5M | Leaf Gate (Mục 3.2) |
| ~~DINOv2-S/14 (frozen)~~ | 22M | ❌ Không dùng — dữ liệu không còn lớp hiếm (few-shot chỉ là hướng mở rộng, Mục 3.5) |
| ~~MobileViT-XS / Swin-T~~ | 2–28M | 🔵 Ablation tùy chọn, chưa chạy |
| ~~EffV2-M / Swin-S / DINOv2-B full FT~~ | 49–86M | ❌ Tránh trên 4GB |

Factory tạo backbone (dùng chung pretrain + fine-tune):

```python
# Final_Pipeline/src/models/backbone.py
"""Factory backbone qua timm. Primary: EfficientNetV2-S (D1). Fallback: B0/MobileNetV3."""
import timm
import torch.nn as nn


def build_backbone(num_classes: int, arch: str = "tf_efficientnetv2_s.in1k",
                   pretrained: bool = True, drop_path_rate: float = 0.1) -> nn.Module:
    """
    arch khuyến nghị theo ưu tiên:
      - "tf_efficientnetv2_s.in1k"       : MẶC ĐỊNH — accuracy tốt nhất trong ngân sách 4GB
      - "tf_efficientnetv2_b0.in1k"      : fallback khi OOM / iterate nhanh
      - "mobilenetv3_large_100.ra_in1k"  : fallback nhẹ hơn nữa
      - "mobilevit_xs.cvnets_in1k"       : ablation hybrid CNN-Transformer (cần AMP)
    drop_path_rate: stochastic depth chống overfit khi data ít.
    """
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes,
                             drop_path_rate=drop_path_rate)


def build_leaf_gate(pretrained: bool = True) -> nn.Module:
    # MobileNetV3-Small — nhẹ, đủ cho bài toán nhị phân leaf/not-leaf (Mục 3.2)
    return timm.create_model("mobilenetv3_small_100.lamb_in1k",
                             pretrained=pretrained, num_classes=2)
```

### 3.2. Thiết kế Leaf Gate

Leaf Gate là bộ phân loại nhị phân **trước** disease classification. Nó chặn input rõ ràng không hợp lệ và ngăn disease head tự tin phân loại ảnh không phải lá.

| Thành phần | Quyết định |
|---|---|
| Input | Ảnh RGB sau secure decode + fix EXIF orientation |
| Output | `leaf_prob` trong `[0,1]` |
| Kiến trúc | MobileNetV3-Small (nhị phân) |
| Loss | Binary cross entropy + balanced sampling |
| Calibration | Temperature scaling (scalar logit) trên calib data |
| Quyết định mặc định | Reject `not_a_leaf` nếu `leaf_prob < T_leaf` (đã calibration) |

Dữ liệu train Leaf Gate:

| Lớp | Nguồn | Số lượng mục tiêu | Ghi chú |
|---|---|---:|---|
| Leaf positive | PRETRAIN + FINETUNE train split | 10k–20k sampled | Cân bằng loài, tránh ảnh holdout |
| Leaf hard positive | Lá field có clutter/occlusion | 1k–2k | Chỉ từ FINETUNE train |
| Non-leaf negative | COCO/ImageNet/OpenImages-like | 5k–10k `[Cần verify license]` | Đất, tay, trời, quả, dụng cụ, vật ngẫu nhiên |
| Plant but not leaf | thân, hoa, quả, cả cây | 1k `[Cần verify]` | Quan trọng cho web upload |

Metric Leaf Gate:

| Metric | Mục tiêu | Vì sao |
|---|---:|---|
| Leaf recall trên lá field hợp lệ | `>=98%` | Tránh từ chối oan người dùng |
| Non-leaf rejection rate | `>=95%` trên negatives curated | Chặn upload không hợp lệ |
| ECE | `<=0.05–0.08` | Probability điều khiển quyết định API |
| False reject | Review thủ công | Vấn đề niềm tin người dùng |

**Nâng cấp từ V2 — Energy-based OOD làm lớp bảo vệ thứ hai** (Liu et al., 2020): tính điểm năng lượng trực tiếp từ logits của model loài đã train, **không cần** tập non-leaf riêng. Điểm năng lượng có cơ sở lý thuyết gắn với mật độ xác suất, ít bị overconfidence hơn softmax; kết hợp AND với binary gate.

```python
# Final_Pipeline/src/models/energy_ood.py
"""Energy-based OOD (Liu, Wang, Owens, Li, 2020 — https://arxiv.org/abs/2010.03759)."""
import torch


def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Điểm năng lượng: THẤP (âm nhiều) = giống in-distribution (lá quen thuộc)."""
    return -T * torch.logsumexp(logits / T, dim=1)


def leaf_gate_decision(binary_logits, species_logits, energy_threshold,
                       binary_threshold=0.5, leaf_T=1.0):
    """Kết hợp binary gate VÀ energy score từ model loài (lớp bảo vệ thứ hai).
    binary_threshold & energy_threshold chọn từ calib split (Mục 5), KHÔNG cảm tính."""
    p_leaf = torch.softmax(binary_logits / leaf_T, dim=1)[:, 1]
    is_leaf = p_leaf > binary_threshold
    is_familiar = energy_score(species_logits) < energy_threshold
    return is_leaf & is_familiar
```

### 3.3. Disease Classification Heads (per-species)

Thiết kế production dùng **một model bệnh mỗi loài** thay vì một model 20-lớp (D6). Người dùng đã chọn loài, nên per-species head giảm nhầm lẫn xuyên loài và đơn giản hoá class mapping.

| Loài | #Lớp head | File model | Thứ tự class index (bất biến) |
|---|---:|---|---|
| Apple | 3 | `apple_effv2s.onnx` | `Apple_scab`, `Cedar_apple_rust`, `healthy` |
| Corn | 4 | `corn_effv2s.onnx` | `Cercospora`, `Common_rust`, `Northern_Blight`, `healthy` |
| Grape | 2 | `grape_effv2s.onnx` | `Black_rot`, `healthy` |
| Pepper bell | 2 | `pepper_bell_effv2s.onnx` | `Bacterial_spot`, `healthy` |
| Tomato | 9 | `tomato_effv2s.onnx` | `Bacterial_spot`, `Early_blight`, `Late_blight`, `Leaf_Mold`, `Septoria`, `Spider_mites`, `TYLCV`, `mosaic_virus`, `healthy` |

Thiết kế head: EfficientNetV2-S backbone → global pooled feature → dropout `p=0.2` → linear classifier → temperature scaling lúc inference.

Per-species thresholds (thiết kế — **hiện chưa build `thresholds.json`**, Mục 1.8/5.4):

| Threshold | Ý nghĩa | Tune trên |
|---|---|---|
| `T_low_confidence` | Xác suất top-1 (đã calibration) tối thiểu | `calib` split |
| `T_margin` | Chênh top-1 trừ top-2 tối thiểu | `calib` split |
| `T_energy` | Ngưỡng energy OOD | `calib` + OOD suite |
| `binary_leaf` | Ngưỡng leaf-gate | PR curve leaf-gate |

> **Lưu ý mapping:** class index bất biến do `split_builder.py` sinh; `class_mapping.json` trong bundle lưu `idx_to_class` với **tên thư mục đầy đủ** (vd `Tomato___Bacterial_spot`). Bảng trên dùng tên rút gọn cho dễ đọc; frontend dịch sang nhãn hiển thị song ngữ qua `api_compat.py` (Mục 7.10).

> **Lưu ý routing:** định tuyến loài **chỉ** do lựa chọn của người dùng + repo registry (HF Hub) quyết định. Không tự động chuyển ảnh sang model loài khác.

### 3.4. Head Tomato & lớp Spider_mites

Tomato head khai báo đủ **9 lớp** (kể cả `Spider_mites`) để giữ class mapping bất biến và tương thích ONNX/API. Với dữ liệu hiện tại `Spider_mites` có **55 ảnh** (train 37) — đủ để nhánh softmax học lớp này như một lớp thường (Chế độ 1), tuy vẫn là lớp ít mẫu nhất của Tomato. Report nên **báo per-class recall/precision riêng cho `Spider_mites`** và các lớp bệnh dễ nhầm, không cần nhánh prototype riêng nữa.

### 3.5. Few-shot / Prototype — hướng mở rộng (hiện KHÔNG kích hoạt)

> **Trạng thái:** với dữ liệu hiện tại **không lớp nào < 30 ảnh**, nên nhánh few-shot/prototype (Chế độ 2) **không chạy** — `train_finetune.py` thấy `rare_folds` rỗng sẽ bỏ qua và không sinh file prototype. Phần này giữ lại làm **hướng mở rộng**: nếu về sau thêm bệnh mới rất ít ảnh (`< rare_class_threshold`), pipeline sẽ tự tách chúng sang `rare_folds` và kích hoạt lại nhánh này.

Ý tưởng (Snell et al., 2017): thay vì train softmax head cho lớp cực ít ảnh (dễ overfit), tính **prototype** = embedding trung bình (đã L2-normalize) của mỗi lớp trên backbone đã fine-tune, rồi phân loại query bằng cosine similarity tới prototype gần nhất. Ưu điểm ở chế độ dữ liệu cực ít: ổn định hơn softmax head và cho phép trả `unknown_or_ood` khi query xa mọi prototype.

Code liên quan vẫn còn trong repo để tái kích hoạt khi cần:
- `src/train/few_shot.py`: `compute_prototypes()`, `classify_by_prototype()`.
- `src/train/train_finetune.py`: `run_few_shot_evaluation()` (K-fold, chỉ chạy khi `rare_folds` không rỗng).

```python
# Final_Pipeline/src/train/few_shot.py  (giữ nguyên — chỉ chạy khi có lớp < ngưỡng)
"""Prototypical classification cho lớp < rare_threshold (Snell, Swersky, Zemel, 2017
   — https://arxiv.org/abs/1703.05175). Không train tham số riêng cho lớp hiếm."""
import torch
import torch.nn.functional as F


def compute_prototypes(support_features: torch.Tensor, support_labels: torch.Tensor,
                       num_classes: int) -> torch.Tensor:
    """support_features [N, D] -> prototypes [num_classes, D] (mean đã L2-normalize)."""
    feats = F.normalize(support_features, dim=1)
    protos = torch.zeros(num_classes, feats.size(1))
    for c in range(num_classes):
        mask = support_labels == c
        if mask.sum() > 0:
            protos[c] = F.normalize(feats[mask].mean(dim=0), dim=0)
    return protos


def classify_by_prototype(query_features: torch.Tensor, prototypes: torch.Tensor):
    """Cosine similarity tới prototype gần nhất -> softmax score."""
    q = F.normalize(query_features, dim=1)
    sim = q @ prototypes.t()          # [Nq, num_classes], cosine vì đã normalize
    return sim.softmax(dim=-1)
```

Guardrail (nếu tái kích hoạt): prototype **không** route ảnh sang model loài khác; species routing chỉ do user selection + registry.

---

## 4. Training Pipeline

Section này ghép mọi kỹ thuật rời rạc thành pipeline chạy được đầu-cuối: cấu trúc thư mục → config → augmentation → loss → domain adaptation → memory utils → datasets → script pretrain → script fine-tune → checkpoint/resume → experiment matrix.

Bảng truy vết kỹ thuật → nguồn:

| Bước pipeline | Kỹ thuật | Tham chiếu |
|---|---|---|
| Backbone | EfficientNetV2-S, PEFT-lite | Mục 3.1, 4.8 |
| Pretrain 38 lớp | AMP + grad accumulation, CutMix/Mixup | Mục 4.5, 4.6, 4.2 |
| Fine-tune lớp thường | Class-Balanced Focal Loss, CutMix/Mixup, PEFT 3 bước, decoupled re-training | Mục 4.2, 4.3, 4.7, 4.9 |
| Calibration | Temperature scaling, ECE | Mục 5 |
| 🔵 Field-BG recompose / MMD (tùy chọn) | Domain adaptation | Mục 4.6 (chưa bật) |
| 🔵 Few-shot prototype (mở rộng) | Prototypical Networks | Mục 3.5 (không kích hoạt) |

### 4.1. Cấu trúc thư mục

```text
D:\DAT301m_Project\Final_Pipeline\
├── configs/
│   ├── pretrain_effv2s.yaml
│   ├── finetune_apple.yaml
│   ├── finetune_corn.yaml
│   ├── finetune_grape.yaml
│   ├── finetune_pepper_bell.yaml
│   ├── finetune_tomato.yaml
│   ├── leaf_gate.yaml               # Mục 4.12
│   └── taxonomy_v1.yaml
├── data/
│   ├── splits/                      # sinh bởi split_builder.py (Mục 2.5)
│   │   ├── apple_split_v1.json
│   │   ├── corn_split_v1.json
│   │   ├── grape_split_v1.json
│   │   ├── pepper_bell_split_v1.json
│   │   └── tomato_split_v1.json
│   ├── leaf_gate/                   # leaf/not_leaf train|calib|val (Mục 4.12)
│   └── golden/                      # golden set (Mục 2.7)
├── src/
│   ├── config.py                    # Mục 4.1
│   ├── data/
│   │   ├── split_builder.py         # Mục 2.5
│   │   ├── datasets.py              # Mục 4.6
│   │   ├── augmentation.py          # Mục 4.2
│   │   └── domain_augmentation.py   # Mục 4.5
│   ├── models/
│   │   ├── backbone.py              # Mục 3.1 (+ build_leaf_gate)
│   │   └── energy_ood.py            # Mục 3.2
│   ├── train/
│   │   ├── losses.py                # Mục 4.4
│   │   ├── samplers.py              # Mục 4.4
│   │   ├── few_shot.py              # Mục 3.5
│   │   ├── domain_loss.py           # Mục 4.6
│   │   ├── memory_utils.py          # Mục 4.5
│   │   ├── peft.py                  # Mục 4.7
│   │   ├── checkpoints.py           # Mục 4.10
│   │   ├── train_pretrain.py        # Mục 4.8
│   │   ├── train_finetune.py        # Mục 4.9
│   │   └── train_leaf_gate.py       # Mục 4.12
│   ├── eval/
│   │   ├── evaluate.py              # Mục 5.2
│   │   ├── calibration.py           # Mục 5.3 (+ CLI --target)
│   │   ├── ood_eval.py              # Mục 5.5
│   │   └── robustness.py            # Mục 5.6
│   ├── cross_test/
│   │   ├── run_direction2.py       # Mục 6.5 — Chiều 2 (public model -> holdout field) ĐÃ CHẠY
│   │   ├── public_models.py        # adapter BackbonePVModel + DeiTPVModel
│   │   ├── label_maps.py           # map 20 lớp mình <-> PlantVillage-38/39
│   │   └── run_cross_test.py       # Mục 6.5b — helper Chiều 1 (PlantDoc, chưa chạy)
│   ├── export/
│   │   ├── export_onnx.py           # Mục 7.3
│   │   ├── parity_test.py           # Mục 7.4
│   │   └── build_bundle.py          # Mục 7.4b
│   ├── deploy/
│   │   ├── app.py                   # Mục 7.5 (FastAPI + validators)
│   │   ├── api_compat.py            # Mục 7.10 (/api/* cho frontend)
│   │   ├── logging_utils.py         # Mục 8.6
│   │   └── static/index.html        # Mục 7.9 (theme PhytoScan)
│   └── tests/                       # Mục 8.10 (pytest)
├── checkpoints/                     # model_<species>.pt, leaf_gate.pt, backbone_pretrained.pt
├── model_bundles/                   # bundle bất biến + active_version_<species>.txt (Mục 7-8)
├── experiments/                     # experiments/{experiment_id}/ (Mục 4.11, 8.9)
└── reports/
```

> Toàn bộ script dùng import tuyệt đối `from src...`; chạy bằng `python -m src.train.train_pretrain` từ root `Final_Pipeline` (Python ≥3.3 không bắt buộc `__init__.py`).

### 4.2. Cấu hình tập trung (config)

Mọi siêu tham số nằm trong YAML — tránh hard-code rải rác, dễ so sánh experiment (mỗi lần chạy chỉ đổi 1 file).

```yaml
# Final_Pipeline/configs/pretrain_effv2s.yaml
seed: 42
data:
  train_dir: "D:/DAT301m_Project/Final_Pipeline/data/1. PRETRAIN DATASET/train"
  val_dir:   "D:/DAT301m_Project/Final_Pipeline/data/1. PRETRAIN DATASET/val"
  img_size: 224
  num_workers: 2              # 4GB VRAM / 16GB RAM — KHÔNG dùng 8
model:
  arch: "tf_efficientnetv2_s.in1k"   # D1 — primary
  num_classes: 38
  drop_path_rate: 0.1
train:
  batch_size: 16
  accumulation_steps: 4       # effective batch = 64
  epochs: 30
  lr_head: 1.0e-3
  weight_decay: 1.0e-4
  label_smoothing: 0.1
  use_amp: true               # bắt buộc trên 4GB
  use_grad_checkpoint: false  # bật true nếu thử MobileViT/Swin
cutmix_mixup:
  enabled: true
  alpha: 0.2
  cutmix_prob: 0.5
  p: 0.5
domain_aug:                   # field-background recomposition (Mục 4.5)
  enabled: true
  p: 0.4
  field_background_dir: "D:/DAT301m_Project/Final_Pipeline/data/2. FINETUNE DATASET"
checkpoint:
  out_path: "D:/DAT301m_Project/Final_Pipeline/checkpoints/backbone_pretrained.pt"
  metric: "macro_f1"
```

```yaml
# Final_Pipeline/configs/finetune_tomato.yaml  (5 loài cùng khuôn, đổi species/đường dẫn)
seed: 42
species: "tomato"
data:
  split_manifest: "D:/DAT301m_Project/Final_Pipeline/data/splits/tomato_split_v1.json"
  img_size: 224
  num_workers: 2
  rare_class_threshold: 30    # ranh giới Chế độ 1 / Chế độ 2 (Mục 2.3)
model:
  arch: "tf_efficientnetv2_s.in1k"
  pretrained_ckpt: "D:/DAT301m_Project/Final_Pipeline/checkpoints/backbone_pretrained.pt"
train:
  batch_size: 16
  accumulation_steps: 4
  epochs_warmup: 5            # Bước 1 — head_only
  epochs_full: 25             # Bước 2 — head_bn_lastblock
  epochs_decoupled: 10        # Bước 3 — decoupled re-training (Kang et al., 2019)
  early_stop_patience: 7      # dừng sớm trong pha nếu val F1 không cải thiện
  lr_head: 1.0e-3
  lr_backbone: 1.0e-5
  weight_decay: 1.0e-4
  use_amp: true
  cutmix_mixup: {enabled: true, alpha: 0.2, cutmix_prob: 0.5, p: 0.5}
  class_balanced_loss: {beta: 0.9999, gamma: 2.0}
  domain_mmd:                 # MMD (Mục 4.5) — TẮT mặc định, bật khi thử nghiệm
    enabled: false
    lambda_max: 0.1
    warmup_epochs: 5
    source_dir: "D:/DAT301m_Project/Final_Pipeline/data/1. PRETRAIN DATASET/train"
checkpoint:
  out_path: "D:/DAT301m_Project/Final_Pipeline/checkpoints/model_tomato.pt"
  metric: "macro_f1"
```

```python
# Final_Pipeline/src/config.py
"""Loader YAML tối giản — tránh phụ thuộc Hydra/OmegaConf cho quy mô dự án cá nhân."""
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

# Dùng: cfg = TrainConfig.load("configs/finetune_tomato.yaml"); cfg["train"]["batch_size"]
```

### 4.3. Augmentation

Augmentation để **giảm domain gap**, không tạo pattern bệnh phi thực tế. Mọi policy phải log trong experiment config để tái lập preprocessing.

| Augmentation | Phase 1 PRETRAIN | Phase 2 FINETUNE | Cường độ | Ghi chú |
|---|---|---|---|---|
| RandomResizedCrop | Yes | Yes | Scale `0.7–1.0` | Giữ vùng bệnh nhìn thấy được |
| Horizontal/Vertical flip | Yes | Yes | `p=0.5 / 0.3` | An toàn cho hướng lá |
| RandAugment | Mild | Yes | `num_ops=2, magnitude=9` | Thay chuỗi augment thủ công dài |
| ColorJitter | Mild | Stronger | brightness/contrast/saturation | Giúp illumination shift field |
| Gaussian blur/noise | No/mild | Mild | `p<=0.15` | Mô phỏng blur điện thoại |
| JPEG compression | No | Mild | quality `50–95` | Robust với web upload |
| Random erasing/CoarseDropout | No | Mild | `p<=0.10` | Tránh xoá cả vùng bệnh |
| MixUp | Optional | Apple/Corn/Tomato ablation | alpha `0.1–0.2` | Không mặc định cho Grape/Pepper nhỏ |
| CutMix | Avoid default | Ablation | alpha `0.2–1.0` | Giữ vùng bệnh sắc nét hơn Mixup |
| Field-background recompose | Yes (Mục 4.5) | Research | `p~0.4` | Giảm domain gap; QC bằng Grad-CAM |

Policy theo loài:
- **Apple**: geometric + photometric chuẩn, MixUp ablation cho phép.
- **Corn**: class-balanced sampling mạnh hơn; tránh augment làm overrepresent context `Northern_Blight`.
- **Grape/Pepper**: augment bảo thủ vì data nhỏ; transform quá mạnh làm val bất ổn.
- **Tomato**: photometric robustness quan trọng; lớp hiếm không over-augment thành đa dạng giả.

```python
# Final_Pipeline/src/data/augmentation.py
"""Augmentation FINETUNE nâng cấp: RandAugment + CutMix/Mixup.
RandAugment: Cubuk et al., 2019 — https://arxiv.org/abs/1909.13719
CutMix: Yun et al., 2019 — https://arxiv.org/abs/1905.04899
Mixup:  Zhang et al., 2017 — https://arxiv.org/abs/1710.09412"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_finetune_transforms(train=True, img_size=224, strong=False):
    if train:
        aug = [
            A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandAugment(num_ops=2, magnitude=9, p=0.9),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.15),
            A.ImageCompression(quality_lower=50, quality_upper=95, p=0.2),
        ]
        if strong:   # lớp biên <30 ảnh — augment mạnh hơn, KHÔNG thay thế few-shot (Mục 3.5)
            aug += [A.CoarseDropout(max_holes=4, max_height=24, max_width=24, p=0.4)]
        aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
        return A.Compose(aug)
    return A.Compose([A.Resize(img_size, img_size),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def get_pretrain_transforms(train=True, img_size=224):
    """Ảnh lab nền sạch — augment nhẹ hơn (RandAugment không cần thiết)."""
    if train:
        return A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
            A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
    return A.Compose([A.Resize(img_size, img_size),
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
```

> Kết hợp với sampler (Mục 4.3-losses): áp CutMix/Mixup **sau** khi WeightedRandomSampler đã lấy mẫu cân bằng, để hai kỹ thuật bổ trợ nhau.

### 4.4. Losses, Samplers & xử lý imbalance

| Kỹ thuật | Dùng khi | Tránh khi | Ghi chú |
|---|---|---|---|
| Standard CE | Apple baseline, pretrain baseline | Imbalance nặng | Minimal baseline phải có |
| Weighted CE | Corn, Tomato, loài nhỏ | Weight quá lớn | Clamp weight tránh gradient bất ổn |
| Class-Balanced Focal | Imbalance + recall lớp hiếm kém | Calibration xấu đi nhiều | So sánh ECE, không chỉ F1 |
| WeightedRandomSampler | Minority bị bỏ qua | Lớp cực nhỏ overfit ngay | Dùng với augment bảo thủ |
| Prototype distance | Lớp <ngưỡng (Chế độ 2) — hiện không có | Thay thế cho mọi lớp | Chỉ hỗ trợ song song; hướng mở rộng (Mục 3.5) |

Bản Final dùng **Class-Balanced Loss** (Cui et al., 2019) thay inverse-frequency thuần: trọng số theo "effective number of samples" `(1−βⁿ)/(1−β)`, phản ánh đúng việc thêm 1 ảnh vào lớp 500 ảnh mang ít thông tin hơn thêm 1 ảnh vào lớp 5 ảnh. Kết hợp Focal (Lin et al., 2017) + label smoothing (Müller et al., 2019).

Quy tắc class weight (dùng chung cho cả sampler): `weight_c = sqrt(total / (num_classes * count_c))`, clamp `[0.25, 4.0]` — tránh weight cực lớn cho lớp nhỏ nhất. (Effective-number dùng trong loss; sqrt-clamp dùng khi cần công thức đơn giản đối chiếu.)

```python
# Final_Pipeline/src/train/losses.py
"""Class-Balanced Focal Loss.
Class-Balanced: Cui et al., 2019 — https://arxiv.org/abs/1901.05555
Focal:          Lin et al., 2017 — https://arxiv.org/abs/1708.02002
Label smoothing: Müller et al., 2019 — https://arxiv.org/abs/1906.02629"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_num_weights(class_counts, beta=0.9999):
    """(1 - beta^n) / (1 - beta), chuẩn hoá về trung bình 1.0."""
    counts = torch.tensor(class_counts, dtype=torch.float)
    eff_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / eff_num
    return weights / weights.sum() * len(counts)


class ClassBalancedFocalLoss(nn.Module):
    """Cho targets là CHỈ SỐ LỚP nguyên (val, hoặc train không CutMix)."""
    def __init__(self, class_counts, beta=0.9999, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.register_buffer("weights", effective_num_weights(class_counts, beta))
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        w = self.weights.to(logits.device)
        ce = F.cross_entropy(logits, targets, weight=w,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class ClassBalancedFocalLossSoft(nn.Module):
    """Tương thích SOFT-LABEL từ CutMix/Mixup (tổ hợp lồi 2 one-hot).
    Khi soft_targets suy biến về one-hot thuần -> thu gọn ĐÚNG về bản gốc ở trên."""
    def __init__(self, class_counts, beta=0.9999, gamma=2.0):
        super().__init__()
        self.register_buffer("weights", effective_num_weights(class_counts, beta))
        self.gamma = gamma

    def forward(self, logits, soft_targets):
        w = self.weights.to(logits.device)
        log_probs = torch.log_softmax(logits, dim=1)
        raw_ce = -(soft_targets * log_probs).sum(dim=1)         # soft-CE, chưa nhân weight
        sample_weight = (soft_targets * w).sum(dim=1)            # tổ hợp lồi trọng số lớp
        pt = torch.exp(-raw_ce)
        return ((1 - pt) ** self.gamma * raw_ce * sample_weight).mean()
```

```python
# Final_Pipeline/src/train/samplers.py
"""WeightedRandomSampler dùng effective_num_weights (nhất quán công thức toàn dự án)."""
from torch.utils.data import WeightedRandomSampler
from src.train.losses import effective_num_weights


def build_balanced_sampler(samples, class_counts, beta=0.9999):
    """samples: list (path, label_idx). Dùng cho Bước 3 decoupled re-training (Mục 4.7)."""
    class_w = effective_num_weights(class_counts, beta=beta)
    sample_weights = [class_w[label].item() for _, label in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
```

### 4.5. VRAM management (AMP + gradient accumulation + probe)

Trên 4GB VRAM, ba kỹ thuật sau là **bắt buộc**, không phải tuỳ chọn:

- **(a) Mixed Precision** (Micikevicius et al., 2017): FP16 giảm ~½ bộ nhớ, không giảm accuracy.
- **(b) Gradient Accumulation**: tích luỹ gradient qua nhiều batch nhỏ → mô phỏng effective batch 32–64 mà không load hết vào VRAM.
- **(c) Gradient Checkpointing** (Chen et al., 2016): O(√n) activation memory, chỉ thêm 1 forward pass. Dùng khi thử MobileViT/Swin; không cần cho EffV2-S ở batch nhỏ.

OOM fallback ladder: (1) clear cache + log config lỗi → (2) `batch_size 16→8`, `accumulation 2→4` → (3) freeze block sớm hơn → (4) tắt augment nặng chạy nhầm trên GPU → (5) chuyển B0 fallback cho ablation bị ảnh hưởng → (6) cắt ablation model lớn trước khi động vào path EffV2-S production.

```python
# Final_Pipeline/src/train/memory_utils.py
"""AMP + gradient accumulation + probe VRAM.
Mixed precision: Micikevicius et al., 2017 — https://arxiv.org/abs/1710.03740
Grad checkpointing: Chen et al., 2016 — https://arxiv.org/abs/1604.06174"""
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
```

Cấu hình DataLoader hợp RAM 16GB (chỉ còn 8–10GB trống):

```python
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=16, sampler=balanced_sampler,
    num_workers=2,             # KHÔNG dùng 8 — RAM hạn chế
    pin_memory=True,           # chi phí RAM tỉ lệ batch, không phải toàn dataset
    persistent_workers=False,  # tránh giữ worker (và RAM) sống giữa epoch
    prefetch_factor=2,
)
```

Mỗi training script phải log: GPU name + driver/CUDA/PyTorch version; peak allocated & reserved VRAM; batch/accumulation/AMP status; số trainable params; time/epoch + checkpoint size.

### 4.6. Domain Adaptation (lab → field) + Datasets

Domain gap lab→field là **trung tâm bài toán**. Bản Final dùng ba lớp kỹ thuật xếp chồng, khuyến nghị bắt đầu Cách 1 (rẻ, bằng chứng trực tiếp trên 4/5 loài):

**Cách 1 — Field-Background Recomposition** (Jeon et al., 2025): cắt lá khỏi nền lab đơn sắc, dán lên nền field ngẫu nhiên (không cần nhãn nền). Dùng như augmentation khi pretrain. Bằng chứng gốc: cải thiện 9.9–23.6 điểm % trên tomato/pepper/grape/apple.

**Cách 2 — MMD feature alignment** (Wu et al., 2023): thêm loss phụ kéo gần phân phối embedding lab↔field, song song CE chính. `lambda_mmd` warm-up 0→0.1 để domain loss không lấn task loss.

**Cách 3 — Few-shot Metric Domain Adaptation** (Kudo et al., 2024): khi có ít nhãn field mỗi lớp, thêm ràng buộc khoảng cách feature source↔target. Tuỳ chọn nâng cao.

**QC bắt buộc cho ảnh synthetic:** mỗi experiment bật domain-aug phải lưu ≥64 mẫu vào `reports/domain_aug_samples/<experiment_id>/`, chạy ablation có/không domain-aug trên cùng validation protocol, và dùng **Grad-CAM** kiểm tra model tập trung vào vùng lá/bệnh thay vì nền. Nếu Grad-CAM bám nền → tắt augment đó, ghi vào manifest.

```python
# Final_Pipeline/src/data/domain_augmentation.py
"""Field-background recomposition (ý tưởng Jeon et al., 2025 — https://dx.doi.org/10.2139/ssrn.5217081)."""
import cv2
import numpy as np
import random


def extract_leaf_mask(lab_image_bgr):
    """Tách lá khỏi nền đơn sắc bằng Otsu (ảnh PRETRAIN nền sạch)."""
    gray = cv2.cvtColor(lab_image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def recompose_on_field_background(lab_image_bgr, field_bg_bgr):
    mask = extract_leaf_mask(lab_image_bgr)
    field_bg = cv2.resize(field_bg_bgr, (lab_image_bgr.shape[1], lab_image_bgr.shape[0]))
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    return (lab_image_bgr * m3 + field_bg * (1 - m3)).astype(np.uint8)


class FieldBackgroundAugment:
    """Dùng trong Dataset.__getitem__ khi train PRETRAIN, p ~ 0.3-0.5."""
    def __init__(self, field_background_paths, p=0.4):
        self.bg_paths, self.p = field_background_paths, p

    def __call__(self, lab_image_bgr):
        if random.random() > self.p or not self.bg_paths:
            return lab_image_bgr
        bg = cv2.imread(random.choice(self.bg_paths))
        return recompose_on_field_background(lab_image_bgr, bg) if bg is not None else lab_image_bgr
```

```python
# Final_Pipeline/src/train/domain_loss.py
"""MMD loss (cảm hứng Wu et al., 2023 — https://doi.org/10.34133/plantphenomics.0038)."""
import torch


def mmd_loss(source_features, target_features, sigma=1.0):
    """Maximum Mean Discrepancy (RBF) — kéo gần phân phối đặc trưng 2 miền."""
    def rbf(x, y):
        return torch.exp(-(torch.cdist(x, y) ** 2) / (2 * sigma ** 2))
    return rbf(source_features, source_features).mean() \
        + rbf(target_features, target_features).mean() \
        - 2 * rbf(source_features, target_features).mean()
# Trong train loop: loss = ce + lambda_mmd * mmd_loss(feat_src, feat_tgt)
```

Dataset tuỳ biến (thay `ImageFolder` để chèn được domain-aug bằng OpenCV trước albumentations, và đọc từ manifest):

```python
# Final_Pipeline/src/data/datasets.py
"""PretrainDataset (đọc theo folder, chèn field-background) +
   FinetuneDataset (đọc từ manifest split_v1.json, Mục 2.5)."""
import os
import cv2
from torch.utils.data import Dataset

IMG_EXT = (".jpg", ".jpeg", ".png")


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
        img = cv2.imread(path)
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
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label
```

### 4.7. PEFT (fine-tuning hiệu quả tham số)

Full-unfreeze buộc AdamW lưu 2 moment cho **mọi** param backbone — lãng phí VRAM. Bản Final dùng PEFT-lite theo mức độ, không cần thư viện phức tạp:

| mode | Train gì | Khi nào dùng |
|---|---|---|
| `head_only` | Chỉ classifier head | Warm-up (Bước 1); lớp cực hiếm |
| `head_bn_only` | + BatchNorm affine (tinh thần BitFit) | Rẻ, cải thiện đáng kể so với chỉ head |
| `head_bn_lastblock` | + 1–2 block cuối backbone | **Mặc định Bước 2** — nhiều năng lực học, vẫn tiết kiệm |
| `full` | Toàn bộ | Chỉ khi probe VRAM còn dư (Apple/Corn/Tomato) |

```python
# Final_Pipeline/src/train/peft.py
"""PEFT-lite: mở khoá dần head -> BN -> block cuối. Rẻ VRAM hơn full-unfreeze.
LoRA (Hu et al., 2021 — https://arxiv.org/abs/2106.09685) là mức nâng cao cho backbone ViT-based."""
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
```

### 4.8. Script Pretrain 38 lớp (`train_pretrain.py`)

Ghép supervised transfer (EffV2-S từ ImageNet) + field-background recomposition (Mục 4.6) + AMP/grad-accumulation (Mục 4.5) + CutMix/Mixup (Mục 4.3, **hợp lệ** ở pretrain vì 38 lớp chung 1 head). Chọn checkpoint theo **macro-F1** trên val (không dùng accuracy — data imbalance).

```python
# Final_Pipeline/src/train/train_pretrain.py
"""Pretrain backbone 38 lớp (Phase 1). Chọn checkpoint theo macro-F1.
Chạy: python -m src.train.train_pretrain --config configs/pretrain_effv2s.yaml"""
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.models.backbone import build_backbone
from src.data.datasets import PretrainDataset
from src.data.augmentation import get_pretrain_transforms, CutMixMixupCollate
from src.data.domain_augmentation import FieldBackgroundAugment
from src.train.memory_utils import train_one_epoch_memory_efficient, probe_vram_usage
from src.eval.evaluate import evaluate


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def soft_target_cross_entropy(logits, soft_targets, label_smoothing=0.0):
    """CE cho nhãn phân phối (khi CutMix/Mixup bật). Tự viết để label_smoothing tường minh."""
    n = logits.size(1)
    logp = torch.log_softmax(logits, dim=1)
    if label_smoothing > 0:
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / n
    return -(soft_targets * logp).sum(dim=1).mean()


def list_images_recursive(root):
    out = []
    for dp, _, fs in os.walk(root):
        out += [os.path.join(dp, f) for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return out


def build_dataloaders(cfg):
    d = cfg["data"]
    train_tf = get_pretrain_transforms(True, d["img_size"])
    val_tf = get_pretrain_transforms(False, d["img_size"])
    dom = cfg.raw.get("domain_aug", {})
    field_aug = None
    if dom.get("enabled", False):
        bg = list_images_recursive(dom["field_background_dir"])
        field_aug = FieldBackgroundAugment(bg, p=dom.get("p", 0.4))
        print(f"[domain_aug] {len(bg)} nền field, p={dom.get('p', 0.4)}")
    train_ds = PretrainDataset(d["train_dir"], train_tf, domain_aug=field_aug)
    val_ds = PretrainDataset(d["val_dir"], val_tf, domain_aug=None)
    cm = cfg.raw.get("cutmix_mixup", {})
    collate = CutMixMixupCollate(cfg["model"]["num_classes"], cm.get("alpha", 0.2),
                                 cm.get("cutmix_prob", 0.5), cm.get("p", 0.5)) if cm.get("enabled") else None
    tl = DataLoader(train_ds, cfg["train"]["batch_size"], shuffle=True, num_workers=d["num_workers"],
                    pin_memory=True, persistent_workers=False, collate_fn=collate)
    vl = DataLoader(val_ds, cfg["train"]["batch_size"], shuffle=False,
                    num_workers=d["num_workers"], pin_memory=True)
    return tl, vl, collate is not None


def train_pretrain(config_path):
    cfg = TrainConfig.load(config_path)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tl, vl, soft = build_dataloaders(cfg)
    model = build_backbone(cfg["model"]["num_classes"], cfg["model"]["arch"],
                           drop_path_rate=cfg["model"]["drop_path_rate"]).to(device)

    # BƯỚC BẮT BUỘC — probe VRAM (Mục 4.5)
    sample, _ = next(iter(tl))
    bs = cfg["train"]["batch_size"]
    rep = probe_vram_usage(model, sample, device, batch_sizes=(bs,))
    print(f"[probe_vram] bs={bs}: {rep[bs]}")
    if rep[bs] == "OOM":
        raise RuntimeError(f"OOM ở bs={bs}. Giảm batch_size hoặc tăng accumulation_steps trong {config_path}.")

    t = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr_head"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    if soft:
        crit = lambda lo, ta: soft_target_cross_entropy(lo, ta, t["label_smoothing"])
    else:
        crit = torch.nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])

    out = cfg["checkpoint"]["out_path"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    best = 0.0
    for ep in range(t["epochs"]):
        loss = train_one_epoch_memory_efficient(model, tl, opt, crit, device,
                t["accumulation_steps"], t["use_amp"], t["use_grad_checkpoint"])
        sched.step()
        f1 = evaluate(model, vl, device)
        print(f"[pretrain] ep {ep+1}/{t['epochs']} loss={loss:.4f} val_macro_f1={f1:.4f}")
        if f1 > best:
            best = f1
            torch.save({"model_state": model.state_dict(), "arch": cfg["model"]["arch"],
                        "num_classes": cfg["model"]["num_classes"], "val_macro_f1": best}, out)
            print(f"  -> saved {out} (macro-F1={best:.4f})")
    print(f"[pretrain] done — best val macro-F1: {best:.4f}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_effv2s.yaml")
    train_pretrain(ap.parse_args().config)
```

### 4.9. Script Fine-tune per-species (`train_finetune.py`)

Script trung tâm Phase 2, phân nhánh theo `rare_class_threshold=30`:

```text
Ảnh loài X (đọc từ manifest split_v1.json)
  ├── Lớp THƯỜNG (Chế độ 1) — TẤT CẢ lớp hiện nay
  │     Bước 1: PEFT head_only (warm-up)
  │     Bước 2: PEFT head_bn_lastblock + Class-Balanced Focal + CutMix/Mixup (+MMD tuỳ chọn)
  │              + early stopping (early_stop_patience)
  │     Bước 3: decoupled re-training head + WeightedRandomSampler + CE THƯỜNG (backbone đóng băng)
  │     -> checkpoints/model_<species>.pt
  └── Lớp HIẾM (Chế độ 2) — HIỆN KHÔNG CÓ (rare_folds rỗng -> bỏ qua)
        (nếu tương lai có lớp < ngưỡng: trích embedding -> prototype -> K-fold, Mục 3.5)
```

Vì sao tách 3 bước: Bước 1–2 học representation; Bước 3 áp dụng Kang et al. (2019) — rebalancing bằng sampler nên **tách pha** khỏi học representation. Điểm quan trọng trong code: **Bước 3 dùng `CrossEntropyLoss` thường (KHÔNG class weight)** đi cùng `WeightedRandomSampler` — nếu dùng cả hai thì "cân bằng gấp đôi", head lệch mạnh về lớp thiểu số và macro-F1 tụt. Bước 3 cũng tắt CutMix/Mixup. Với dữ liệu hiện tại **mọi lớp đều qua đủ 3 bước** (không lớp nào rẽ sang Chế độ 2).

**Phần 1/2 — helpers** (load backbone, optimizer discriminative-LR, một pha train, MMD-aware loop):

```python
# Final_Pipeline/src/train/train_finetune.py  (phần 1/2)
"""Fine-tune 1 loài — Chế độ 1 (Bước 1->2->3) + Chế độ 2 (few-shot).
Chạy: python -m src.train.train_finetune --config configs/finetune_tomato.yaml"""
import argparse
import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.models.backbone import build_backbone
from src.data.datasets import FinetuneDataset, PretrainDataset
from src.data.augmentation import get_finetune_transforms, get_pretrain_transforms, CutMixMixupCollate
from src.train.losses import effective_num_weights, ClassBalancedFocalLossSoft
from src.train.samplers import build_balanced_sampler
from src.train.peft import set_trainable_params
from src.train.few_shot import compute_prototypes, classify_by_prototype
from src.train.domain_loss import mmd_loss
from src.train.memory_utils import probe_vram_usage
from src.train.train_pretrain import set_seed
from src.eval.evaluate import evaluate


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pretrained_backbone(model, ckpt_path, device):
    """Nạp backbone pretrain 38 lớp, BỎ head (số lớp khác). Lọc theo tên+shape."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state, msd = ckpt["model_state"], model.state_dict()
    filtered = {k: v for k, v in state.items() if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(filtered, strict=False)
    print(f"  [load_pretrained] nạp {len(filtered)}/{len(state)} tensor (head khởi tạo mới)")
    return model


def build_optimizer(model, lr_head, lr_backbone, weight_decay):
    """Discriminative LR: head nhanh hơn backbone vừa mở khoá."""
    head = [p for n, p in model.named_parameters()
            if p.requires_grad and ("classifier" in n or "fc" in n)]
    back = [p for n, p in model.named_parameters()
            if p.requires_grad and not ("classifier" in n or "fc" in n)]
    groups = [{"params": head, "lr": lr_head}]
    if back:
        groups.append({"params": back, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_regular_dataloaders(cfg, manifest):
    d = cfg["data"]
    train_ds = FinetuneDataset(manifest, "train", get_finetune_transforms(True, d["img_size"]))
    val_ds = FinetuneDataset(manifest, "val", get_finetune_transforms(False, d["img_size"]))
    idx_to_class = {v: k for k, v in manifest["class_to_idx"].items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    counts = [sum(1 for _, y in train_ds.samples if y == i) for i in range(len(classes))]
    cm = cfg.raw.get("train", {}).get("cutmix_mixup", {})
    collate = CutMixMixupCollate(len(classes), cm.get("alpha", 0.2),
                                 cm.get("cutmix_prob", 0.5), cm.get("p", 0.5)) if cm.get("enabled") else None
    tl = DataLoader(train_ds, cfg["train"]["batch_size"], shuffle=True, num_workers=d["num_workers"],
                    pin_memory=True, persistent_workers=False, collate_fn=collate)
    vl = DataLoader(val_ds, cfg["train"]["batch_size"], shuffle=False,
                    num_workers=d["num_workers"], pin_memory=True)
    return tl, vl, train_ds, classes, counts


def build_mmd_source_loader(source_dir, img_size, batch_size, num_workers):
    ds = PretrainDataset(source_dir, get_pretrain_transforms(False, img_size), domain_aug=None)
    return DataLoader(ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


def train_one_epoch_finetune(model, loader, opt, crit, device, accumulation_steps=4,
                             use_amp=True, mmd_source_loader=None, lambda_mmd=0.0):
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train(); opt.zero_grad(set_to_none=True)
    run_loss = 0.0
    mmd_iter = iter(mmd_source_loader) if mmd_source_loader is not None else None
    for step, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            logits = model(imgs)
            loss = crit(logits, targets)
            if mmd_iter is not None and lambda_mmd > 0:
                try:
                    src, _ = next(mmd_iter)
                except StopIteration:
                    mmd_iter = iter(mmd_source_loader); src, _ = next(mmd_iter)
                src = src.to(device, non_blocking=True)
                fs = model.forward_head(model.forward_features(src), pre_logits=True)
                ft = model.forward_head(model.forward_features(imgs), pre_logits=True)
                loss = loss + lambda_mmd * mmd_loss(fs, ft)
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        run_loss += loss.item() * accumulation_steps
    return run_loss / len(loader)


def _train_phase(model, tl, vl, opt, crit, device, *, epochs, accumulation_steps, use_amp,
                 best_f1, ckpt_meta, ckpt_path, phase, mmd_loader=None, lambda_max=0.0, mmd_warmup=0,
                 patience=0):
    """Chạy 1 pha, lưu checkpoint tốt nhất XUYÊN SUỐT 3 bước (best_f1 chuyền tiếp).
    patience>0: dừng sớm nếu val F1 không cải thiện sau `patience` epoch liên tiếp (xét trong pha)."""
    best_in_phase = -1.0
    stall = 0
    for ep in range(epochs):
        lam = lambda_max * min(1.0, (ep + 1) / max(1, mmd_warmup)) if (mmd_loader and lambda_max > 0) else 0.0
        loss = train_one_epoch_finetune(model, tl, opt, crit, device, accumulation_steps,
                                        use_amp, mmd_loader, lam)
        f1 = evaluate(model, vl, device)
        msg = f"[{phase}] ep {ep+1}/{epochs} loss={loss:.4f} val_macro_f1={f1:.4f}"
        if lam > 0:
            msg += f" lambda_mmd={lam:.3f}"
        print(msg)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({**ckpt_meta, "model_state": model.state_dict(), "val_macro_f1": best_f1}, ckpt_path)
            print(f"    -> saved ({phase}) {ckpt_path} (macro-F1={best_f1:.4f})")
        # early stopping xét theo cải thiện TRONG pha (epsilon bỏ qua nhiễu nhỏ)
        if f1 > best_in_phase + 1e-4:
            best_in_phase = f1; stall = 0
        else:
            stall += 1
            if patience and stall >= patience:
                print(f"    -> early stop ({phase}): {patience} epoch không cải thiện")
                break
    return best_f1
```

Phần 2/2 (Chế độ 2 few-shot + hàm điều phối `train_finetune`) ở mục tiếp theo.

**Phần 2/2 — Chế độ 2 few-shot + điều phối:**

```python
# Final_Pipeline/src/train/train_finetune.py  (phần 2/2 — nối tiếp phần 1/2)

@torch.no_grad()
def extract_embeddings(model, items, transform, device, batch_size=16):
    """Trích embedding pre-logits theo batch (tránh tràn VRAM).
    items: list (path, label_idx). forward_head(pre_logits=True) -> vector ĐÃ pooled."""
    model.eval()
    feats, labels = [], []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        imgs = []
        for path, _ in chunk:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            imgs.append(transform(image=img)["image"])
        batch = torch.stack(imgs).to(device)
        emb = model.forward_head(model.forward_features(batch), pre_logits=True)
        feats.append(emb.float().cpu())
        labels.extend(y for _, y in chunk)
    return torch.cat(feats), torch.tensor(labels)


def run_few_shot_evaluation(cfg, model, rare_folds, device, ckpt_path):
    """Chế độ 2: dùng backbone TỐT NHẤT (Chế độ 1) để trích embedding, đánh giá K-fold
    (mỗi ảnh làm query đúng 1 lần). Cuối cùng lưu prototype trên TOÀN BỘ ảnh hiếm để suy luận."""
    species = cfg["species"]
    print(f"\n===== [{species}] CHẾ ĐỘ 2: few-shot prototype (Snell et al., 2017) =====")
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"], strict=False); model.eval()

    rare_classes = sorted(rare_folds.keys())
    cls_to_idx = {c: i for i, c in enumerate(rare_classes)}
    num_rare = len(rare_classes)
    val_tf = get_finetune_transforms(False, cfg["data"]["img_size"])

    items = [(fold, path, cls_to_idx[cls])
             for cls, entries in rare_folds.items() for fold, path in entries]
    if not items:
        print(f"[{species}] rare_folds rỗng — bỏ qua."); return
    n_folds = max(f for f, _, _ in items) + 1

    fold_acc, fold_f1 = [], []
    for f in range(n_folds):
        support = [(p, y) for fi, p, y in items if fi != f]
        query = [(p, y) for fi, p, y in items if fi == f]
        if not support or not query:
            continue
        sf, sl = extract_embeddings(model, support, val_tf, device)
        qf, ql = extract_embeddings(model, query, val_tf, device)
        protos = compute_prototypes(sf, sl, num_rare)
        preds = classify_by_prototype(qf, protos.to(qf.device)).argmax(dim=1)
        fold_acc.append((preds == ql).float().mean().item())
        fold_f1.append(f1_score(ql.numpy(), preds.numpy(), average="macro", zero_division=0))
        print(f"  fold {f}: n_query={len(query)} acc={fold_acc[-1]:.4f} macro_f1={fold_f1[-1]:.4f}")
    if fold_acc:
        print(f"[{species}] Chế độ 2 ({len(fold_acc)} fold): "
              f"acc={np.mean(fold_acc):.4f}±{np.std(fold_acc):.4f} "
              f"macro_f1={np.mean(fold_f1):.4f}±{np.std(fold_f1):.4f}")

    # Prototype CUỐI: dùng toàn bộ ảnh hiếm (để suy luận thật)
    all_items = [(p, y) for _, p, y in items]
    af, al = extract_embeddings(model, all_items, val_tf, device)
    final_protos = compute_prototypes(af, al, num_rare)
    proto_path = ckpt_path.replace(".pt", "_prototypes.pt")
    torch.save({"prototypes": final_protos, "rare_classes": rare_classes,
                "class_to_idx": cls_to_idx, "species": species,
                "embed_dim": final_protos.size(1)}, proto_path)
    print(f"[{species}] Lưu {num_rare} prototype -> {proto_path}")


def train_finetune(config_path):
    cfg = TrainConfig.load(config_path)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    species, t = cfg["species"], cfg["train"]
    manifest = load_manifest(cfg["data"]["split_manifest"])

    # ===== CHẾ ĐỘ 1 =====
    tl, vl, train_ds, classes, counts = build_regular_dataloaders(cfg, manifest)
    print(f"[{species}] Chế độ 1 — {len(classes)} lớp thường: {dict(zip(classes, counts))}")
    model = build_backbone(len(classes), cfg["model"]["arch"], drop_path_rate=0.1).to(device)
    model = load_pretrained_backbone(model, cfg["model"]["pretrained_ckpt"], device)

    sample, _ = next(iter(tl)); bs = t["batch_size"]
    rep = probe_vram_usage(model, sample, device, batch_sizes=(bs,))
    print(f"[probe_vram] bs={bs}: {rep[bs]}")
    if rep[bs] == "OOM":
        raise RuntimeError(f"OOM ở bs={bs}. Giảm batch_size / tăng accumulation_steps trong {config_path}.")

    cb = t["class_balanced_loss"]
    crit_cb = ClassBalancedFocalLossSoft(counts, beta=cb["beta"], gamma=cb["gamma"]).to(device)
    ckpt_path = cfg["checkpoint"]["out_path"]
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    meta = {"arch": cfg["model"]["arch"], "num_classes": len(classes), "classes": classes, "species": species}
    best = 0.0

    # Bước 1 — warm-up head
    print(f"\n===== [{species}] BƯỚC 1: head_only =====")
    set_trainable_params(model, "head_only")
    opt = build_optimizer(model, t["lr_head"], t["lr_backbone"], t["weight_decay"])
    best = _train_phase(model, tl, vl, opt, crit_cb, device, epochs=t["epochs_warmup"],
                        accumulation_steps=t["accumulation_steps"], use_amp=t["use_amp"],
                        best_f1=best, ckpt_meta=meta, ckpt_path=ckpt_path, phase="warmup")

    # Bước 2 — head_bn_lastblock + CB-Focal + CutMix + MMD tuỳ chọn
    print(f"\n===== [{species}] BƯỚC 2: head_bn_lastblock =====")
    set_trainable_params(model, "head_bn_lastblock")
    opt = build_optimizer(model, t["lr_head"], t["lr_backbone"], t["weight_decay"])
    mmd_loader, lam_max, warm = None, 0.0, 0
    mcfg = t.get("domain_mmd", {})
    if mcfg.get("enabled"):
        mmd_loader = build_mmd_source_loader(mcfg["source_dir"], cfg["data"]["img_size"],
                                             t["batch_size"], cfg["data"]["num_workers"])
        lam_max, warm = mcfg["lambda_max"], mcfg["warmup_epochs"]
        print(f"  [MMD] bật: lambda_max={lam_max}, warmup={warm}")
    best = _train_phase(model, tl, vl, opt, crit_cb, device, epochs=t["epochs_full"],
                        accumulation_steps=t["accumulation_steps"], use_amp=t["use_amp"],
                        best_f1=best, ckpt_meta=meta, ckpt_path=ckpt_path, phase="full",
                        mmd_loader=mmd_loader, lambda_max=lam_max, mmd_warmup=warm,
                        patience=t.get("early_stop_patience", 0))

    # Bước 3 — decoupled re-training (backbone đóng băng, sampler cân bằng, KHÔNG CutMix)
    print(f"\n===== [{species}] BƯỚC 3: decoupled re-training =====")
    set_trainable_params(model, "head_only")
    opt = build_optimizer(model, t["lr_head"], t["lr_backbone"], t["weight_decay"])
    bal_ds = FinetuneDataset(manifest, "train", get_finetune_transforms(True, cfg["data"]["img_size"]))
    sampler = build_balanced_sampler(bal_ds.samples, counts, beta=cb["beta"])
    bal_loader = DataLoader(bal_ds, t["batch_size"], sampler=sampler,
                            num_workers=cfg["data"]["num_workers"], pin_memory=True)
    # Decoupled (Kang et al., 2019): balanced sampler ĐI VỚI CE THƯỜNG (KHÔNG class weight).
    # Dùng cả hai = cân bằng gấp đôi -> head lệch mạnh về lớp thiểu số, F1 tụt.
    crit_ce = nn.CrossEntropyLoss()
    best = _train_phase(model, bal_loader, vl, opt, crit_ce, device, epochs=t["epochs_decoupled"],
                        accumulation_steps=t["accumulation_steps"], use_amp=t["use_amp"],
                        best_f1=best, ckpt_meta=meta, ckpt_path=ckpt_path, phase="decoupled")
    print(f"\n[{species}] Chế độ 1 DONE — best macro-F1: {best:.4f}")

    # ===== CHẾ ĐỘ 2 (hiện KHÔNG chạy — rare_folds rỗng) =====
    rare_folds = manifest.get("rare_folds", {})
    if rare_folds:
        run_few_shot_evaluation(cfg, model, rare_folds, device, ckpt_path)
    else:
        print(f"[{species}] không có lớp hiếm — bỏ qua Chế độ 2.")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    train_finetune(ap.parse_args().config)
```

**Ba điểm thiết kế còn hiệu lực:**
1. `load_pretrained_backbone` lọc theo **shape** → tự loại head lệch số lớp, không hard-code tên.
2. `best_f1` chuyền qua cả 3 bước → luôn giữ một `model_<species>.pt` là trạng thái macro-F1 cao nhất toàn cục; early stopping xét cải thiện trong từng pha.
3. Bước 3 dùng `WeightedRandomSampler` + `CrossEntropyLoss` **thường** (không class weight) để tránh cân bằng gấp đôi (Kang et al., 2019).

> Điểm về Chế độ 2 (nạp checkpoint tốt nhất trước khi trích embedding, K-fold tách khỏi prototype suy luận) chỉ áp dụng khi `rare_folds` không rỗng — hiện không kích hoạt (Mục 3.5).

### 4.10. Checkpoint & Resume

Checkpoint phải đủ để resume chính xác và giải thích được metric sau này.

| Trường | Nội dung |
|---|---|
| `model_state_dict` | Backbone + classifier |
| `optimizer_state_dict` | AdamW state |
| `scheduler_state_dict` | LR scheduler |
| `scaler_state_dict` | AMP GradScaler |
| `epoch`, `best_metric` | Epoch cuối + best val macro-F1 |
| `class_to_idx`, `idx_to_class` | Mapping bất biến |
| `transform_config` | Resize/crop/normalization/color |
| `split_manifest_sha256` | Hash manifest split |
| `config_hash` | Hash config train |
| `rng_state` | Python/NumPy/PyTorch RNG |

Quy tắc resume: chỉ resume khi `class_to_idx` + manifest hash + config hash khớp; đổi mapping/augment → experiment ID mới; lưu `last.pt` mỗi epoch + `best.pt` khi macro-F1 cải thiện; **export chỉ từ `best.pt`**.

```python
# Final_Pipeline/src/train/checkpoints.py
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
```

### 4.11. Experiment Matrix

Bộ tối thiểu phải hoàn thành trước khi báo cáo cuối (E0–E6 bắt buộc, E7–E9 tuỳ chọn):

| ID | Stage | Model | Loài | Kỹ thuật | Mục đích | Bắt buộc? |
|---|---|---|---|---|---|---|
| E0 | Pretrain | EffV2-S | 38 lớp | CE + field-bg recompose | Backbone sanity | Yes |
| E1 | Finetune | EffV2-S | Apple | CE baseline | Minimal baseline | Yes |
| E2 | Finetune | EffV2-S | Apple | domain aug + CB-Focal | Strong Apple candidate | Yes |
| E3 | Finetune | EffV2-S | Corn | CB-Focal + sampler | Imbalance handling | Yes |
| E4 | Finetune | EffV2-S | Grape | backbone LR thấp + augment bảo thủ | Small-data | Yes |
| E5 | Finetune | EffV2-S | Pepper | backbone LR thấp + augment bảo thủ | Small-data | Yes |
| E6 | Finetune | EffV2-S | Tomato | CB-Focal + decoupled | 9 lớp, imbalance | Yes |
| E7 | Ablation | EffV2-B0 | All/subset | fallback | So tốc độ/VRAM | Optional |
| E8 | Ablation | Swin-T | Apple/Tomato | transformer | So kiến trúc | Optional |
| E9 | Ablation | DINOv2-S frozen | Tomato | few-shot embedding | So foundation model | Optional |

Ưu tiên: hoàn thành E0–E6 trước ablation. Nếu thiếu thời gian, cắt ablation trước; **không** cắt calibration, ONNX parity, security test, holdout report.

Quy tắc so sánh ablation: cùng split + cùng seed; không so nếu mapping/split/img_size/preprocessing đổi mà không ghi chú; rank theo macro-F1 → calibration → robustness → latency; model tăng accuracy nhưng hại rare-class recall / ECE nặng thì **không** tự động tốt hơn.

### 4.12. Script train Leaf Gate (`train_leaf_gate.py`)

Leaf Gate (Mục 3.2) là chốt chặn nhị phân leaf / not-leaf, train **độc lập** với 5 model loài và chạy một lần. Backbone MobileNetV3-Small (nhẹ, ~2.5M). Dữ liệu gồm positive (lá từ PRETRAIN + FINETUNE train split — **không** đụng holdout) và negative (ảnh không phải lá: tay, đất, trời, vật thể; ghi `[Cần verify license]` cho nguồn).

Cấu trúc thư mục dữ liệu leaf-gate (chuẩn bị thủ công một lần):

```text
D:\DAT301m_Project\Final_Pipeline\data\leaf_gate\
├── train\{leaf, not_leaf}\*.jpg
├── calib\{leaf, not_leaf}\*.jpg      # cho temperature scaling (Mục 5.3)
└── val\{leaf, not_leaf}\*.jpg
```

```python
# Final_Pipeline/src/train/train_leaf_gate.py
"""Train Leaf Gate nhị phân (MobileNetV3-Small, Mục 3.2). Chạy 1 lần, độc lập 5 loài.
Mục tiêu: leaf recall >= 98% trên lá field hợp lệ, non-leaf rejection >= 95%.
Chạy: python -m src.train.train_leaf_gate --config configs/leaf_gate.yaml"""
import argparse
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score

from src.config import TrainConfig
from src.models.backbone import build_leaf_gate
from src.data.augmentation import get_finetune_transforms
from src.train.memory_utils import train_one_epoch_memory_efficient, probe_vram_usage
from src.train.train_pretrain import set_seed

IMG_EXT = (".jpg", ".jpeg", ".png")
CLASS_TO_IDX = {"not_leaf": 0, "leaf": 1}   # index 1 = leaf (khớp app.py: softmax[...,1])


class LeafGateDataset(Dataset):
    """Đọc {split}/{leaf,not_leaf}/*. Nhãn nhị phân: leaf=1, not_leaf=0."""
    def __init__(self, root_split_dir, transform=None):
        self.transform = transform
        self.samples = []
        for cls, idx in CLASS_TO_IDX.items():
            cdir = os.path.join(root_split_dir, cls)
            if not os.path.isdir(cdir):
                continue
            for f in os.listdir(cdir):
                if f.lower().endswith(IMG_EXT):
                    self.samples.append((os.path.join(cdir, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label


@torch.no_grad()
def eval_leaf_gate(model, loader, device):
    """Trả (macro_f1, leaf_recall, notleaf_recall) — theo dõi đúng 2 chỉ tiêu Mục 3.2."""
    model.eval()
    yt, yp = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        yp.extend(logits.argmax(1).cpu().tolist())
        yt.extend(labels.tolist())
    macro = f1_score(yt, yp, average="macro", zero_division=0)
    leaf_rec = recall_score(yt, yp, pos_label=1, zero_division=0)
    notleaf_rec = recall_score(yt, yp, pos_label=0, zero_division=0)
    return macro, leaf_rec, notleaf_rec


def train_leaf_gate(config_path):
    cfg = TrainConfig.load(config_path)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, t = cfg["data"], cfg["train"]
    train_tf = get_finetune_transforms(True, d["img_size"])
    val_tf = get_finetune_transforms(False, d["img_size"])
    tl = DataLoader(LeafGateDataset(os.path.join(d["root"], "train"), train_tf),
                    t["batch_size"], shuffle=True, num_workers=d["num_workers"], pin_memory=True)
    vl = DataLoader(LeafGateDataset(os.path.join(d["root"], "val"), val_tf),
                    t["batch_size"], shuffle=False, num_workers=d["num_workers"], pin_memory=True)

    model = build_leaf_gate(pretrained=True).to(device)
    sample, _ = next(iter(tl))
    if probe_vram_usage(model, sample, device, batch_sizes=(t["batch_size"],))[t["batch_size"]] == "OOM":
        raise RuntimeError("OOM leaf-gate — giảm batch_size.")

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    # Cân bằng leaf/not-leaf bằng class weight (thường không quá lệch)
    crit = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.05))
    out = cfg["checkpoint"]["out_path"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    best = 0.0
    for ep in range(t["epochs"]):
        loss = train_one_epoch_memory_efficient(model, tl, opt, crit, device,
                t["accumulation_steps"], t["use_amp"])
        macro, leaf_rec, nl_rec = eval_leaf_gate(model, vl, device)
        print(f"[leaf_gate] ep {ep+1}/{t['epochs']} loss={loss:.4f} "
              f"macro_f1={macro:.4f} leaf_recall={leaf_rec:.4f} notleaf_recall={nl_rec:.4f}")
        if macro > best:
            best = macro
            torch.save({"model_state": model.state_dict(), "arch": "mobilenetv3_small_100",
                        "class_to_idx": CLASS_TO_IDX, "num_classes": 2, "val_macro_f1": best}, out)
            print(f"  -> saved {out} (macro-F1={best:.4f})")
    print(f"[leaf_gate] done — best macro-F1: {best:.4f}. "
          f"Tiếp theo: calibrate (Mục 5.3) trên calib split, chọn ngưỡng từ PR curve.")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/leaf_gate.yaml")
    train_leaf_gate(ap.parse_args().config)
```

Config leaf-gate:

```yaml
# Final_Pipeline/configs/leaf_gate.yaml
seed: 42
data:
  root: "D:/DAT301m_Project/Final_Pipeline/data/leaf_gate"   # train/calib/val với leaf & not_leaf
  img_size: 224
  num_workers: 2
train:
  batch_size: 32          # model nhẹ, cho phép batch lớn hơn
  accumulation_steps: 1
  epochs: 15
  lr: 1.0e-3
  weight_decay: 1.0e-4
  label_smoothing: 0.05
  use_amp: true
checkpoint:
  out_path: "D:/DAT301m_Project/Final_Pipeline/checkpoints/leaf_gate.pt"
  metric: "macro_f1"
```

> Sau train, leaf-gate được calibration (temperature scaling, Mục 5.3) trên `calib` split và chọn `binary_leaf` threshold từ PR curve (precision >= 0.95), rồi export ONNX + đóng vào bundle `model_bundles/leaf_gate/` (Mục 7). `class_to_idx` đặt `leaf=1` để khớp `app.py` đọc `softmax(...)[:, 1]`.

---

## 5. Evaluation & Calibration

### 5.1. Định nghĩa metrics

Mỗi báo cáo loài phải có metric tổng hợp **và** per-class. Accuracy đơn lẻ không chấp nhận vì imbalance nặng giấu lỗi minority.

| Metric | Định nghĩa | Vì sao dùng |
|---|---|---|
| Accuracy | Đúng / tổng | Sanity check đơn giản |
| Per-class Precision | TP/(TP+FP) mỗi lớp | Đo báo động giả từng bệnh |
| Per-class Recall | TP/(TP+FN) mỗi lớp | Đo bỏ sót ca bệnh |
| Per-class F1 | Harmonic mean | Tóm tắt per-class chính |
| **Macro-F1** | Mean F1 không trọng số | **Metric chọn model chính** |
| Weighted-F1 | F1 theo cỡ lớp | Hữu ích nhưng giấu rare failure |
| Balanced accuracy | Mean recall | Hữu ích khi imbalance |
| Confusion matrix | Ma trận lỗi | Bắt buộc để chẩn đoán |
| ECE | Expected Calibration Error | Đo độ tin cậy probability |
| Brier score | MSE của probability | Chất lượng confidence |
| NLL | Negative log-likelihood | Mục tiêu temperature scaling |
| `rare_class_recall/precision` | Chỉ khi có lớp Chế độ 2 | Hiện không áp dụng (không lớp <30); thay bằng per-class recall cho lớp ít mẫu như `Spider_mites` |

Vì sao **macro-F1** chứ không accuracy: data imbalance, accuracy bị lớp đa số chi phối; macro-F1 lấy trung bình không trọng số nên phạt nặng việc bỏ rơi minority. Mọi vòng chọn checkpoint trong dự án dùng macro-F1.

Reporting rule: mỗi bảng hiển thị `val`, `calib` (khi liên quan), và `holdout` cuối. Bảng holdout sinh **một lần** sau khi đã chọn model.

### 5.2. Các stage đánh giá

| Stage | Data | Mục đích | Được đổi model sau đó? |
|---|---|---|---|
| Dev sanity | Subset nhỏ | Bắt bug code | Có |
| Validation | `val` split | Chọn model + ablation | Có |
| Calibration | `calib` split | Fit temperature + thresholds | Không đổi kiến trúc sau calibration cuối |
| Robustness | Bản corrupt của `val` | Stress test | Chỉ nếu ghi chú |
| Golden smoke | Golden set | An toàn deploy | Sửa bug, không tune accuracy |
| Holdout | `holdout` split | Metric dự án cuối | **Không** — báo cáo là cuối |
| Cross-test | Public datasets/models | So sánh ngoài | Không, báo cáo riêng |

Bảng report tối thiểu cuối: (1) per-species validation, (2) per-species holdout, (3) per-class P/R/F1 holdout, (4) confusion matrix mỗi loài, (5) calibration trước/sau temperature, (6) OOD/reject, (7) robustness drop, (8) cross-test summary.

```python
# Final_Pipeline/src/eval/evaluate.py
"""Đánh giá — evaluate() là 'đơn vị đo' dùng chung cho train_pretrain/train_finetune
(chọn checkpoint theo macro-F1). Hỗ trợ softmax head + prototype few-shot."""
import argparse
import torch
from sklearn.metrics import f1_score, confusion_matrix, classification_report


@torch.no_grad()
def collect_predictions(model, loader, device, use_amp=True):
    model.eval()
    y_true, y_pred = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            logits = model(imgs)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


def evaluate(model, loader, device, verbose=False, class_names=None, use_amp=True):
    """Trả macro-F1. verbose=True in thêm classification report + confusion matrix."""
    y_true, y_pred = collect_predictions(model, loader, device, use_amp)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if verbose:
        print(f"\n=== Classification report (macro-F1={macro_f1:.4f}) ===")
        print(classification_report(y_true, y_pred, target_names=class_names,
                                    zero_division=0, digits=4))
        print("=== Confusion matrix (hàng=thật, cột=dự đoán) ===")
        print(confusion_matrix(y_true, y_pred))
    return macro_f1


@torch.no_grad()
def evaluate_prototype(query_features, query_labels, prototypes, verbose=False, class_names=None):
    """Đánh giá nhánh Chế độ 2 — trả macro-F1 để so sánh đồng nhất với softmax."""
    from src.train.few_shot import classify_by_prototype
    preds = classify_by_prototype(query_features, prototypes.to(query_features.device)).argmax(1).cpu()
    yt = query_labels.cpu().tolist()
    macro_f1 = f1_score(yt, preds.tolist(), average="macro", zero_division=0)
    if verbose:
        print(classification_report(yt, preds.tolist(), target_names=class_names,
                                    zero_division=0, digits=4))
    return macro_f1


def _run_standalone(config_path, split):
    from src.config import TrainConfig
    from src.models.backbone import build_backbone
    from src.data.datasets import FinetuneDataset
    from src.data.augmentation import get_finetune_transforms
    from src.train.train_finetune import load_manifest
    from torch.utils.data import DataLoader
    cfg = TrainConfig.load(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(cfg["checkpoint"]["out_path"], map_location=device)
    model = build_backbone(ckpt["num_classes"], ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    manifest = load_manifest(cfg["data"]["split_manifest"])
    ds = FinetuneDataset(manifest, split, get_finetune_transforms(False, cfg["data"]["img_size"]))
    loader = DataLoader(ds, cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], pin_memory=True)
    evaluate(model, loader, device, verbose=True, class_names=ckpt["classes"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val", choices=["train", "calib", "val", "holdout"])
    args = ap.parse_args()
    _run_standalone(args.config, args.split)
```

### 5.3. Calibration (temperature scaling + ECE + Brier)

Mạng hiện đại thường **overconfident** (Guo et al., 2017): softmax cao hơn accuracy thực. Với hệ có reject, hệ quả trực tiếp là ngưỡng chọn cảm tính (0.7) không tương ứng mức tin cậy thật. Temperature scaling học **một** scalar `T>0`, chia logits cho `T` trước softmax, tối ưu NLL trên `calib` — **không** đổi argmax (nên không đổi accuracy), chỉ hiệu chỉnh confidence.

**Calibration V2 cho hệ nhiều tầng:** một `T` chung chỉ đủ demo. Bản Final calibration **riêng** cho: `leaf_gate` và từng `species_head`. Mỗi checkpoint sinh `<ckpt>_calibration.json` gồm `temperature`, `ece_before/after`, `brier_before/after`.

> **Kết quả thực tế (calib split, cần cải thiện):** temperature scaling **chưa đạt gate ECE ≤ 0.05** ở hầu hết thành phần. Số đo hiện tại:
>
> | Thành phần | T | ECE trước → sau |
> |---|---:|---|
> | Apple | 0.68 | 0.183 → **0.094** |
> | Corn | 1.10 | 0.112 → 0.133 (xấu đi) |
> | Grape | 1.51 | 0.199 → 0.119 |
> | Pepper bell | 2.06 | 0.272 → 0.213 |
> | Tomato | 1.17 | 0.071 → 0.084 (xấu đi) |
> | Leaf-gate | 2.24 | 0.882 → 0.741 |
>
> Nguyên nhân khả dĩ: calib split nhỏ ở loài ít dữ liệu; leaf-gate cần xem lại nhãn/negatives. **Hệ quả cho reject policy:** vì confidence chưa đáng tin, khi build `thresholds.json` (Mục 5.4) phải đặt ngưỡng **bảo thủ hơn** và không dựa hoàn toàn vào xác suất tuyệt đối. Đây là việc còn lại được đánh dấu ở Mục 1.8.

```python
# Final_Pipeline/src/eval/calibration.py
"""Hiệu chỉnh confidence.
Temperature scaling: Guo et al., 2017 — https://arxiv.org/abs/1706.04599
ECE: Naeini et al., 2015 — https://ojs.aaai.org/index.php/AAAI/article/view/9602
Học DUY NHẤT scalar T>0 tối ưu NLL trên calib split -> KHÔNG đổi accuracy,
chỉ hiệu chỉnh probability -> chọn threshold reject trên xác suất ĐÃ hiệu chỉnh."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, init_T=1.5):
        super().__init__()
        self.log_T = nn.Parameter(torch.log(torch.tensor(init_T)))   # T=exp(log_T)>0 luôn dương

    @property
    def temperature(self):
        return self.log_T.exp().item()

    def forward(self, logits):
        return logits / self.log_T.exp()

    def fit(self, logits, labels, max_iter=50):
        """L-BFGS như đề xuất gốc — hội tụ nhanh với bài toán 1 tham số."""
        logits, labels = logits.detach(), labels.detach()
        opt = torch.optim.LBFGS([self.log_T], lr=0.01, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        opt.step(closure)
        return self.temperature


@torch.no_grad()
def expected_calibration_error(probs, labels, n_bins=15):
    """ECE: chia confidence thành n_bins, |accuracy - confidence| bình quân có trọng số."""
    conf, pred = probs.max(dim=1)
    acc = pred.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=probs.device)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.float().mean() > 0:
            ece += (acc[m].float().mean() - conf[m].mean()).abs() * m.float().mean()
    return ece.item()


@torch.no_grad()
def brier_score(probs, labels):
    """Brier multiclass: MSE giữa probability và one-hot."""
    oh = F.one_hot(labels, probs.size(1)).float()
    return ((probs - oh) ** 2).sum(dim=1).mean().item()


@torch.no_grad()
def collect_logits(model, loader, device, use_amp=True):
    """Trích logits 1 lần rồi tối ưu T trên tensor tĩnh (không forward lại khi L-BFGS lặp)."""
    model.eval()
    all_logits, all_labels = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            logits = model(imgs)
        all_logits.append(logits.float().cpu()); all_labels.append(targets.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def calibrate_head(model, calib_loader, device):
    """Trích logits trên calib -> ECE/Brier trước -> fit T -> ECE/Brier sau.
    Trả dict để ghi calibration.json (Mục 7). Chọn threshold TRÊN xác suất đã chia T."""
    logits, labels = collect_logits(model, calib_loader, device)
    p_before = torch.softmax(logits, dim=1)
    ece_b, brier_b = expected_calibration_error(p_before, labels), brier_score(p_before, labels)
    T = TemperatureScaler().fit(logits, labels)
    p_after = torch.softmax(logits / T, dim=1)
    ece_a, brier_a = expected_calibration_error(p_after, labels), brier_score(p_after, labels)
    print(f"[calibrate] T={T:.4f} | ECE {ece_b:.4f}->{ece_a:.4f} | Brier {brier_b:.4f}->{brier_a:.4f}")
    return {"temperature": T, "ece_before": ece_b, "ece_after": ece_a,
            "brier_before": brier_b, "brier_after": brier_a}
```

CLI chạy calibration cho từng thành phần (species head / leaf-gate) trên `calib` split và ghi `calibration.json`:

```python
# Final_Pipeline/src/eval/calibration.py  (nối tiếp — CLI entry)
def _run_calibration(config_path, target):
    """target='species' | 'leaf_gate'. Trích logits trên calib -> fit T -> ghi calibration.json.
    Chạy: python -m src.eval.calibration --config configs/finetune_tomato.yaml --target species
          python -m src.eval.calibration --config configs/leaf_gate.yaml --target leaf_gate"""
    import json
    import torch
    from torch.utils.data import DataLoader
    from src.config import TrainConfig
    cfg = TrainConfig.load(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if target == "leaf_gate":
        from src.models.backbone import build_leaf_gate
        from src.train.train_leaf_gate import LeafGateDataset
        from src.data.augmentation import get_finetune_transforms
        ckpt = torch.load(cfg["checkpoint"]["out_path"], map_location=device)
        model = build_leaf_gate(pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"])
        import os
        ds = LeafGateDataset(os.path.join(cfg["data"]["root"], "calib"),
                             get_finetune_transforms(False, cfg["data"]["img_size"]))
    else:
        from src.models.backbone import build_backbone
        from src.data.datasets import FinetuneDataset
        from src.data.augmentation import get_finetune_transforms
        from src.train.train_finetune import load_manifest
        ckpt = torch.load(cfg["checkpoint"]["out_path"], map_location=device)
        model = build_backbone(ckpt["num_classes"], ckpt["arch"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        manifest = load_manifest(cfg["data"]["split_manifest"])
        ds = FinetuneDataset(manifest, "calib", get_finetune_transforms(False, cfg["data"]["img_size"]))

    loader = DataLoader(ds, cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], pin_memory=True)
    cal = calibrate_head(model, loader, device)
    out = cfg["checkpoint"]["out_path"].replace(".pt", "_calibration.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=2)
    print(f"[calibration] -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Calibration temperature scaling trên calib split")
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", default="species", choices=["species", "leaf_gate"])
    a = ap.parse_args()
    _run_calibration(a.config, a.target)
```

> Prototype branch (Chế độ 2) hiện **không có** vì không còn lớp hiếm (Mục 3.5); khi nào tái kích hoạt thì mới cần `T_proto`/`radius_p95`. CLI hiện hỗ trợ hai `--target`: `species` và `leaf_gate`.

### 5.4. Thresholds & reject policy tuning

> **Trạng thái: CHƯA build `thresholds.json`.** Hiện API chạy với default (`binary_leaf=0.5`, `T_energy=1e9` → energy-OOD gần như không kích hoạt, `T_low_confidence=0.0`, `T_margin=0.0`). Nghĩa là các nhánh reject `unknown_or_ood` / `low_confidence` / `ambiguous` **về mặt hiệu lực chưa hoạt động** — đây là việc còn lại quan trọng nhất của tầng serving (Mục 1.8).

Sau khi có `T`, tune threshold trên `calib` (không phải val, không phải holdout):

| Threshold | Cách chọn | Mục tiêu |
|---|---|---|
| `T_low_confidence` | Quét trên coverage–accuracy curve | Accuracy tối thiểu cho ca chấp nhận |
| `T_margin` | Phân phối (top1−top2) khi đúng vs sai | Tách ca mơ hồ |
| `T_energy` | Phân phối energy in-dist vs OOD | Chặn OOD với false-reject thấp |
| `binary_leaf` | PR curve leaf-gate | Precision >= 0.95 để hạn chế nhận nhầm |

Reject taxonomy trả về (khớp API Mục 7): `not_a_leaf`, `unknown_or_ood`, `low_confidence`, `ambiguous`, `ok`. Vì calibration chưa đạt ECE tốt (Mục 5.3), ngưỡng nên đặt bảo thủ và ưu tiên leaf-gate + margin hơn là xác suất tuyệt đối.

### 5.5. Đánh giá OOD & reject

> **Trạng thái: khung.** `src/eval/ood_eval.py` chỉ cung cấp helper `fpr_at_tpr()` + `ood_auroc()`; chưa có runner đầu-cuối gắn tập OOD thật, và `T_energy` chưa được tune (Mục 5.4). Phần dưới là thiết kế mục tiêu.

Ba nhóm input cần từ chối đúng:

| Nhóm OOD | Ví dụ | Reject mong đợi |
|---|---|---|
| Không phải lá | Tay, đất, trời, vật thể | `not_a_leaf` (leaf-gate) |
| Lá sai loài | Chọn tomato nhưng upload lá nho | `unknown_or_ood` (energy/prototype) |
| Bệnh ngoài danh mục | Bệnh không có trong 20 lớp | `unknown_or_ood` hoặc `low_confidence` |
| Ảnh chất lượng kém | Quá mờ/tối/sáng | `low_confidence` |

Metric OOD: FPR@95%TPR (tỉ lệ nhận nhầm OOD khi giữ 95% in-dist), AUROC energy-vs-softmax, reject-accuracy tradeoff (coverage–accuracy). Tập OOD lấy từ ảnh non-leaf curated + lá loài khác + ảnh cố ý làm hỏng.

```python
# Final_Pipeline/src/eval/ood_eval.py
"""Đánh giá OOD/reject: AUROC + FPR@95%TPR trên điểm phân tách in-dist vs OOD."""
import numpy as np
from sklearn.metrics import roc_auc_score


def fpr_at_tpr(scores_in, scores_out, tpr_target=0.95):
    """scores: CAO = giống in-dist. Tìm ngưỡng giữ tpr_target của in-dist, đo FPR trên OOD."""
    thr = np.quantile(scores_in, 1 - tpr_target)
    fpr = (np.array(scores_out) >= thr).mean()
    return float(fpr), float(thr)


def ood_auroc(scores_in, scores_out):
    y = np.concatenate([np.ones_like(scores_in), np.zeros_like(scores_out)])
    s = np.concatenate([scores_in, scores_out])
    return float(roc_auc_score(y, s))
```

### 5.6. Robustness suite

> **Trạng thái: khung.** `src/eval/robustness.py` định nghĩa `CORRUPTIONS` (5 loại: blur, brightness, noise, jpeg, rotate) + `build_corruption_transform()`; chưa có runner đầu-cuối tự động quét toàn bộ. Các loại occlusion/motion-blur/downscale trong bảng dưới là mục tiêu mở rộng.

Vì input thật là ảnh điện thoại ngoài đồng, robustness là **bằng chứng chính** chứ không phải phụ. Áp corruption lên bản sao `val` (không đụng holdout), báo cáo macro-F1 drop từng mức.

| Corruption | Mức | Mô phỏng | Có trong code |
|---|---|---|---|
| Gaussian blur | 3/7/13 | Lấy nét kém | ✅ |
| Brightness | +20% / +40% | Nắng/bóng | ✅ |
| Gaussian noise | std 0.04/0.1/0.2 | ISO cao | ✅ |
| JPEG | quality 50/30/15 | Nén web | ✅ |
| Rotation | 15/30/45° | Cầm nghiêng | ✅ |
| Motion blur / Occlusion / Downscale | — | Rung tay / lá che / ảnh xa | 🔵 mở rộng |

```python
# Final_Pipeline/src/eval/robustness.py
"""Benchmark robustness: áp corruption tăng dần, đo macro-F1 drop. KHÔNG dùng holdout."""
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
# Với mỗi (loại, mức): build val loader kèm corruption -> evaluate() -> ghi macro-F1 drop.
```

Fairness theo điều kiện chụp: hiệu năng không nên lệch mạnh theo loài, lớp, nguồn ảnh, thiết bị, ánh sáng, nền, nhóm rare-vs-regular. Thiếu metadata thì ghi rõ giới hạn (chi tiết Mục 8.13).

---

## 6. Cross-Test Plan (so sánh với nghiên cứu công bố)

> **Trạng thái:** **Chiều 2 (public model → holdout field của mình) ĐÃ CHẠY** (2026-07-12) — code ở `src/cross_test/run_direction2.py` + `public_models.py` + `label_maps.py`, kết quả ở Mục 6.5. **Chiều 1 (model mình → PlantDoc) tạm gác** vì FINETUNE đã chứa 742 ảnh `PLANTDOC_` (cần de-dup hash trước khi so, chưa làm). `evaluate_with_label_map()`/`summarize()` trong `run_cross_test.py` là helper cho chiều 1 khi triển khai sau.

Mục tiêu: đặt pipeline cạnh công trình đã công bố trên **cùng dataset công khai** để (1) có mốc tham chiếu định lượng macro-F1/accuracy, và (2) chứng minh điểm mạnh riêng — khả năng thu hẹp domain gap lab→field, nơi hầu hết paper đạt >99% trên PlantVillage đều sụp đổ trên ảnh field.

### 6.1. Papers & benchmark tham chiếu

> **Trạng thái xác minh nguồn:** đã đối chiếu online (2026-07-12) với arXiv, DOI.org và trang nhà xuất bản. P1–P7 xác nhận đúng; tiêu đề P6/P7 đã cập nhật theo bản published (trước đó là mô tả rút gọn). Chi tiết ở Mục 12.

| # | Bài báo | Năm | Dataset (công khai?) | Model chính | Metric chính | Link |
|---|---|---|---|---|---|---|
| P1 | Hughes & Salathé — *An open access repository of images on plant health* | 2015 | **PlantVillage** — 54,306 ảnh, 38 lớp (14 loài) ✅ | (bài giới thiệu dataset) | — (dataset gốc) | [arXiv:1511.08060](https://arxiv.org/abs/1511.08060) |
| P2 | Mohanty, Hughes, Salathé — *Using Deep Learning for Image-Based Plant Disease Detection* | 2016 | PlantVillage ✅ | GoogLeNet/AlexNet (transfer) | Acc **99.35%** trên test PlantVillage; **~31%** trên ảnh internet OOD | [DOI:10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419) |
| P3 | Singh, Misra, Okafor et al. — *PlantDoc: A Dataset for Visual Plant Disease Detection* | 2020 | **PlantDoc** — 2,598 ảnh field, 13 loài, 17 lớp ✅ | dataset field + CNN baseline | +31% accuracy khi thêm PlantDoc vào training | [arXiv:1911.10317](https://arxiv.org/abs/1911.10317) |
| P4 | Borhani, Khoramdel, Najafi — *A deep learning based approach ... using vision transformer* | 2022 | PlantVillage (+biến thể) ✅ | ViT + bản lai CNN+ViT | Acc ~**99%** trên PlantVillage | [DOI:10.1038/s41598-022-15163-0](https://doi.org/10.1038/s41598-022-15163-0) |
| P5 | Li et al. — *PMVT: a lightweight vision transformer for plant disease identification* | 2023 | Nhiều bộ field ✅ | PMVT (MobileViT-based, ~2–3M) | Acc cạnh tranh, tham số rất thấp | [DOI:10.3389/fpls.2023.1256773](https://doi.org/10.3389/fpls.2023.1256773) |
| P6 | Wu et al. — *From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease Recognition in the Wild* | 2023 | Lab + field ✅ | CNN + UDA (domain alignment) | Cải thiện accuracy miền đích (field) | [DOI:10.34133/plantphenomics.0038](https://doi.org/10.34133/plantphenomics.0038) |
| P7 | Jeon et al. — *Bridging the Lab-to-Field Gap in Plant Disease Diagnosis Through Unsupervised Domain Adaptation Enhanced by Background Recomposition* | 2025 | Lab + nền field ✅ | CNN + tái tổ hợp nền field | +9.9–23.6 điểm % trên ảnh field | [SSRN:10.2139/ssrn.5217081](https://dx.doi.org/10.2139/ssrn.5217081) |

Ghi chú thay đổi so với nháp: sửa 1 lỗi tên tác giả (P3); nâng cấp trạng thái xuất bản 1 mục (P7). Hai con số 99.35% và ~31% ở P2 giữ nguyên (đã đúng nguyên văn). P1–P4 là mốc benchmark kinh điển; P5–P7 (2023–2025) trực tiếp giải bài toán khó của dự án (lab→field, model nhẹ) và đã được dùng làm căn cứ kỹ thuật ở Mục 3–4 — nhất quán nội tại.

### 6.2. Phát hiện cốt lõi để định vị dự án

Điểm mấu chốt từ P2 và P3: **accuracy cao trên PlantVillage gần như vô nghĩa nếu không kiểm trên ảnh field.** Mohanty et al. (2016) đạt 99.35% trên PlantVillage nhưng rơi còn ~31% khi ảnh test đến từ điều kiện khác; Singh et al. (2020) tạo PlantDoc chính vì lý do đó. Nên pipeline này **không** đặt mục tiêu ganh đua 99.x% trên lab, mà tập trung:

- **Domain adaptation chủ động** (Mục 4.6: field-background recomposition + MMD) — đối chiếu P6, P7.
- **Xử lý imbalance & lớp hiếm** (Mục 4.4, 3.5: CB-Focal + prototype few-shot) — điểm P2/P4 bỏ qua vì PlantVillage khá cân bằng.
- **Model nhẹ chạy 4GB VRAM** (Mục 3.1) — cùng triết lý P5 (PMVT).

### 6.3. Ba kịch bản test chéo (đo domain gap)

Con số accuracy đơn lẻ trên val không nói lên khoảng cách miền. Đề xuất **3 kịch bản**:

1. **In-domain (lab → lab):** train trên PlantVillage/PRETRAIN, test trên val cùng phân phối. Con số "trần trên", so với P2/P4.
2. **Cross-domain zero-shot (lab → field):** train **chỉ** trên lab, test **không** fine-tune trên PlantDoc (giao loài/bệnh). Phơi bày domain gap — kỳ vọng thấp, giống P2.
3. **Cross-domain + adaptation (lab + domain-aug → field):** bật field-background recomposition (Mục 4.6 Cách 1) và/hoặc MMD (Cách 2), test lại PlantDoc.

> **Độ chênh (3) − (2) chính là "giá trị gia tăng" định lượng của domain adaptation** — đây là biểu đồ trung tâm của báo cáo. Nếu (3) > (2) rõ rệt: bằng chứng trực tiếp domain adaptation hoạt động; nếu không: soi lại chất lượng ảnh nền field hoặc `lambda_mmd`.

Ngoài ra, chiều ngược lại (**public model → dữ liệu của ta**): chạy một checkpoint PlantVillage công khai trên FINETUNE holdout của ta, để định vị dữ liệu ta khó tới đâu so với lab.

### 6.4. Quy tắc mapping nhãn & de-duplication

PlantDoc và PlantVillage **không** dùng cùng bộ tên lớp. Trước khi test kịch bản 2–3:

| Quy tắc | Bắt buộc |
|---|---|
| Bảng ánh xạ nhãn thủ công | Ví dụ `Tomato_late_blight` ↔ `Tomato Late blight`; chỉ đánh giá trên **giao** hai tập lớp |
| Chỉ giao lớp | Bỏ lớp không map được khỏi metric, ghi rõ số lớp đã dùng |
| De-dup xuyên dataset | Đối chiếu file hash; ảnh resize/đổi tên dùng perceptual hash |
| Không rò rỉ | Không ảnh cross-test nào trùng train/calib/val (Mục 2.5) |
| Ghi chú độ phủ | Report nêu rõ % lớp/loài được so sánh, tránh macro-F1 sai lệch |

Bỏ qua mapping/de-dup là lỗi âm thầm hay gặp nhất khi so cross-dataset — làm macro-F1 lệch hẳn.

### 6.5. Kết quả Chiều 2 (public model → holdout FIELD của mình) — ĐÃ CHẠY 2026-07-12

Chạy 2 model chỉ-biết-lab trên đúng holdout field 5 loài của mình, đặt cạnh model đã fine-tune.
Đánh giá **closed-set per-species** (mỗi loài chỉ xét K lớp thuộc loài, công bằng với head per-species).

| Model | Train trên | apple | corn | grape | pepper | tomato | **macro-F1 TB** | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Của mình (fine-tune FIELD)** | field | 0.841 | 0.591 | 0.487 | 0.667 | 0.391 | **0.595** | 0.780 |
| DeiT-39 ngoài | PlantVillage lab | 0.592 | 0.259 | 0.649 | 0.703 | 0.193 | **0.479** | 0.559 |
| Backbone-38 của mình | PRETRAIN lab | 0.244 | 0.157 | 0.487 | 0.540 | 0.133 | **0.312** | 0.236 |

Nguồn model: DeiT — [rescu/deit-base-patch16-224-finetuned-plantvillage](https://huggingface.co/rescu/deit-base-patch16-224-finetuned-plantvillage) ([config 39 lớp](https://huggingface.co/rescu/deit-base-patch16-224-finetuned-plantvillage/raw/main/config.json)); Backbone-38 = `checkpoints/backbone_pretrained.pt` (PlantVillage-38, dataset [Hughes & Salathé 2015](https://arxiv.org/abs/1511.08060)).

**Cách đọc:**
- Cả hai model **chỉ-biết-lab** đều thua rõ model fine-tune field (0.312 và 0.479 vs **0.595**) → **lượng hoá domain gap lab→field**, đúng phát hiện Mohanty et al. 2016 ([DOI](https://doi.org/10.3389/fpls.2016.01419)).
- **Kiểm chứng không phải model hỏng:** backbone-38 đạt **99.77% macro-F1 trên PRETRAIN lab val** và đoán đúng 100% Apple_scab ảnh lab — nhưng chỉ 26% trên Apple field. Sụt hoàn toàn do đổi miền.
- DeiT ngoài > backbone của mình vì DeiT lớn hơn (ViT-base 86M) và train đa dạng hơn; nhưng vẫn dưới model field-adapted.
- Cột `open_offspecies_rate` trong `reports/cross_test/direction2_*_summary.json` cho biết tỉ lệ model công khai đoán "lạc sang loài khác" khi cho chọn toàn 38/39 lớp (open-set) — cao ở apple/grape, phản ánh nhầm lẫn xuyên loài trên ảnh field.

Cách chạy lại:
```bash
python -m src.cross_test.run_direction2 --model mine       # baseline fine-tune (offline)
python -m src.cross_test.run_direction2 --model backbone   # B: lab-38 của mình (offline)
python -m src.cross_test.run_direction2 --model deit       # A: DeiT ngoài (cần transformers + tải HF ~330MB)
```

### 6.5b. Chiều 1 (model mình → dataset công bố) — CHƯA CHẠY

| Kịch bản | Dataset test | Trạng thái | Ghi chú |
|---|---|---|---|
| Zero-shot lab→field | PlantDoc (giao lớp) | 🔵 tạm gác | FINETUNE đã chứa 742 ảnh `PLANTDOC_` → cần de-dup SHA256 trước khi so |
| + Domain adaptation | PlantDoc (giao lớp) | 🔵 tạm gác | Bật field-bg/MMD rồi test lại (Mục 4.6) |

Khi làm chiều 1: `git clone https://github.com/pratikkayal/PlantDoc-Dataset` (CC-BY-4.0), xây bảng ánh xạ 28 lớp PlantDoc ↔ 20 lớp mình, **de-dup bằng file hash** với train/calib/val, rồi dùng `evaluate_with_label_map()`/`summarize()` trong `run_cross_test.py`.

```python
# Final_Pipeline/src/cross_test/run_cross_test.py  (helper cho CHIỀU 1 — điền dataset khi làm)
"""Chạy cross-test theo mapping nhãn giao + log prediction để audit (Mục 6).
Mỗi prediction ghi: image_id, source_dataset, mapped_label, pred_label, confidence, correct."""
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
```

---

## 7. Deployment (Web tối ưu)

Mục tiêu tầng phục vụ: chạy được **trên máy không GPU** (VPS rẻ hoặc chính laptop khi demo), phản hồi thấp, image Docker nhẹ. Chìa khoá: **tách huấn luyện (PyTorch + CUDA) khỏi phục vụ (ONNX Runtime CPU)** — inference không cần PyTorch, không cần CUDA. Bundle model được phân phối qua **Hugging Face Hub** và tải về khi cần (Mục 7.2).

Kiến trúc suy luận một ảnh:

```text
Ảnh upload
    │
    ├─(0) Validate: dung lượng, MIME, magic bytes, pixel, EXIF (Mục 7.5)
    │
    ├─(1) Leaf-gate (MobileNetV3-Small, Mục 3.2)
    │      softmax(binary_logits / T) < ngưỡng?         -> REJECT not_a_leaf
    │      energy_score(species_logits) > T_energy?     -> REJECT unknown_or_ood
    │
    ├─(2) Model loài (per-species, Chế độ 1): softmax head
    │      argmax + confidence đã calibration (chia T)
    │      margin top1-top2 < T_margin?                 -> ambiguous
    │      confidence < T_low_confidence?               -> low_confidence
    │
    └─(3) Nếu loài có prototype (Chế độ 2) và nghi lớp hiếm:
           embedding -> classify_by_prototype -> hợp nhất (Mục 3.5)
```

### 7.1. Model bundle bất biến

Mỗi model deploy là một **bundle bất biến** trong `model_bundles/<model_version>/`. API **không** load rời rạc checkpoint/mapping/threshold/prototype từ nhiều nơi. Bundle tối thiểu: `model.onnx`, `manifest.json`, `class_mapping.json`, `preprocessing.json`, `calibration.json`, `thresholds.json`, `taxonomy.json`, `metrics.json`, `parity_report.json`, `checksums.sha256`, và nếu có lớp hiếm thì `prototypes.npz`.

```json
{
  "model_version": "tomato_2026_07_04_effv2s_ab12cd",
  "species": "tomato",
  "git_commit": "ab12cd34",
  "config_hash": "sha256:...",
  "dataset_version": "finetune_split_v1_2026_07_04",
  "taxonomy_version": "taxonomy_v1",
  "split_manifest_sha256": "sha256:...",
  "onnx": {"file": "model.onnx", "opset": 17, "quantization": "fp32"},
  "class_mapping": "class_mapping.json",
  "preprocessing": "preprocessing.json",
  "calibration": "calibration.json",
  "thresholds": "thresholds.json",
  "prototype_file": "prototypes.npz",
  "metrics": "metrics.json",
  "parity_report": "parity_report.json",
  "created_at": "2026-07-04T00:00:00+07:00"
}
```

**Class mapping là thành phần bắt buộc của bundle.** API load `idx_to_class` **từ bundle**, tuyệt đối không tự suy bằng `sorted(os.listdir())`. Golden test kiểm cùng một ảnh cho ra cùng index và cùng tên lớp ở PyTorch, ONNX và API (Mục 2.7).

`preprocessing.json` khoá resize/normalization để training và serving khớp byte-for-byte về mặt logic:

```json
{
  "img_size": 224, "resize": "bilinear",
  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
  "channel_order": "RGB", "layout": "NCHW"
}
```

### 7.2. Registry & phân phối bundle (Hugging Face Hub)

**Thực tế triển khai:** tầng serving **không** đọc registry cục bộ `active_version_*.txt` lúc chạy, mà **tải bundle từ Hugging Face Hub** (private repos) qua `huggingface_hub.snapshot_download`, xác thực bằng biến môi trường `HF_TOKEN`. Ánh xạ loài → repo được khai báo trong `app.py` (`REPO_MAP`):

| Loài | HF repo |
|---|---|
| apple | `ruhua2305/apple_effv2s_onnx` |
| corn | `ruhua2305/corn_effv2s_onnx` |
| grape | `ruhua2305/grape_effv2s_onnx` |
| pepper_bell | `ruhua2305/pepper_bell_effv2s_onnx` |
| tomato | `ruhua2305/tomato_effv2s_onnx` |
| leaf-gate | `ruhua2305/leaf_gate_MobileNetV3Small` |

**Lazy load per-species:** startup chỉ tải leaf-gate (~6MB); model từng loài chỉ được tải khi có request đầu tiên cho loài đó (tiết kiệm RAM cho VPS nhỏ). Bundle được cache trong thư mục tạm (`hf_cache`).

Registry cục bộ `model_bundles/active_version_<species>.txt` + `build_bundle.py` vẫn được dùng để **đóng gói và version hoá offline** trước khi push lên HF Hub; đây là phương án thay thế khi chạy hoàn toàn local không mạng.

### 7.3. ONNX export + quantization

Xuất từng model PyTorch (`model_<species>.pt`, `leaf_gate.pt`) sang ONNX, tuỳ chọn lượng tử hoá:

- **FP16** — giảm ~50% dung lượng, gần như không mất accuracy; hợp khi CPU/GPU hỗ trợ half tốt.
- **INT8 (dynamic)** — giảm ~75%, nhanh hơn trên CPU, đổi một chút accuracy → **bắt buộc đo lại macro-F1 + parity** trước khi chốt (Jacob et al., 2017).

```python
# Final_Pipeline/src/export/export_onnx.py
"""PyTorch -> ONNX cho tầng phục vụ. ONNX inference KHÔNG cần PyTorch/CUDA -> Docker nhẹ.
Chạy: python -m src.export.export_onnx --config configs/finetune_tomato.yaml --out model_bundles/tmp/tomato/model.onnx"""
import argparse
import json
import os
import torch
from src.config import TrainConfig
from src.models.backbone import build_backbone


def export_to_onnx(config_path, out_path, img_size=224, opset=17):
    device = "cpu"        # export CPU đủ nhanh, tránh phụ thuộc CUDA lúc build
    cfg = TrainConfig.load(config_path)
    ckpt = torch.load(cfg["checkpoint"]["out_path"], map_location=device)
    model = build_backbone(ckpt["num_classes"], ckpt["arch"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(model, dummy, out_path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, do_constant_folding=True)
    # class_mapping.json cạnh model -> API map index -> tên lớp (Mục 7.1)
    meta = out_path.replace("model.onnx", "class_mapping.json")
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({"idx_to_class": {i: c for i, c in enumerate(ckpt["classes"])},
                   "arch": ckpt["arch"], "img_size": img_size,
                   "species": ckpt.get("species", "")}, f, ensure_ascii=False, indent=2)
    print(f"[export] {out_path} + {meta} ({ckpt['num_classes']} lớp)")
    return out_path


def quantize_dynamic_int8(onnx_path, out_path):
    """INT8 dynamic — giảm ~75%. BẮT BUỘC đo lại macro-F1 + parity sau bước này."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(onnx_path, out_path, weight_type=QuantType.QInt8)
    print(f"[int8] {os.path.getsize(onnx_path)/1e6:.1f}MB -> {os.path.getsize(out_path)/1e6:.1f}MB")


def convert_fp16(onnx_path, out_path):
    import onnx
    from onnxconverter_common import float16
    onnx.save(float16.convert_float_to_float16(onnx.load(onnx_path)), out_path)
    print(f"[fp16] -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantize", default="none", choices=["none", "int8", "fp16"])
    a = ap.parse_args()
    export_to_onnx(a.config, a.out)
    if a.quantize == "int8":
        quantize_dynamic_int8(a.out, a.out.replace(".onnx", "_int8.onnx"))
    elif a.quantize == "fp16":
        convert_fp16(a.out, a.out.replace(".onnx", "_fp16.onnx"))
```

### 7.4. Parity test (bắt buộc trước deploy)

Sau export, chạy cùng golden set qua PyTorch FP32/AMP, ONNX FP32 và ONNX INT8. Chỉ deploy INT8 nếu `macro_f1_drop <= 0.01`, `rare_recall_drop <= 0.03`, `ECE_delta <= 0.02` và top-1 mismatch trong ngưỡng. Kết quả lưu `parity_report.json` trong bundle.

```python
# Final_Pipeline/src/export/parity_test.py
"""Parity PyTorch vs ONNX: cùng input -> so top-1 + max abs logit diff.
Chạy: python -m src.export.parity_test --bundle model_bundles/tomato_.../ --golden-set data/golden/tomato.json"""
import argparse
import json
import numpy as np
import onnxruntime as ort
import torch
from src.config import TrainConfig
from src.models.backbone import build_backbone


def run_parity(pt_ckpt, onnx_path, sample_batch_np, atol=1e-3):
    """sample_batch_np: [N,3,H,W] float32 đã preprocess. Trả (top1_match_rate, max_abs_diff)."""
    device = "cpu"
    ckpt = torch.load(pt_ckpt, map_location=device)
    model = build_backbone(ckpt["num_classes"], ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    with torch.no_grad():
        pt_logits = model(torch.from_numpy(sample_batch_np)).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logits"], {"input": sample_batch_np})[0]

    top1_match = (pt_logits.argmax(1) == onnx_logits.argmax(1)).mean()
    max_diff = float(np.abs(pt_logits - onnx_logits).max())
    passed = bool(top1_match == 1.0 and max_diff < atol)
    return {"top1_match_rate": float(top1_match), "max_abs_diff": max_diff,
            "atol": atol, "passed": passed}
# Ghi kết quả -> bundle/parity_report.json. passed=False -> CHẶN release (Mục 8.14).
```

### 7.4b. Đóng gói bundle bất biến (`build_bundle.py`)

Bước này gom mọi artefact rời (ONNX, mapping, calibration, thresholds, prototype, metrics, parity) vào một thư mục `model_bundles/<model_version>/`, sinh `manifest.json` + `checksums.sha256`. Sau khi build, bundle là **read-only**; API chỉ load từ bundle (Mục 7.1).

```python
# Final_Pipeline/src/export/build_bundle.py
"""Đóng gói bundle bất biến (Mục 7.1). Gom artefact -> manifest.json + checksums.sha256.
Chạy: python -m src.export.build_bundle --species tomato --version tomato_2026_07_04_effv2s_ab12cd \
        --onnx model_bundles/tmp/tomato/model.onnx --git-commit ab12cd34"""
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone, timedelta

BUNDLES = r"D:/DAT301m_Project/Final_Pipeline/model_bundles"
CKPT = r"D:/DAT301m_Project/Final_Pipeline/checkpoints"


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _copy_if_exists(src, dst_dir, dst_name):
    """Copy artefact vào bundle nếu tồn tại; trả tên file hoặc None (ghi rõ khi thiếu)."""
    if src and os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, dst_name))
        return dst_name
    print(f"  [WARN] thiếu artefact: {src} (bỏ qua {dst_name})")
    return None


def build_bundle(species, version, onnx_path, git_commit, opset=17, quantization="fp32"):
    out = os.path.join(BUNDLES, version)
    os.makedirs(out, exist_ok=True)

    # 1) Bắt buộc: ONNX + class_mapping + preprocessing
    files = {}
    files["onnx"] = _copy_if_exists(onnx_path, out, "model.onnx")
    cmap_src = onnx_path.replace("model.onnx", "class_mapping.json")
    files["class_mapping"] = _copy_if_exists(cmap_src, out, "class_mapping.json")
    # preprocessing.json cố định theo train (Mục 7.1)
    with open(os.path.join(out, "preprocessing.json"), "w", encoding="utf-8") as f:
        json.dump({"img_size": 224, "resize": "bilinear",
                   "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                   "channel_order": "RGB", "layout": "NCHW"}, f, indent=2)
    files["preprocessing"] = "preprocessing.json"

    # 2) Tuỳ artefact có sẵn: calibration / thresholds / prototypes / metrics / parity
    base = os.path.join(CKPT, f"model_{species}")
    files["calibration"] = _copy_if_exists(base + "_calibration.json", out, "calibration.json")
    files["thresholds"] = _copy_if_exists(base + "_thresholds.json", out, "thresholds.json")
    files["prototype_file"] = _copy_if_exists(base + "_prototypes.pt", out, "prototypes.pt")
    files["metrics"] = _copy_if_exists(base + "_metrics.json", out, "metrics.json")
    files["parity_report"] = _copy_if_exists(
        onnx_path.replace("model.onnx", "parity_report.json"), out, "parity_report.json")
    files["taxonomy"] = _copy_if_exists(
        r"D:/DAT301m_Project/Final_Pipeline/configs/taxonomy_v1.yaml", out, "taxonomy.json")

    # 3) manifest.json
    manifest = {
        "model_version": version, "species": species, "git_commit": git_commit,
        "onnx": {"file": "model.onnx", "opset": opset, "quantization": quantization},
        **{k: v for k, v in files.items() if v},
        "created_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 4) checksums.sha256 cho MỌI file trong bundle (trừ chính nó)
    with open(os.path.join(out, "checksums.sha256"), "w", encoding="utf-8") as f:
        for fn in sorted(os.listdir(out)):
            if fn == "checksums.sha256":
                continue
            f.write(f"{_sha256(os.path.join(out, fn))}  {fn}\n")

    # 5) Gate: bắt buộc phải có ONNX + class_mapping (Mục 7.1, 8.14)
    if not (files["onnx"] and files["class_mapping"]):
        raise RuntimeError("Bundle THIẾU model.onnx hoặc class_mapping.json — CHẶN release.")
    print(f"[bundle] {out} sẵn sàng ({len([v for v in files.values() if v])} artefact).")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Đóng gói bundle bất biến (Mục 7.1)")
    ap.add_argument("--species", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--git-commit", default="unknown")
    ap.add_argument("--quantization", default="fp32", choices=["fp32", "fp16", "int8"])
    a = ap.parse_args()
    build_bundle(a.species, a.version, a.onnx, a.git_commit, quantization=a.quantization)
```

> `test_bundle_checksum.py` (Mục 8.10) đọc `checksums.sha256` và re-hash từng file để phát hiện bundle bị sửa. Con trỏ `active_version_<species>.txt` chỉ được trỏ tới bundle đã build xong + parity pass (Mục 8.13).

### 7.5. API phục vụ (FastAPI + ONNX Runtime + HF Hub)

`app.py` tải leaf-gate lúc startup và **lazy load** model từng loài từ Hugging Face Hub khi có request đầu tiên, tiền xử lý khớp `preprocessing.json`, và ra quyết định leaf-gate trước khi chẩn đoán loài. Reject taxonomy khớp Mục 5.4 (nhưng threshold hiện dùng default vì chưa build `thresholds.json`).

```python
# Final_Pipeline/src/deploy/app.py
"""API chẩn đoán bệnh lá — FastAPI + ONNX Runtime (CPU-only, không CUDA).
Tải bundle từ Hugging Face Hub (private repos) khi startup/lazy-load.
Luồng: validate -> leaf-gate -> loài -> [prototype].
Chạy dev: uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000"""
import io
import json
import os
import tempfile
import glob
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageOps
from huggingface_hub import snapshot_download

app = FastAPI(title="Plant Disease Diagnosis API", version="1.0")

# Ánh xạ loài -> repo_id trên Hugging Face (Mục 7.2)
REPO_MAP = {
    "apple": "ruhua2305/apple_effv2s_onnx",
    "corn": "ruhua2305/corn_effv2s_onnx",
    "grape": "ruhua2305/grape_effv2s_onnx",
    "pepper_bell": "ruhua2305/pepper_bell_effv2s_onnx",
    "tomato": "ruhua2305/tomato_effv2s_onnx",
}
LEAF_GATE_REPO = "ruhua2305/leaf_gate_MobileNetV3Small"

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 12_000_000
ALLOWED_MAGIC = {b"\xff\xd8\xff": "jpg", b"\x89PNG\r\n\x1a\n": "png"}

_sessions, _classes, _preproc, _cal, _thr = {}, {}, {}, {}, {}
_leaf_sess, _leaf_T = None, 1.0


def _load_session(path):
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False            # tiết kiệm RAM
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def _load_model_from_hf(repo_id, token=None):
    """Tải toàn bộ file của repo về thư mục tạm (có cache) và trả đường dẫn."""
    return snapshot_download(repo_id=repo_id, token=token, local_dir_use_symlinks=False,
                             cache_dir=os.path.join(tempfile.gettempdir(), "hf_cache"))


def _load_species_model(species):
    """Lazy load model 1 loài từ HF Hub, tìm .onnx + class_mapping/preprocessing/calibration/thresholds."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    bundle_dir = _load_model_from_hf(REPO_MAP[species], token=token)
    onnx_files = glob.glob(os.path.join(bundle_dir, "**", "*.onnx"), recursive=True)
    if not onnx_files:
        raise FileNotFoundError(f"Không tìm thấy .onnx trong {bundle_dir}")
    _sessions[species] = _load_session(onnx_files[0])
    # class_mapping.json (idx_to_class) — LẤY TỪ bundle, không suy từ folder (Mục 7.1)
    cm_files = glob.glob(os.path.join(bundle_dir, "**", "class_mapping.json"), recursive=True)
    if cm_files:
        cm = json.load(open(cm_files[0], encoding="utf-8"))
        idx_map = cm["idx_to_class"]
        _classes[species] = [idx_map[str(i)] for i in range(len(idx_map))]
    else:
        _classes[species] = ["unknown"]
    # preprocessing / calibration / thresholds (default nếu thiếu)
    pp = glob.glob(os.path.join(bundle_dir, "**", "preprocessing.json"), recursive=True)
    _preproc[species] = json.load(open(pp[0], encoding="utf-8")) if pp else \
        {"img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    cal = glob.glob(os.path.join(bundle_dir, "**", "calibration.json"), recursive=True)
    _cal[species] = json.load(open(cal[0], encoding="utf-8")) if cal else {"temperature": 1.0}
    thr = glob.glob(os.path.join(bundle_dir, "**", "thresholds.json"), recursive=True)
    _thr[species] = json.load(open(thr[0], encoding="utf-8")) if thr else {}


@app.on_event("startup")
def _startup():
    """Chỉ tải leaf_gate lúc startup (nhẹ ~6MB); model loài lazy load khi có request."""
    global _leaf_sess, _leaf_T
    token = os.environ.get("HF_TOKEN")
    try:
        lg_dir = _load_model_from_hf(LEAF_GATE_REPO, token=token)
        lg_onnx = glob.glob(os.path.join(lg_dir, "**", "*.onnx"), recursive=True)
        if lg_onnx:
            _leaf_sess = _load_session(lg_onnx[0])
            lg_cal = glob.glob(os.path.join(lg_dir, "**", "calibration.json"), recursive=True)
            if lg_cal:
                _leaf_T = json.load(open(lg_cal[0], encoding="utf-8")).get("temperature", 1.0)
    except Exception as e:
        print(f"[ERROR] Không thể tải leaf_gate: {e}")
    print(f"[startup] leaf_gate={'on' if _leaf_sess else 'off'}")


def validate_image_upload(raw: bytes) -> Image.Image:
    """Kiểm tra an toàn upload (Mục 7.6) — trả PIL RGB đã fix EXIF."""
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, detail={"code": "file_too_large", "message": "Ảnh vượt quá 8MB."})
    if not any(raw.startswith(sig) for sig in ALLOWED_MAGIC):
        raise HTTPException(415, detail={"code": "unsupported_media", "message": "Chỉ hỗ trợ JPEG/PNG hợp lệ."})
    img = Image.open(io.BytesIO(raw))
    if img.width * img.height > MAX_PIXELS:
        raise HTTPException(413, detail={"code": "too_many_pixels", "message": "Ảnh quá lớn để xử lý an toàn."})
    return ImageOps.exif_transpose(img).convert("RGB")


def _preprocess(img, pp):
    mean = np.array(pp["mean"], dtype=np.float32); std = np.array(pp["std"], dtype=np.float32)
    img = img.resize((pp["img_size"], pp["img_size"]))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)


def _energy(logits, T=1.0):
    z = logits / T; return float(-T * (z.max() + np.log(np.exp(z - z.max()).sum())))


@app.post("/predict")
async def predict(species: str, file: UploadFile = File(...)):
    """Chẩn đoán 1 ảnh. Lazy load model loài nếu chưa có. Threshold LẤY TỪ bundle (hiện default)."""
    if species not in _sessions:
        try:
            _load_species_model(species)
        except Exception as e:
            raise HTTPException(500, detail=f"Lỗi tải model {species}: {e}")
    img = validate_image_upload(await file.read())
    x = _preprocess(img, _preproc[species])
    T = _cal[species].get("temperature", 1.0)
    thr = _thr[species]
    species_logits = _sessions[species].run(["logits"], {"input": x})[0]

    # (1) Leaf-gate + energy OOD (hiện dùng default vì chưa build thresholds.json — Mục 5.4)
    if _leaf_sess is not None:
        lg = _leaf_sess.run(None, {"input": x})[0]
        p_leaf = float(_softmax(lg / _leaf_T)[0, 1])
        if p_leaf < thr.get("binary_leaf", 0.5):
            return {"status": "rejected", "reason": "not_a_leaf", "p_leaf": round(p_leaf, 4)}
        if _energy(species_logits, T) > thr.get("T_energy", 1e9):
            return {"status": "rejected", "reason": "unknown_or_ood",
                    "energy": round(_energy(species_logits, T), 4)}

    # (2) Chẩn đoán loài (đã calibration)
    probs = _softmax(species_logits / T)[0]
    order = probs.argsort()[::-1]
    classes = _classes[species]
    top1, top2 = probs[order[0]], (probs[order[1]] if len(order) > 1 else 0.0)
    if top1 < thr.get("T_low_confidence", 0.0):
        return {"status": "low_confidence", "confidence": round(float(top1), 4)}
    if (top1 - top2) < thr.get("T_margin", 0.0):
        return {"status": "ambiguous", "candidates": [classes[order[0]], classes[order[1]]],
                "confidence": round(float(top1), 4)}
    return {"status": "ok", "species": species, "prediction": classes[order[0]],
            "confidence": round(float(top1), 4),
            "top3": [{"label": classes[i], "prob": round(float(probs[i]), 4)} for i in order[:3]]}


@app.get("/health")
def health():
    return {"status": "healthy", "species_loaded": list(_sessions.keys())}


@app.get("/ready")
def ready():
    ok = len(_sessions) > 0
    return {"status": "ready" if ok else "not_ready", "species_loaded": list(_sessions.keys())}


# /api/* cho frontend PhytoScan + phục vụ tĩnh (Mục 7.10)
from fastapi.staticfiles import StaticFiles
from src.deploy.api_compat import router as api_router
app.include_router(api_router)
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
```

> Lưu ý `/ready` trả `ready` chỉ khi đã có ít nhất 1 model loài được lazy-load; ngay sau startup (chỉ có leaf-gate) nó báo `not_ready` cho tới request loài đầu tiên.

### 7.6. Bảo mật API & xử lý lỗi

> **Cảnh báo bảo mật (bắt buộc nêu):** endpoint `/predict` **chưa có** authentication hay rate-limit. Với demo học tập chạy local thì chấp nhận được, nhưng **trước khi mở ra Internet công khai** cần tối thiểu: (a) API key hoặc OAuth, (b) rate-limiting (ví dụ `slowapi`), (c) giới hạn kích thước upload, (d) chạy sau HTTPS/reverse-proxy. Không deploy công khai ở trạng thái hiện tại.

Checklist bảo mật upload (đã hiện thực một phần trong `validate_image_upload`, Mục 7.5):

| Kiểm tra | Cách làm | Đã có |
|---|---|---|
| Giới hạn dung lượng | `MAX_BYTES = 8MB` → 413 | ✅ |
| MIME/magic bytes | Chỉ JPEG/PNG hợp lệ → 415 | ✅ |
| Giới hạn pixel | `MAX_PIXELS = 12M` chống decompression bomb → 413 | ✅ |
| Fix EXIF orientation | `ImageOps.exif_transpose` | ✅ |
| Không tin filename gốc | Dùng UUID nếu cần lưu; strip EXIF | Cần khi bật lưu ảnh |
| Decode timeout | Bọc decode có timeout | `[Cần bổ sung khi deploy]` |
| Rate limit / concurrency | `slowapi` + queue length; bận → `503 server_busy` | `[Cần bổ sung khi deploy]` |
| CORS explicit | Không wildcard khi public | `[Cần bổ sung khi deploy]` |
| Lỗi 4xx có cấu trúc | `{code, message}` như trên | ✅ |

`/health` cho liveness; `/ready` xác nhận bundle load đủ, class mapping khớp checksum và golden smoke test qua.

### 7.7. Docker image nhẹ (chỉ ONNX Runtime, không CUDA)

Chìa khoá image nhỏ: base `python:3.11-slim`, chỉ `onnxruntime` (CPU, **không** `onnxruntime-gpu`), **không** cài `torch`.

```dockerfile
# Final_Pipeline/Dockerfile — tầng phục vụ CPU-only (Mục 7.7)
FROM python:3.11-slim
WORKDIR /app

# Chỉ dependency cho INFERENCE — KHÔNG torch, KHÔNG CUDA -> image nhỏ hơn nhiều
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Cần src/ là package: giữ __init__.py các cấp
COPY src/__init__.py ./src/__init__.py
COPY src/models/__init__.py src/models/energy_ood.py ./src/models/
COPY src/deploy/ ./src/deploy/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.deploy.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

> ⚠️ **Cần sửa trong Dockerfile hiện tại:** `EXPOSE`/healthcheck dùng cổng `8000` nhưng `CMD` đang chạy `--port 8080` → không khớp, healthcheck sẽ fail. Thống nhất về một cổng (8000). Ngoài ra bundle **không** còn được `COPY` vào image (model tải từ HF Hub lúc chạy), nên cần truyền `HF_TOKEN` qua env khi `docker run`.

```text
# Final_Pipeline/requirements-serve.txt — TÁCH RIÊNG khỏi requirements train (không torch/CUDA)
fastapi>=0.110
uvicorn[standard]>=0.29
onnxruntime>=1.17          # CPU-only; KHÔNG onnxruntime-gpu
pillow>=10.0
numpy>=1.24
python-multipart>=0.0.9    # cần cho UploadFile
httpx>=0.27               # gọi LLM API ngoài cho /api/consult (Mục 7.10)
huggingface-hub           # tải bundle từ HF Hub (Mục 7.2)
python-dotenv             # đọc .env (LLM_API_KEY, HF_TOKEN)
```

**So sánh dung lượng (ước lượng — tự đo lại):** image training kèm `torch`+CUDA thường **vài GB**; image serving chỉ ONNX Runtime CPU thường **vài trăm MB**. Chênh lệch chủ yếu từ việc bỏ `torch` + thư viện CUDA — đúng lý do xuất ONNX ở Mục 7.3.

```bash
# Build & chạy
docker build -t plant-disease-api:2026-07-12-ab12cd .
# Truyền HF_TOKEN (tải bundle private) + khoá LLM qua env (KHÔNG bake vào image)
docker run -p 8000:8000 \
  -e HF_TOKEN="hf_..." \
  -e LLM_API_KEY="..." \
  -e LLM_BASE_URL="https://api.groq.com/openai/v1" \
  -e LLM_MODEL="llama-3.3-70b-versatile" \
  plant-disease-api:2026-07-12-ab12cd
# Gọi thử
curl -X POST "http://localhost:8000/predict?species=tomato" -F "file=@test_leaf.jpg"
```

### 7.8. UX & disclaimer khi phục vụ

UX hiển thị top-3, confidence đã calibration và margin. Margin thấp → "không phân biệt chắc giữa A và B". Reject → hiển thị lý do (`not_a_leaf` / `unknown_or_ood` / `low_confidence` / `ambiguous`) và hướng dẫn chụp lại. Luôn kèm disclaimer: hệ thống **chỉ hỗ trợ tham khảo**, không thay thế chuyên gia nông nghiệp; không tự ý dùng thuốc bảo vệ thực vật chỉ dựa trên dự đoán; kết quả có thể sai khi ảnh kém, bệnh ngoài danh mục, loài chưa hỗ trợ hoặc triệu chứng phức tạp.

### 7.9. Frontend web (theme PhytoScan)

Frontend là một **landing page tĩnh** (`src/deploy/static/index.html`) theo theme **PhytoScan** — giao diện xanh lá trên nền tối, font Inter, hero + section demo upload. Đây là lớp trình bày mỏng gọi API ONNX ở Mục 7.5; không chứa logic mô hình.

Design tokens (trích từ theme PhytoScan, giữ nguyên để nhất quán thương hiệu):

| Token | Giá trị | Dùng cho |
|---|---|---|
| `--primary` | `#4CAF50` | Nút chính, nhấn mạnh |
| `--primary-light` | `#66BB6A` | Hover |
| `--secondary` | `#00E676` | Điểm nhấn, kết quả, link active |
| `--accent` | `#8BC34A` | Phụ trợ |
| `--bg-dark` | `#050805` | Nền trang |
| `--text-light` | `#E8F5E9` | Chữ chính |
| `--text-muted` | `#A5D6A7` | Chữ phụ |
| Font | `Inter` (400–800) | Toàn trang |
| `--radius` | `20px` | Card, khung upload |

Cấu trúc section (khớp file theme): `hero` (#home) → `demo` (upload + kết quả) → `features` → `how-it-works` → `crops` (Apple/Corn/Grape/Pepper/Tomato) → `pipeline` → `consult` (chatbot tư vấn) → `pricing`. Có sẵn i18n (song ngữ) qua thuộc tính `data-i18n`.

**Lưu ý khớp danh mục loài:** theme gốc liệt kê 4 loài (Apple/Grape/Pepper/Tomato) và không có Corn ở phần crops — khi ghép vào dự án phải **bổ sung Corn** cho đủ 5 loài đã hỗ trợ (Mục 3.3). Loài do `/api/species` trả về động, không hard-code trong HTML.

### 7.10. API compatibility layer cho frontend

Frontend PhytoScan gọi `/api/predict`, `/api/species`, `/api/diseases`, `/api/consult` với response shape riêng, **khác** `/predict` ở Mục 7.5. Router `/api/*` (`api_compat.py`) bọc quanh logic inference đã có, và bổ sung **i18n song ngữ (en/vi)** cho tên loài + tên bệnh, cùng logic **định tuyến chủ đề tư vấn theo từ khoá** (thuốc/phòng ngừa/triệu chứng/lây lan).

```python
# Final_Pipeline/src/deploy/api_compat.py  (trích — cấu trúc thực tế)
"""Router /api/* khớp hợp đồng frontend PhytoScan."""
import os, re, time, httpx
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
router = APIRouter(prefix="/api")

# Cấu hình LLM qua ENV (mặc định Groq — rẻ/nhanh; đổi LLM_BASE_URL để dùng provider khác)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "15"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# SPECIES_I18N / DISEASE_I18N: từ điển en/vi cho dropdown + nhãn hiển thị (đầy đủ 5 loài / 20 lớp)
# translate_species() / translate_disease(): map class_name -> nhãn song ngữ (có fallback keyword)

@router.get("/species")
def species_list(language: str = "en"):
    from src.deploy.app import REPO_MAP
    keys = list(REPO_MAP.keys())
    return {"species": keys,
            "items": [{"value": k, "label": translate_species(k, language)} for k in keys]}

@router.get("/diseases")
def diseases(species: str, language: str = "en"):
    """Trả danh mục bệnh — luôn dùng DISEASE_I18N để dropdown hiển thị ngay (không chờ model load)."""
    keys = list(DISEASE_I18N.get(species, {}).keys()) or ["healthy"]
    return {"species": species, "diseases": keys,
            "items": [{"value": k, "label": translate_disease(species, k, language)} for k in keys]}

@router.post("/predict")
async def api_predict(species: str = Form(...), file: UploadFile = File(...), language: str = Form("en")):
    from src.deploy.app import predict
    t0 = time.time()
    try:
        result = await predict(species=species, file=file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if result.get("status") == "ok":
        return {"status": result["prediction"],
                "predictions": {d["label"]: d["prob"] for d in result["top3"]},
                "confidence_level": _confidence_level(result["confidence"]),
                "latency_ms": round((time.time() - t0) * 1000, 1)}
    return {"status": result.get("status"), "reason": result.get("reason"),
            "predictions": {}, "confidence_level": "Low"}

@router.post("/consult")
async def consult(payload: dict):
    """Tư vấn LLM ngoài. CHỈ gửi TEXT (loài/bệnh/câu hỏi) — không ảnh, không PII.
    Định tuyến chủ đề theo từ khoá (thuốc/phòng ngừa/triệu chứng/lây lan) để trả lời đúng trọng tâm.
    Fallback tĩnh nếu thiếu key / lỗi mạng."""
    species, disease = payload.get("species", ""), payload.get("disease", "")
    message = (payload.get("message", "") or "").strip()
    language = payload.get("language", "vi")
    if not species or not disease:
        return {"reply": _select_species_disease_msg(language)}
    # ... phân loại câu hỏi -> chọn 'focus' cho system prompt ...
    reply = _call_llm(system, user) or _fallback_reply(species, disease, language)
    return {"reply": reply, "disclaimer": _disclaimer_text(language)}
```

Helper gọi LLM (trong `api_compat.py`) — dùng REST HTTP nên **không** thêm SDK nặng vào image serving; đổi provider chỉ cần đổi env var:

```python
# Final_Pipeline/src/deploy/api_compat.py  (helper gọi LLM — nối tiếp)
def _call_llm(system_prompt: str, user_prompt: str):
    """Gọi LLM ngoài (chat completions, OpenAI-compatible). Trả text hoặc None nếu key trống / lỗi.
    An toàn: chỉ text đi ra ngoài; timeout ngắn; nuốt lỗi để không sập API chính."""
    if not LLM_API_KEY:
        return None
    try:
        resp = httpx.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={"model": LLM_MODEL, "max_tokens": LLM_MAX_TOKENS, "temperature": 0.3,
                  "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": user_prompt}]},
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[consult] LLM lỗi ({e}) -> fallback tĩnh")
        return None
```

> Mặc định trong code trỏ tới model `llama-3.3-70b-versatile` (hợp với Groq — nhanh/rẻ); đặt `LLM_BASE_URL=https://api.groq.com/openai/v1` + `LLM_API_KEY` tương ứng. Đổi `LLM_BASE_URL`/`LLM_MODEL` để dùng OpenAI/DeepSeek/OpenRouter/vLLM local. `LLM_MAX_TOKENS` mặc định 2048 — hạ xuống để tiết kiệm chi phí nếu cần.

Router `/api/*` và static mount đã được gắn ở cuối `app.py` (xem Mục 7.5): `app.include_router(api_router)` rồi `app.mount("/", StaticFiles(..., html=True))`. Static mount đặt **sau** mọi route API để không che các endpoint.

> Giữ **hai lớp**: `/predict` (thô, để test/parity/script) và `/api/predict` (cho UI). Cả hai dùng chung một hàm inference nên không lệch logic. Cảnh báo bảo mật Mục 7.6 áp dụng cho **cả hai** — thêm auth/rate-limit trước khi public. `/api/consult` gọi **LLM API ngoài** (chi phí thấp) qua REST, cấu hình bằng env var `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`; nguyên tắc an toàn: **chỉ gửi text** (loài/bệnh/câu hỏi), không bao giờ gửi ảnh hay PII, và fallback về câu trả lời tĩnh khi thiếu key hoặc lỗi mạng.

---


## 8. Versioning, Monitoring & Rollback

Section này biến pipeline thành hệ thống **deploy được, theo dõi được, rollback được** — không đổi kiến trúc mô hình ở Section 2–7.

### 8.1. Taxonomy, class mapping & thêm lớp mới

Taxonomy dùng **ID ổn định** thay vì folder name. `configs/taxonomy_v1.yaml` lưu `class_id`, tên hiển thị, alias và mapping từ nhãn dataset sang nhãn nội bộ.

```yaml
# Final_Pipeline/configs/taxonomy_v1.yaml (trích)
taxonomy_version: taxonomy_v1
classes:
  - class_id: tomato_late_blight
    species: tomato
    display_name: "Tomato — Late blight"
    aliases: ["Tomato Late blight", "Tomato_late_blight", "late_blight"]
  - class_id: tomato_spider_mites
    species: tomato
    display_name: "Tomato — Spider mites (Two-spotted)"
    aliases: ["Spider_mites Two-spotted_spider_mite"]
    note: "Lớp ít nhất của Tomato (55 ảnh) — train như lớp thường, theo dõi recall riêng"
```

Quy tắc thêm lớp:
- **Bệnh mới ít ảnh:** dùng prototype tạm thời nếu đủ điều kiện (Chế độ 2); chưa retrain head.
- **Bệnh mới đủ ảnh:** retrain species head + tạo bundle mới + calibration mới.
- **Loài mới:** config riêng, calibration riêng, mapping riêng, smoke test riêng.
- Mọi thay đổi taxonomy tăng version (`taxonomy_v2`), không sửa in-place `taxonomy_v1`.

### 8.2. Model registry & deployment strategy

| Chiến lược | Khi nào | Làm gì |
|---|---|---|
| **Shadow (offline)** | Luôn, trước mọi deploy | Version cũ + mới cùng chạy golden set; so prediction, confidence, reject reason, latency |
| **Canary** | Chỉ khi phục vụ thật | Định tuyến % nhỏ request sang version mới, theo dõi metric |
| **Active swap** | Sau khi shadow + smoke pass | Đổi `active_version_<species>.txt`, restart |

Với demo hiện tại: bundle được version hoá offline (`build_bundle.py`) rồi **push lên HF Hub**; đổi version = push bundle mới / cập nhật `REPO_MAP` và restart. Shadow offline + smoke test nên chạy trước khi cập nhật repo phục vụ.

```bash
# Đóng gói + version offline, sau đó push lên HF Hub (repo trong REPO_MAP)
python -m src.export.build_bundle --species tomato \
       --version tomato_2026_07_12_effv2s_ab12cd \
       --onnx model_bundles/tmp/tomato/model.onnx --git-commit ab12cd34
# huggingface-cli upload ruhua2305/tomato_effv2s_onnx model_bundles/tomato_2026_07_12_effv2s_ab12cd

# Rollback: trỏ REPO_MAP về commit/revision bundle cũ đã biết tốt rồi restart service.
```

> Registry cục bộ `active_version_<species>.txt` vẫn tồn tại trong repo cho phương án chạy hoàn toàn local; khi đó `app.py` cần chuyển sang đọc bundle cục bộ thay vì HF Hub.

### 8.3. Điều kiện rollback cụ thể

Rollback **không** dựa vào cảm giác.

| Môi trường | Trigger rollback |
|---|---|
| Demo/local | smoke test fail; parity report fail; API không load bundle; class mapping checksum mismatch; p95 latency golden benchmark vượt ngưỡng |
| Triển khai thật | error rate 5xx vượt ngưỡng; p95/p99 latency tăng mạnh; reject rate bất thường; confidence distribution lệch nhiều so baseline; model trả một lớp áp đảo bất thường; health/readiness fail; memory vượt ngưỡng; canary metrics thấp hơn bản cũ |

Rollback phải quay lại **cả gói**: model bundle, preprocessing, mapping, calibration, prototype, taxonomy và Docker image/API code tương ứng. **Không** rollback riêng file ONNX.

### 8.4. Rollback code, dependency & Docker image

Docker image tag bất biến, ví dụ `plant-disease-api:2026-07-04-ab12cd`. Giữ ít nhất **2 image + 2 bundle** gần nhất. `deploy/deploy.sh` nhận `IMAGE_TAG` + `ACTIVE_MODEL_VERSION`; `deploy/rollback.sh` đổi về cặp image+bundle đã biết tốt. Serving image chỉ chứa inference deps; training deps ở image/environment riêng.

### 8.5. Rollback dữ liệu & training set

Dataset **không** sửa in-place. Mỗi lần train tham chiếu một `dataset_version` + manifest cố định (Mục 2.5). Dữ liệu mới (kể cả từ feedback loop) vào vùng `candidate/`, sau review mới sang `approved/` + tạo manifest version mới. Batch dữ liệu gây lỗi → loại manifest version đó khỏi registry, retrain từ manifest trước.

### 8.6. Monitoring inference & drift detection

Mỗi request log **JSON một dòng**, không lưu ảnh mặc định. Trường tối thiểu: `timestamp`, `request_id`, `model_version`, `species`, `predicted_class`, `top1_confidence`, `top1_top2_margin`, `ood_score`, `reject_reason`, `image_quality_score`, `latency_ms`, `input_bytes`, `status_code`, `error_code`. Nếu lưu ảnh để debug/active learning, phải có consent + strip EXIF.

```python
# Final_Pipeline/src/deploy/logging_utils.py
import json, time, uuid


def log_prediction(logger_file, species, model_version, result, latency_ms, input_bytes):
    """Ghi 1 dòng JSON để monitor + drift. KHÔNG ghi ảnh."""
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
```

Drift theo dõi qua: phân phối embedding, confidence, reject rate, class distribution, tỷ lệ low-confidence, tỷ lệ feedback sai, latency. **Cảnh báo drift không tự động retrain** — chỉ tạo danh sách mẫu cần review.

### 8.7. Feedback loop & active learning

UI thêm nút "đúng", "sai", "không chắc", và cho nhập nhãn đề xuất nếu người dùng biết. Mẫu ưu tiên review khi: confidence thấp, OOD cao, margin top1/top2 thấp, regular-head và prototype-head bất đồng (Mục 3.5), thuộc lớp hiếm, hoặc nằm trong drift cluster mới. **Feedback không đi thẳng vào training** — phải qua review để tránh sai nhãn hoặc poisoning.

### 8.8. Training resilience

Checkpoint định kỳ lưu đủ để resume (Mục 4.10). Script train có `--resume`. Gặp OOM → thử giảm batch size theo policy (Mục 4.5 ladder) hoặc dừng với hướng dẫn rõ ràng; không để crash mất toàn bộ tiến trình.

### 8.9. Reproducibility & experiment naming

Mỗi experiment có ID: `YYYY-MM-DD_<species>_<arch>_<key-feature>_seed<seed>_<config-hash>`, ví dụ `2026-07-04_tomato_effv2s_domainaug_seed42_ab12cd`. Thư mục `experiments/<experiment_id>/` lưu: config copy, metrics, confusion matrix, calibration plot, reliability diagram, parity report, Grad-CAM samples, manifest. Lưu thêm `requirements.txt`, `pip freeze`, phiên bản PyTorch/ONNX Runtime/CUDA/cuDNN, seed, worker seed, class list cố định.

### 8.10. Test tự động bắt buộc (pytest)

Dự án nhỏ vẫn cần pytest cho các lỗi âm thầm:

| Test | Bắt lỗi gì |
|---|---|
| `test_class_mapping.py` | Class mapping ổn định giữa train/bundle/API |
| `test_preprocessing_parity.py` | Preprocessing train vs API khớp |
| `test_onnx_parity.py` | PyTorch vs ONNX top-1 + logit diff |
| `test_api_upload_security.py` | Oversize/MIME/magic-byte/pixel-bomb bị chặn |
| `test_calibration_loading.py` | Temperature + thresholds load đúng từ bundle |
| `test_prototype_inference.py` | Nhánh few-shot chạy + shape đúng |
| `test_bundle_checksum.py` | checksums.sha256 khớp file |
| `test_bundle_rollback.py` | Load được version cũ |
| `test_readiness.py` | `/ready` phản ánh đúng trạng thái load |

```bash
pytest src/tests/test_class_mapping.py
pytest src/tests/test_preprocessing_parity.py
pytest src/tests/test_onnx_parity.py
pytest src/tests/test_api_upload_security.py
pytest src/tests/test_bundle_rollback.py
```

### 8.11. Security & resource control

API giới hạn file upload, pixel count, MIME/magic bytes, decode timeout, concurrency, queue length (Mục 7.6). Model load một lần lúc startup. Server bận → `503 server_busy` có kiểm soát. CORS explicit, không wildcard khi public. Rate limit theo IP/API key nếu mở ngoài local. Không lưu filename gốc; cần lưu file thì dùng UUID.

### 8.12. Robustness, fairness & giới hạn báo cáo

Báo cáo có benchmark robustness (Mục 5.6): blur, noise, low-light, overexposure, compression, rotation, occlusion, lá diện tích nhỏ. "Fairness" ở bài toán này nghĩa là hiệu năng không lệch quá mạnh theo species, class, nguồn ảnh, thiết bị, ánh sáng, nền, nhóm rare-vs-regular. **Thiếu metadata thì nêu giới hạn, không suy diễn.**

### 8.13. Checklist deploy cuối cùng

Trước khi đổi `active_version_<species>.txt`, phải đạt **tất cả**:

- [ ] Bundle manifest đủ file + checksum khớp (`test_bundle_checksum.py`).
- [ ] Class mapping golden test khớp PyTorch / ONNX / API.
- [ ] Calibration đã chạy cho head tương ứng + bản ONNX (ECE báo cáo).
- [ ] Parity test đạt ngưỡng (`macro_f1_drop<=0.01`, top-1 match).
- [ ] API security test pass (oversize/MIME/magic/pixel).
- [ ] `/health` và `/ready` pass.
- [ ] Latency p50/p95/p99 đạt mục tiêu (local p95 <= 3s).
- [ ] Shadow evaluation không phát hiện lệch bất thường.
- [ ] Docker image + previous bundle sẵn sàng rollback (>= 2 bản).
- [ ] Holdout report đã sinh **một lần** và đóng băng (Mục 5.2).

---

## 9. CLI tổng hợp

Toàn bộ lệnh chạy **theo đúng thứ tự** từ root `D:\DAT301m_Project\Final_Pipeline`. Mỗi lệnh dùng `python -m` để `src/` là package hợp lệ. **Chú ý: đúng 5 loài** apple / corn / grape / pepper_bell / tomato (không có potato).

```bash
# ==========================================
# PHASE 0 — Xây split manifest (chạy 1 LẦN)
# ==========================================
# Mỗi loài 1 manifest JSON (Mục 2.5) chứa train/calib/val/holdout + rare_folds + hash.
# Nếu dữ liệu không đổi, KHÔNG chạy lại (giữ reproducibility).
python -m src.data.split_builder --species all --seed 42

# ==========================================
# PHASE 1 — Pretrain backbone 38 lớp + train Leaf Gate
# ==========================================
# Sinh checkpoints/backbone_pretrained.pt — điểm khởi tạo CHUNG cho 5 lần fine-tune.
# Ước lượng: ~3-5h trên RTX 3050 Ti (30 epoch, AMP, batch 16, accum 4, EffV2-S).
python -m src.train.train_pretrain --config configs/pretrain_effv2s.yaml
# Leaf Gate nhị phân (độc lập, ~15-25 phút) — cần data/leaf_gate/{train,calib,val}/{leaf,not_leaf}
python -m src.train.train_leaf_gate --config configs/leaf_gate.yaml

# ==========================================
# PHASE 2 — Fine-tune 5 loài (tuần tự — an toàn VRAM)
# ==========================================
# Mỗi loài sinh model_<species>.pt (Chế độ 1). Chế độ 2/prototype KHÔNG chạy (không lớp <30).
python -m src.train.train_finetune --config configs/finetune_apple.yaml
python -m src.train.train_finetune --config configs/finetune_corn.yaml
python -m src.train.train_finetune --config configs/finetune_grape.yaml
python -m src.train.train_finetune --config configs/finetune_pepper_bell.yaml
python -m src.train.train_finetune --config configs/finetune_tomato.yaml

# ==========================================
# PHASE 3 — Đánh giá + calibration
# ==========================================
# Báo cáo chi tiết trên val (classification report + confusion matrix).
python -m src.eval.evaluate --config configs/finetune_apple.yaml --split val
python -m src.eval.evaluate --config configs/finetune_corn.yaml --split val
python -m src.eval.evaluate --config configs/finetune_grape.yaml --split val
python -m src.eval.evaluate --config configs/finetune_pepper_bell.yaml --split val
python -m src.eval.evaluate --config configs/finetune_tomato.yaml --split val
# Calibration (temperature scaling) trên calib split — sinh <ckpt>_calibration.json.
python -m src.eval.calibration --config configs/finetune_apple.yaml --target species
python -m src.eval.calibration --config configs/finetune_corn.yaml --target species
python -m src.eval.calibration --config configs/finetune_grape.yaml --target species
python -m src.eval.calibration --config configs/finetune_pepper_bell.yaml --target species
python -m src.eval.calibration --config configs/finetune_tomato.yaml --target species
python -m src.eval.calibration --config configs/leaf_gate.yaml --target leaf_gate

# ==========================================
# PHASE 4 — Export ONNX + parity + đóng gói bundle
# ==========================================
# 4.1 Export ONNX (lặp cho 5 loài + leaf_gate)
python -m src.export.export_onnx --config configs/finetune_tomato.yaml \
       --out model_bundles/tmp/tomato/model.onnx --quantize int8
# 4.2 Parity test PyTorch vs ONNX trên golden set (bắt buộc, Mục 7.4)
python -m src.export.parity_test --bundle model_bundles/tmp/tomato/ \
       --pt-ckpt checkpoints/model_tomato.pt --golden-set data/golden/tomato.json
# 4.3 Đóng gói bundle bất biến + checksums (Mục 7.4b)
python -m src.export.build_bundle --species tomato \
       --version tomato_2026_07_04_effv2s_ab12cd \
       --onnx model_bundles/tmp/tomato/model.onnx --git-commit ab12cd34
# (lặp 4.1-4.3 cho 5 loài; leaf_gate export vào model_bundles/leaf_gate/)

# ==========================================
# PHASE 4b — Đánh giá HOLDOUT (chỉ CHẠY MỘT LẦN, sau khi chốt model)
# ==========================================
python -m src.eval.evaluate --config configs/finetune_apple.yaml --split holdout
# (lặp cho 5 loài — KHÔNG tuning gì sau bước này)

# ==========================================
# PHASE 5 — Serving (API + frontend PhytoScan)
# ==========================================
# Bundle tải từ HF Hub (Mục 7.2) — cần HF_TOKEN cho private repo; LLM key cho /api/consult.
export HF_TOKEN=hf_...          # (Windows: set HF_TOKEN=... hoặc dùng .env)
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
# Frontend PhytoScan phục vụ tĩnh tại "/" (Mục 7.9); API UI tại /api/* (Mục 7.10).
# Hoặc Docker: docker build -t plant-disease-api:2026-07-12-ab12cd . && docker run -p 8000:8000 -e HF_TOKEN=... ...

# ==========================================
# Test tự động (chạy trước deploy — Mục 8.10)
# ==========================================
pytest src/tests/
```

**Lưu ý thực hành:**
- Luôn để `probe_vram_usage` chạy đầu tiên (đã tích hợp trong cả 2 script train) — OOM thì script tự dừng với thông báo rõ ràng thay vì crash giữa chừng.
- Theo dõi realtime: `nvidia-smi -l 2` ở terminal thứ hai.
- Chạy tuần tự 5 loài, khuyến nghị chạy qua đêm.

---

## 10. Timeline & ước lượng thời gian

Ước lượng trên RTX 3050 Ti 4GB, batch 16, accum 4, EffV2-S, AMP:

| Phase | Công việc | Thời gian ước lượng | Ghi chú |
|---|---|---|---|
| 0 | Split builder (hash toàn bộ ảnh) | ~10–20 phút | Chạy 1 lần |
| 1 | Pretrain 38 lớp (30 epoch) | ~3–5 giờ | Backbone chung |
| 1 | Train Leaf Gate (MobileNetV3-Small) | ~15–25 phút | Chạy 1 lần, độc lập |
| 2 | Fine-tune Apple (data lớn) | ~1.5–2 giờ | 3 bước |
| 2 | Fine-tune Corn | ~40–60 phút | |
| 2 | Fine-tune Grape | ~15–25 phút | data nhỏ, freeze backbone |
| 2 | Fine-tune Pepper | ~15–25 phút | data nhỏ |
| 2 | Fine-tune Tomato (9 lớp) | ~1–1.5 giờ | 9 lớp, 3 bước decoupled |
| 3 | Đánh giá + calibration 5 loài | ~30–45 phút | |
| 4 | Export ONNX + parity + bundle | ~30–45 phút | |
| 5 | Dựng API + Docker + smoke test | ~1–2 giờ | Một lần |
| **Tổng** | **Train + deploy đầu-cuối** | **~9–13 giờ GPU** | Khuyến nghị chạy qua đêm |

Thứ tự khuyến nghị: Phase 0 → 1 → 2 (tuần tự 5 loài) → 3 → 4 → 4b (holdout) → 5. Trước mỗi lần train, rà **Memory Safety Checklist** (Appendix D) — hai phút này tiết kiệm hàng giờ debug OOM.

---

## 11. Giới hạn & lưu ý còn để ngỏ

Những điểm cần quyết định/bổ sung khi triển khai thật:

- **Thresholds reject chưa build (ưu tiên cao nhất):** chưa có `thresholds.json` nên energy-OOD + ambiguous/low-confidence chưa thực sự hoạt động (Mục 5.4). Cần quét `calib` để chọn `binary_leaf`, `T_energy`, `T_low_confidence`, `T_margin` rồi đóng vào bundle/HF repo.
- **Calibration chưa đạt ECE ≤ 0.05:** hầu hết loài và đặc biệt leaf-gate (ECE 0.74) còn cao (Mục 5.3). Cần xem lại calib split / nhãn leaf-gate; trong lúc đó đặt threshold bảo thủ.
- **Xác minh citations:** đã đối chiếu online — sửa 1 lỗi ID (FMDA, Mục 12 #17: `arXiv:2203.12119` → `arXiv:2412.18859`) và cập nhật tiêu đề thật cho P6/P7. Các mục còn lại đã xác nhận đúng.
- **Golden set tối giản:** hiện chỉ là danh sách ảnh cho parity; cần bổ sung `expected_top1`/`min_confidence` để làm smoke test class-mapping đầy đủ (Mục 2.7).
- **`source_group_id` = `[Cần verify]`:** split chưa group theo nguồn/photographer vì thiếu metadata (Mục 2.4) — ghi rõ giới hạn thay vì tuyên bố độc lập nguồn nghiêm ngặt.
- **Cross-test Chiều 2 đã chạy** (Mục 6.5); **Chiều 1 (PlantDoc) chưa chạy:** cần gắn dataset thật + bảng ánh xạ nhãn giao + de-dup hash (FINETUNE đã chứa 742 ảnh PlantDoc) trước khi báo cáo con số.
- **Domain adaptation (field-bg recompose, MMD) chưa bật:** code có nhưng mặc định tắt; nếu bật cần QC Grad-CAM (Mục 4.6).
- **Few-shot/prototype không kích hoạt:** dữ liệu không còn lớp < 30 (Spider_mites nay 55). Giữ làm hướng mở rộng (Mục 3.5).
- **Đo lại sau quantization:** nếu dùng INT8 (Mục 7.3) phải chạy lại `evaluate` + `parity_test` trên model đã lượng tử hoá.
- **Bảo mật API:** `/predict` + `/api/*` chưa có auth/rate-limit (Mục 7.6). `HF_TOKEN` và LLM key phải qua env/`.env`, không commit.
- **Dockerfile lệch cổng:** `CMD --port 8080` vs `EXPOSE 8000`/healthcheck 8000 — cần thống nhất (Mục 7.7).
- **Latency chưa đo thực:** mục tiêu p95 <= 3s là giả định; lazy-load lần đầu mỗi loài có độ trễ tải bundle từ HF Hub — cần benchmark thực.
- **Resume training chưa dùng:** `checkpoints.py` có helper nhưng train script chưa gọi `--resume` (Mục 8.8).

---

## 12. Tài liệu tham khảo

> **Trạng thái xác minh:** toàn bộ 27 citation đã được đối chiếu online (2026-07-12). 26/27 chính xác; đã sửa lỗi ID duy nhất ở #17 (FMDA: `2203.12119` → `2412.18859`) và cập nhật tiêu đề thật cho #15 (Jeon), #16 (Wu). Một số mục dùng tiêu đề rút gọn (bỏ subtitle) — chấp nhận được.

**Kiến trúc & backbone:**
1. Tan, M., & Le, Q. (2021). *EfficientNetV2: Smaller Models and Faster Training.* [arXiv:2104.00298](https://arxiv.org/abs/2104.00298) — Mục 3.1 (backbone chính)
2. Howard, A., et al. (2019). *Searching for MobileNetV3.* [arXiv:1905.02244](https://arxiv.org/abs/1905.02244) — Mục 3.2 leaf-gate; Section 7
3. Mehta, S., & Rastegari, M. (2021). *MobileViT.* [arXiv:2110.02178](https://arxiv.org/abs/2110.02178) — Mục 3.1 hybrid
4. Oquab, M., et al. (2023). *DINOv2.* [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) — Mục 3.5 embedding lớp hiếm
5. Caron, M., et al. (2021). *Emerging Properties in Self-Supervised ViT (DINO).* [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) — Mục 3.1 nền tảng SSL

**Augmentation & regularization:**
6. Yun, S., et al. (2019). *CutMix.* [arXiv:1905.04899](https://arxiv.org/abs/1905.04899) — Mục 4.3
7. Zhang, H., et al. (2017). *mixup.* [arXiv:1710.09412](https://arxiv.org/abs/1710.09412) — Mục 4.3
8. Cubuk, E. D., et al. (2019). *RandAugment.* [arXiv:1909.13719](https://arxiv.org/abs/1909.13719) — Mục 4.3
9. Müller, R., et al. (2019). *When Does Label Smoothing Help?* [arXiv:1906.02629](https://arxiv.org/abs/1906.02629) — Mục 4.3–4.4

**Mất cân bằng & few-shot:**
10. Cui, Y., et al. (2019). *Class-Balanced Loss.* [arXiv:1901.05555](https://arxiv.org/abs/1901.05555) — Mục 4.4
11. Lin, T.-Y., et al. (2017). *Focal Loss.* [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) — Mục 4.4
12. Snell, J., et al. (2017). *Prototypical Networks.* [arXiv:1703.05175](https://arxiv.org/abs/1703.05175) — Mục 3.5, Chế độ 2
13. Kang, B., et al. (2019). *Decoupling Representation and Classifier.* [arXiv:1910.09217](https://arxiv.org/abs/1910.09217) — Mục 4.9 Bước 3

**PEFT & fine-tuning:**
14. Hu, E. J., et al. (2021). *LoRA.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — Mục 4.7

**Domain adaptation (lab → field):**
15. Jeon, W., Kim, T., Choi, S., & Yang, K. (2025). *Bridging the Lab-to-Field Gap in Plant Disease Diagnosis Through Unsupervised Domain Adaptation Enhanced by Background Recomposition.* [SSRN:10.2139/ssrn.5217081](https://dx.doi.org/10.2139/ssrn.5217081) — Mục 4.6; P7
16. Wu, X., et al. (2023). *From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease Recognition in the Wild.* Plant Phenomics. [DOI:10.34133/plantphenomics.0038](https://doi.org/10.34133/plantphenomics.0038) — Mục 4.6; P6
17. Kudo, S., Kagiwada, S., & Iyatomi, H. (2024). *Few-shot Metric Domain Adaptation: Practical Learning Strategies for an Automated Plant Disease Diagnosis.* [arXiv:2412.18859](https://arxiv.org/abs/2412.18859) — Mục 4.6 Cách 3

**OOD & calibration:**
18. Liu, W., et al. (2020). *Energy-based OOD Detection.* [arXiv:2010.03759](https://arxiv.org/abs/2010.03759) — Mục 3.2
19. Guo, C., et al. (2017). *On Calibration of Modern Neural Networks.* [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) — Mục 5.3
20. Naeini, M. P., et al. (2015). *Obtaining Well Calibrated Probabilities (ECE).* [AAAI 2015](https://ojs.aaai.org/index.php/AAAI/article/view/9602) — Mục 5.3

**Tiết kiệm bộ nhớ & nén:**
21. Micikevicius, P., et al. (2017). *Mixed Precision Training.* [arXiv:1710.03740](https://arxiv.org/abs/1710.03740) — Mục 4.5
22. Chen, T., et al. (2016). *Training Deep Nets with Sublinear Memory Cost.* [arXiv:1604.06174](https://arxiv.org/abs/1604.06174) — Mục 4.5
23. Jacob, B., et al. (2017). *Quantization and Training for Integer-Arithmetic-Only Inference.* [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) — Mục 7.3
24. Hinton, G., et al. (2015). *Distilling the Knowledge in a Neural Network.* [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) — tham chiếu nén model

**Dataset & benchmark (Section 6):**
25. Hughes & Salathé (2015). *PlantVillage.* [arXiv:1511.08060](https://arxiv.org/abs/1511.08060) — P1
26. Mohanty, et al. (2016). *Deep Learning for Plant Disease Detection.* [DOI:10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419) — P2
27. Singh, et al. (2020). *PlantDoc.* [arXiv:1911.10317](https://arxiv.org/abs/1911.10317) — P3
28. Borhani, et al. (2022). *ViT for plant disease.* [DOI:10.1038/s41598-022-15163-0](https://doi.org/10.1038/s41598-022-15163-0) — P4
29. Li, et al. (2023). *PMVT.* [DOI:10.3389/fpls.2023.1256773](https://doi.org/10.3389/fpls.2023.1256773) — P5; Mục 3.1

---

## 13. Phụ lục

### Appendix A — Taxonomy đầy đủ (20 lớp / 5 loài)

| species | class_index | class_name | class_id nội bộ | Chế độ |
|---|---:|---|---|---|
| apple | 0 | Apple_scab | apple_scab | 1 |
| apple | 1 | Cedar_apple_rust | apple_cedar_rust | 1 |
| apple | 2 | healthy | apple_healthy | 1 |
| corn | 0 | Cercospora | corn_cercospora | 1 |
| corn | 1 | Common_rust | corn_common_rust | 1 |
| corn | 2 | Northern_Blight | corn_northern_blight | 1 |
| corn | 3 | healthy | corn_healthy | 1 |
| grape | 0 | Black_rot | grape_black_rot | 1 |
| grape | 1 | healthy | grape_healthy | 1 |
| pepper_bell | 0 | Bacterial_spot | pepper_bacterial_spot | 1 |
| pepper_bell | 1 | healthy | pepper_healthy | 1 |
| tomato | 0 | Bacterial_spot | tomato_bacterial_spot | 1 |
| tomato | 1 | Early_blight | tomato_early_blight | 1 |
| tomato | 2 | Late_blight | tomato_late_blight | 1 |
| tomato | 3 | Leaf_Mold | tomato_leaf_mold | 1 |
| tomato | 4 | Septoria | tomato_septoria | 1 |
| tomato | 5 | Spider_mites | tomato_spider_mites | 1 (55 ảnh — lớp ít nhất) |
| tomato | 6 | TYLCV | tomato_tylcv | 1 |
| tomato | 7 | mosaic_virus | tomato_mosaic_virus | 1 |
| tomato | 8 | healthy | tomato_healthy | 1 |

> class_index là thứ tự bất biến do `split_builder.py` sinh (sort tên lớp). API/ONNX/bundle phải dùng đúng thứ tự này (Mục 7.1).

### Appendix B — Bảng probe VRAM (điền sau khi chạy trên máy)

| arch | batch | AMP | accum | peak allocated | peak reserved | Kết luận |
|---|---:|---|---:|---|---|---|
| EffV2-S | 8 | on | 8 | *(điền)* | *(điền)* | *(feasible/OOM)* |
| EffV2-S | 16 | on | 4 | *(điền)* | *(điền)* | mục tiêu mặc định |
| EffV2-S | 32 | on | 2 | *(điền)* | *(điền)* | có thể OOM |
| EffV2-B0 | 16 | on | 4 | *(điền)* | *(điền)* | fallback |
| EffV2-B0 | 32 | on | 2 | *(điền)* | *(điền)* | fallback |

Quy tắc loại: reserved `> 3.6GB` ở dry-run 20 step → loại khỏi production, chuyển fallback.

### Appendix C — Deployment gates (tổng hợp)

| Gate | Ngưỡng | Chặn gì nếu fail |
|---|---|---|
| ONNX parity | top-1 match 100%, `max_abs_diff < 1e-3` | Release bundle |
| INT8 parity | `macro_f1_drop<=0.01`, `rare_recall_drop<=0.03`, `ECE_delta<=0.02` | Deploy INT8 |
| Calibration | ECE `<=0.05` mỗi loài (nếu được) | Nới → threshold reject chặt hơn |
| Golden smoke | 100% top-1 canonical + mapping khớp | Deploy |
| API security | oversize/MIME/magic/pixel test pass | Demo |
| Latency | local p95 `<= 3s` | Fallback backbone nhỏ / preload |
| Holdout | Apple/Tomato F1>=0.90, Corn>=0.80, Grape/Pepper>=0.70 | Tuyên bố demo cuối |

### Appendix D — Memory Safety Checklist (rà trước MỌI lần train)

| # | Kiểm tra | Rủi ro | Xác nhận |
|---|---|---|---|
| C1 | `use_amp: true` trong config | OOM VRAM | Mở YAML |
| C2 | `batch_size × accumulation_steps <= 64` | OOM VRAM | Tính: 16×4=64 ✓ |
| C3 | `num_workers: 2` (không >4 trên 16GB RAM) | OOM RAM | Mở YAML |
| C4 | `persistent_workers: false` | OOM RAM | Xem DataLoader |
| C5 | `probe_vram_usage` chạy ĐẦU TIÊN | OOM VRAM | Tích hợp sẵn 2 script train |
| C6 | Val KHÔNG augmentation (CutMix/Mixup/domain) | Rò rỉ | `get_*_transforms(train=False)` |
| C7 | Split chạy TRƯỚC augmentation; manifest cố định | Rò rỉ | `split_builder.py` chạy 1 lần |
| C8 | Seed cố định `42` | Không tái lập | Config `seed: 42` |
| C9 | Metric chọn checkpoint = `macro_f1` | Bỏ rơi minority | Config `metric: macro_f1` |
| C10 | Holdout KHÔNG động tới khi chưa chốt model | Rò rỉ holdout | Chỉ Phase 4b mở holdout |

### Appendix E — requirements (training)

```text
# Final_Pipeline/requirements.txt — môi trường TRAIN (tách khỏi requirements-serve.txt)
torch>=2.2            # + CUDA build khớp driver máy
timm>=0.9
albumentations>=1.4
opencv-python>=4.9
scikit-learn>=1.4
onnx>=1.16
onnxruntime>=1.17
onnxconverter-common>=1.14   # cho convert FP16
pyyaml>=6.0
numpy>=1.24
pillow>=10.0
pytest>=8.0
```

---

## 14. Trạng thái tài liệu

Tài liệu đã đồng bộ với code thực tế (rà 2026-07-12). Trạng thái từng section (xem thêm bảng chi tiết Mục 1.8):

| Section | Nội dung | Trạng thái |
|---|---|---|
| 1 | Executive Summary + quyết định + thay đổi + bảng trạng thái 1.8 | ✅ |
| 2 | Dataset Analysis + split 70/10/10/10 + manifest lock (18,402 ảnh) | ✅ có code |
| 3 | Model Architecture (EffV2-S, leaf-gate; few-shot là mở rộng) | ✅ có code |
| 4 | Training Pipeline (pretrain + fine-tune 5 loài 3 bước + leaf-gate) | ✅ code chạy được |
| 5 | Evaluation & Calibration | ✅ code; ⚠️ ECE chưa đạt, thresholds chưa build |
| 6 | Cross-Test Plan | ✅ Chiều 2 đã chạy (Mục 6.5); Chiều 1 tạm gác |
| 7 | Deployment (ONNX + parity + FastAPI + HF Hub + PhytoScan) | ✅ code chạy được |
| 8 | Versioning, Monitoring & Rollback | ✅ có code |
| 9–14 | CLI, Timeline, Giới hạn, References, Phụ lục | ✅ |

Đã triển khai thực tế: split manifest 5 loài + hash, pretrain backbone, fine-tune 5 loài (PEFT 3 bước + CB-Focal + decoupled), leaf-gate, temperature scaling, ONNX export + parity, bundle bất biến, FastAPI + PhytoScan frontend + /api/consult (LLM), serving qua Hugging Face Hub.

Việc còn lại (Mục 1.8, 11): build `thresholds.json` để kích hoạt reject policy; cải thiện calibration đạt ECE ≤ 0.05 (nhất là leaf-gate); hoàn thiện golden set + cross-test **Chiều 1** (PlantDoc, cần de-dup hash); sửa cổng Dockerfile (8080↔8000); tùy chọn bật domain adaptation. Cross-test **Chiều 2 đã chạy** (Mục 6.5) — xác nhận domain gap lab→field. Few-shot/prototype giữ làm hướng mở rộng vì dữ liệu hiện không còn lớp < 30. Backbone chốt EfficientNetV2-S (fallback B0), tối ưu cho RTX 3050 Ti 4GB / RAM 16GB.
