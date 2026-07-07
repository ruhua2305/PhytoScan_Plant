# PhytoScan — DAT301m Plant Disease Classification

Hệ thống phân loại **bệnh cây trồng qua ảnh lá** cho **5 loài**: `apple`, `corn`, `grape`,
`pepper_bell`, `tomato` (tổng **20 lớp bệnh/khỏe**). Sản phẩm cuối là **web demo chạy local**
(FastAPI + frontend PhytoScan): người dùng chọn loài, upload ảnh lá, hệ thống kiểm tra ảnh hợp lệ
bằng **Leaf Gate**, chẩn đoán bệnh kèm **độ tin cậy đã hiệu chỉnh**, và biết **từ chối** khi ảnh
không phải lá / sai loài / ngoài phân phối / độ tin cậy thấp.

Toàn bộ thiết kế chi tiết nằm trong [`DAT301m_Plan_Final.md`](DAT301m_Plan_Final.md). README này
tập trung vào **cách tổ chức dữ liệu và cách chạy**.

## Đặc điểm chính

| Hạng mục | Giá trị |
|---|---|
| Phần cứng đích | RTX 3050 Ti Laptop **4GB VRAM** / **16GB RAM** / Windows |
| Backbone | EfficientNetV2-S (primary), B0 (fallback) |
| Định tuyến | Per-species: 5 model riêng, người dùng chọn loài |
| Chia dữ liệu | 70/10/10/10 = train/calib/val/holdout, manifest khoá + SHA256 |
| Lớp hiếm | `Tomato/Spider_mites` (7 ảnh) → few-shot prototype, không train head riêng |
| Chống 4GB OOM | AMP + gradient accumulation + PEFT + `probe_vram_usage()` |
| Phục vụ | ONNX Runtime (CPU-only), bundle bất biến + rollback |
| Tư vấn | `/api/consult` gọi LLM ngoài (chỉ gửi text, cấu hình qua ENV) |

## Mục lục

1. [Cài đặt môi trường](#1-cài-đặt-môi-trường)
2. [Tổ chức thư mục dataset](#2-tổ-chức-thư-mục-dataset)
3. [Dataset Leaf Gate](#3-dataset-leaf-gate)
4. [Quy trình chạy đầu-cuối](#4-quy-trình-chạy-đầu-cuối)
5. [Serving, LLM & frontend](#5-serving-llm--frontend)
6. [Test & xử lý sự cố](#6-test--xử-lý-sự-cố)

---

## 1. Cài đặt môi trường

Yêu cầu: **Python 3.10+**, GPU NVIDIA + CUDA driver để train (serving thì chỉ cần CPU).

```bash
# 1) Tạo virtualenv (khuyến nghị) — chạy từ root D:\DAT301m_Project\Final_Pipeline
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

# 2a) Môi trường TRAIN (máy có GPU)
pip install -r requirements.txt
#   LƯU Ý torch+CUDA: cài bản khớp driver, ví dụ CUDA 12.1:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2b) Môi trường SERVING (CPU-only, image nhẹ — có thể tách máy khác)
pip install -r requirements-serve.txt
```

Kiểm tra nhanh GPU + phiên bản (nên chạy trước khi train):

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

- Nếu `CUDA: False` → kiểm tra lại driver / bản torch. Train trên CPU sẽ **rất chậm**, không khuyến nghị.
- `requirements.txt` (train) và `requirements-serve.txt` (serve) **tách biệt** có chủ đích: image phục vụ không cài `torch`/CUDA nên nhẹ hơn nhiều lần.

---

## 2. Tổ chức thư mục dataset

### 2.1. Dữ liệu NGUỒN (bạn chuẩn bị — nằm TRONG `Final_Pipeline`)

Dữ liệu gốc đặt tại `D:\DAT301m_Project\Final_Pipeline\data\`. Code chỉ đọc, không sửa. Cấu trúc bắt buộc
là `<dataset>/<species>/<class_name>/*.jpg` — mỗi lớp một thư mục con:

```
D:\DAT301m_Project\Final_Pipeline\data\
├── 1. PRETRAIN DATASET\          # ~53.599 ảnh, 38 lớp — ảnh lab nền sạch (chỉ để pretrain)
│   ├── train\
│   │   ├── Apple___Apple_scab\ *.jpg
│   │   ├── Tomato___healthy\   *.jpg
│   │   └── ... (38 lớp)
│   └── val\
│       └── ... (cùng 38 lớp)
│
├── 2. FINETUNE DATASET\          # ~18.354 ảnh, 20 lớp, 5 loài — ảnh FIELD (miền đích thật)
│   ├── apple\
│   │   ├── Apple_scab\        *.jpg
│   │   ├── Cedar_apple_rust\  *.jpg
│   │   └── healthy\           *.jpg
│   ├── corn\
│   │   ├── Cercospora\  Common_rust\  Northern_Blight\  healthy\
│   ├── grape\
│   │   ├── Black_rot\  healthy\
│   ├── pepper_bell\
│   │   ├── Bacterial_spot\  healthy\
    └── tomato\
        ├── Bacterial_spot\ Early_blight\ Late_blight\ Leaf_Mold\ Septoria\
        ├── Spider_mites\  TYLCV\  mosaic_virus\  healthy\
```

> Tên thư mục loài thật trên đĩa có thể khác key nội bộ (`Corn_(maize)`↔`corn`,
> `Pepper,_bell`↔`pepper_bell`); `split_builder.py` đã tự ánh xạ qua `SPECIES_DIR_NAME`.

> **Tên lớp** phải khớp `configs/taxonomy_v1.yaml` (cột `class_name`). `class_index` do
> `split_builder.py` sinh tự động bằng cách **sort tên lớp** — bất biến, dùng chung cho ONNX/API.
> Nếu tên thư mục lớp khác taxonomy, sửa taxonomy hoặc đổi tên thư mục cho trùng.

### 2.2. Dữ liệu SINH RA (code tạo — nằm TRONG `Final_Pipeline`)

Bạn không tạo tay các thư mục này; pipeline sẽ ghi vào:

```
D:\DAT301m_Project\Final_Pipeline\
├── data\
│   ├── splits\                   # split_builder.py -> manifest 70/10/10/10 + SHA256
│   │   ├── apple_split_v1.json   corn_split_v1.json   grape_split_v1.json
│   │   └── pepper_bell_split_v1.json   tomato_split_v1.json
│   ├── golden\                   # BẠN tạo: golden set cho parity/smoke test
│   │   └── tomato.json           # {"images": ["<abs_path>", ...]} — vài ảnh chuẩn mỗi loài
│   └── leaf_gate\                # dataset Leaf Gate (mục 3): leaf tự trích + not_leaf bạn thêm
│       ├── train\{leaf, not_leaf}\ *.jpg
│       ├── calib\{leaf, not_leaf}\ *.jpg
│       └── val\  {leaf, not_leaf}\ *.jpg
├── checkpoints\                  # *.pt: backbone_pretrained, model_<species>, leaf_gate
│   └── model_<species>_prototypes.pt   # prototype lớp hiếm (nếu có)
├── model_bundles\                # bundle bất biến để phục vụ + con trỏ active
│   ├── <version>\                # model.onnx, manifest.json, class_mapping.json, ...
│   └── active_version_<species>.txt    # trỏ tới bundle đang chạy (đổi = rollback)
├── experiments\                  # log mỗi lần train (config, metric, confusion, Grad-CAM)
└── reports\                      # báo cáo cross-test, robustness
```

### 2.3. Golden set (cho Phase 4 parity)

Tạo tay file JSON liệt kê vài ảnh đại diện mỗi loài (lấy từ split `train`/`val`, tránh holdout):

```json
// data/golden/tomato.json
{ "images": [
  "D:/DAT301m_Project/Final_Pipeline/data/2. FINETUNE DATASET/tomato/Early_blight/xxx.jpg",
  "D:/DAT301m_Project/Final_Pipeline/data/2. FINETUNE DATASET/tomato/healthy/yyy.jpg"
] }
```

Golden set dùng để so PyTorch vs ONNX (parity) và smoke test trước khi deploy — nên chọn ảnh
"kinh điển", rõ ràng, đủ các lớp chính của loài.

---

## 3. Dataset Leaf Gate

Leaf Gate là bộ phân loại **nhị phân**: `leaf` (lá cây hợp lệ) vs `not_leaf` (mọi thứ còn lại).
Nó là **chốt chặn đầu tiên** — chặn ảnh rác trước khi đưa vào model loài. Dự án **chưa** có sẵn
ảnh `not_leaf`, bạn phải tự dựng theo hướng dẫn dưới.

### 3.1. Lớp `leaf` (positive) — dùng script tự trích

Chạy script để trích ảnh lá từ **split train** của các manifest (đã chạy Phase 0 trước):

```bash
python -m src.data.build_leaf_gate_dataset --per-species 700 --seed 42
# thêm --dry-run để chỉ đếm, chưa copy

# Nếu ảnh not_leaf gom trong 1 thư mục (chưa chia), truyền thêm để script tự chia 80/10/10:
python -m src.data.build_leaf_gate_dataset --per-species 700 --seed 42 \
       --not-leaf-src "D:/path/to/not_leaf_raw"
```

Script tự động: chỉ lấy ảnh `split=train` (không đụng holdout/val/calib của bài toán loài → tránh
rò rỉ), rải đều theo lớp trong từng loài, rồi chia 80/10/10 vào `leaf_gate/{train,calib,val}/leaf/`.
Nếu dùng `--not-leaf-src`, script cũng chia not_leaf tương tự và in tỷ lệ leaf:not_leaf từng split.

### 3.2. Lớp `not_leaf` (negative) — TIÊU CHÍ CHỌN

Mục tiêu: dạy gate từ chối mọi ảnh **không có một chiếc lá làm chủ thể**. Chọn theo 7 nhóm,
ưu tiên nhóm đầu vì đó là ca người dùng hay chụp nhầm và khó phân biệt nhất:

| # | Nhóm | Ví dụ | Vì sao cần |
|---|---|---|---|
| 1 | Bộ phận cây KHÁC lá | thân, cành, rễ, hoa, quả, hạt, vỏ | Sát miền nhất → khó & quan trọng nhất |
| 2 | Cả cây / tán xa | bụi cây, tán cây chụp xa, bãi cỏ | Người dùng chụp cả cây thay vì 1 lá |
| 3 | Nền nông nghiệp không lá chủ thể | đất, luống, chậu, lưới, bạt nhà kính | Ảnh "hụt" chủ thể |
| 4 | Bàn tay / người / vật cầm | ngón tay, găng, tay che lá | Rất hay gặp khi chụp thực tế |
| 5 | Vật thể đời thường | bàn, tường, giấy, điện thoại, xe, đồ ăn | Ảnh tải nhầm hoàn toàn |
| 6 | Ảnh nhiễu / hỏng | quá tối/sáng, mờ toàn phần, ảnh chụp màn hình | Ca ảnh rác |
| 7 | Lá NGOÀI 5 loài *(nâng cao)* | lá cây cảnh, cây ngoài danh mục | Xem lưu ý bên dưới |

**Nguyên tắc:**
- Đa dạng nền, ánh sáng, góc chụp, độ phân giải — càng giống điều kiện người dùng thật càng tốt.
- Tỷ lệ gợi ý: **~60% nhóm 1–4** (khó, sát miền) + **~40% nhóm 5–6** (dễ).
- **Nhóm 7 — khuyến nghị KHÔNG trộn vào `not_leaf`.** "Là lá nhưng sai loài" nên để tầng
  **energy-OOD + ngưỡng loài** xử lý (Mục 3.2 kế hoạch), không phải việc của leaf-gate. Trộn vào
  sẽ khiến gate từ chối luôn mọi lá lạ và làm mờ ranh giới hai loại reject. Chỉ thêm nếu bạn cố ý
  muốn gate chặn mọi lá ngoài danh mục.

**Nguồn ảnh negative gợi ý:** ImageNet/COCO/Open Images (lọc lớp không liên quan lá); ảnh nền/đồ
vật tự chụp bằng điện thoại (đúng thiết bị người dùng). Bắt đầu ~1.500–3.000 ảnh là đủ.

### 3.3. Cân bằng, split & license

- Giữ tỷ lệ `leaf : not_leaf` khoảng **1:1 đến 1:1.5**. Lệch quá → gate thiên vị.
- Chia `not_leaf` theo cùng **80/10/10** vào `{train,calib,val}/not_leaf/`, stratified theo nhóm.
- **`[Cần verify license]`** cho MỌI ảnh negative từ nguồn ngoài — ghi lại nguồn + giấy phép.
- Strip EXIF (GPS/thiết bị) khỏi ảnh tự chụp nếu định chia sẻ dataset.
- Cấu trúc đích cuối cùng (kiểm lại trước khi train):

```
Final_Pipeline/data/leaf_gate/
├── train/  leaf/ (nhiều)      not_leaf/ (≈ tương đương)
├── calib/  leaf/ (~10%)       not_leaf/ (~10%)
└── val/    leaf/ (~10%)       not_leaf/ (~10%)
```

Train + hiệu chỉnh:

```bash
python -m src.train.train_leaf_gate --config configs/leaf_gate.yaml
python -m src.eval.calibration --config configs/leaf_gate.yaml --target leaf_gate
# rồi chọn ngưỡng binary_leaf từ PR curve: mục tiêu precision >= 0.95, leaf recall >= 0.98
```

---

## 4. Quy trình chạy đầu-cuối

Chạy mọi lệnh từ root `D:\DAT301m_Project\Final_Pipeline` (dùng `python -m ...` để import `src` chạy đúng).
Thứ tự: **0 → 1 → 2 → 3 → 4 → 4b → 5**.

### Phase 0 — Split manifest (1 lần)
```bash
python -m src.data.split_builder --species all --seed 42
```
→ Sinh `data/splits/<species>_split_v1.json` (70/10/10/10 + SHA256 + rare_folds). Đây là **nguồn
sự thật** về chia dữ liệu; mọi bước sau đọc từ đây.

### Phase 1 — Pretrain backbone + Leaf Gate
```bash
python -m src.train.train_pretrain  --config configs/pretrain_effv2s.yaml
python -m src.train.train_leaf_gate --config configs/leaf_gate.yaml
```
→ `checkpoints/backbone_pretrained.pt` (điểm khởi tạo CHUNG cho 5 loài) + `checkpoints/leaf_gate.pt`.
Đầu mỗi script có `probe_vram_usage()` — nếu in `OOM` thì dừng có thông báo, hãy giảm `batch_size`
hoặc tăng `accumulation_steps` trong config.

### Phase 2 — Fine-tune 5 loài (tuần tự cho an toàn VRAM)
```bash
python -m src.train.train_finetune --config configs/finetune_apple.yaml
python -m src.train.train_finetune --config configs/finetune_corn.yaml
python -m src.train.train_finetune --config configs/finetune_grape.yaml
python -m src.train.train_finetune --config configs/finetune_pepper_bell.yaml
python -m src.train.train_finetune --config configs/finetune_tomato.yaml
```
→ Mỗi loài sinh `model_<species>.pt`. Loài có lớp hiếm (tomato) sinh thêm
`model_<species>_prototypes.pt`. Train gồm 3 bước: warmup head → mở BN + block cuối → decoupled
re-training với sampler cân bằng.

### Phase 3 — Đánh giá + calibration
```bash
python -m src.eval.evaluate    --config configs/finetune_tomato.yaml --split val
python -m src.eval.calibration --config configs/finetune_tomato.yaml --target species
# (lặp cho 5 loài + leaf_gate)
```
→ `evaluate` in classification report + confusion matrix. `calibration` sinh
`model_<species>_calibration.json` (nhiệt độ T + ECE trước/sau). Threshold reject chọn trên xác
suất ĐÃ hiệu chỉnh, dùng split `calib` (không đụng val/holdout).

### Phase 4 — Export ONNX + parity + đóng gói bundle
```bash
python -m src.export.export_onnx  --config configs/finetune_tomato.yaml \
       --out model_bundles/tmp/tomato/model.onnx
python -m src.export.parity_test  --bundle model_bundles/tmp/tomato/ \
       --pt-ckpt checkpoints/model_tomato.pt --golden-set data/golden/tomato.json
python -m src.export.build_bundle --species tomato \
       --version tomato_2026_07_04_effv2s_ab12cd \
       --onnx model_bundles/tmp/tomato/model.onnx --git-commit <commit>
```
→ `export_onnx` sinh `model.onnx` + `class_mapping.json`. `parity_test` **chặn** nếu top-1 lệch
hoặc sai số > atol. `build_bundle` gom mọi artefact → `manifest.json` + `checksums.sha256`, và
**chặn release** nếu thiếu ONNX/class_mapping. Lặp cho 5 loài + leaf_gate.

### Phase 4b — Holdout (CHỈ CHẠY MỘT LẦN, sau khi chốt mọi thứ)
```bash
python -m src.eval.evaluate --config configs/finetune_tomato.yaml --split holdout
```
→ Con số holdout là bằng chứng cuối cùng. **Không** tuning gì sau bước này (nếu tuning theo holdout
là rò rỉ, mất giá trị báo cáo).

### Phase 5 — Serving
```bash
echo tomato_2026_07_07_effv2s_ab12cd > model_bundles/active_version_tomato.txt
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```
→ Chi tiết ở mục 5.

---

## 5. Serving, LLM & frontend

### 5.1. Kích hoạt bundle & chạy API
API chỉ phục vụ bundle được trỏ bởi `active_version_<species>.txt`. Đổi con trỏ = đổi model
(hoặc **rollback** về bản cũ):

```bash
# trỏ tới bundle muốn phục vụ (lặp cho từng loài đã có bundle)
echo tomato_2026_07_04_effv2s_ab12cd > model_bundles/active_version_tomato.txt
# leaf-gate đặt bundle tại model_bundles/leaf_gate/ (API tự nạp nếu có)

uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

Mở trình duyệt: **http://localhost:8000** → frontend PhytoScan. Các endpoint:

| Endpoint | Ý nghĩa |
|---|---|
| `GET /health`, `GET /ready` | Liveness / readiness |
| `POST /predict?species=<sp>` (form `file`) | API thô (test/script) |
| `POST /api/predict` (form `species,file,language`) | Cho frontend |
| `GET /api/species`, `GET /api/diseases?species=` | Danh mục cho dropdown |
| `POST /api/consult` | Tư vấn LLM (mục 5.3) |

### 5.2. Chạy bằng Docker (serving CPU-only)
```bash
docker build -t phytoscan-api:latest .
docker run -p 8000:8000 \
  -e LLM_API_KEY="sk-..." -e LLM_BASE_URL="https://api.openai.com/v1" -e LLM_MODEL="gpt-4o-mini" \
  phytoscan-api:latest
```

### 5.3. Tư vấn LLM ngoài (`/api/consult`)
Cấu hình qua biến môi trường — **chỉ gửi text** (loài/bệnh/câu hỏi), không gửi ảnh/PII. Thiếu key
hoặc lỗi mạng thì tự trả câu tĩnh an toàn.

```bash
export LLM_API_KEY="sk-..."
export LLM_BASE_URL="https://api.openai.com/v1"   # đổi sang Groq/DeepSeek/OpenRouter cho rẻ
export LLM_MODEL="gpt-4o-mini"                     # model rẻ để tiết kiệm
export LLM_MAX_TOKENS="400"
```

> ⚠️ Bảo mật: `/predict` và `/api/*` hiện **chưa** có auth/rate-limit. Trước khi mở Internet công
> khai phải thêm API key + rate-limiting + HTTPS. Riêng `/api/consult` nên có rate-limit riêng vì
> đây là endpoint tốn tiền theo lượt gọi.

### 5.4. Frontend PhytoScan
`src/deploy/static/index.html` — landing page (theme xanh lá nền tối, font Inter) được `app.py`
phục vụ tĩnh tại `/`. Danh sách loài lấy động từ `/api/species`, nên chỉ hiện loài đã có bundle
active. Không cần build gì thêm.

---

## 6. Test & xử lý sự cố

### 6.1. Chạy test
```bash
pytest src/tests/
```
Bộ test tách hai loại: test **logic thuần** (checksum, class-mapping, readiness/rollback,
parity-metrics, preprocessing, upload-security) chạy mọi nơi; test cần **torch/onnx/PIL** tự
`skip` nếu thiếu dependency, và sẽ chạy đầy đủ trên máy đã cài `requirements.txt`.

### 6.2. Lỗi thường gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `probe_vram ... OOM` khi train | 4GB không đủ ở batch hiện tại → giảm `batch_size` (16→8) và/hoặc tăng `accumulation_steps` để giữ effective batch. Cân nhắc backbone B0. |
| `RuntimeError: OOM` giữa chừng | Đóng app ngốn VRAM khác; đặt `img_size: 224`; đảm bảo `use_amp: true`. |
| `FileNotFoundError` split manifest | Chưa chạy Phase 0 (`split_builder`). |
| `KeyError` khi load class_mapping | class_index không liên tục → kiểm tên lớp khớp `taxonomy_v1.yaml`. |
| Parity FAIL | ONNX lệch PyTorch → kiểm opset, tránh op động; nếu do INT8 thì đo lại macro-F1, cân nhắc FP16. |
| API `species_unavailable` | Chưa đặt `active_version_<species>.txt` hoặc bundle lỗi. |
| Import lỗi khi chạy `api_compat` trực tiếp | Chỉ chạy qua `uvicorn src.deploy.app:app`, không chạy `api_compat.py` riêng. |
| DataLoader treo/chậm trên Windows | Giảm `num_workers` (2 → 0) nếu gặp lỗi spawn. |

### 6.3. Lưu ý phần cứng (4GB VRAM là ràng buộc hạng nhất)
- **AMP + gradient accumulation + PEFT** bắt buộc để vừa 4GB. Không dùng batch thật 32/64.
- `probe_vram_usage()` chạy đầu mỗi script train — OOM thì dừng có thông báo, không crash giữa chừng.
- `num_workers=2` (16GB RAM); **không** dùng 8.
- `img_size=224` mặc định; `256` chỉ sau khi probe VRAM còn dư.
- Backbone bị **loại khỏi production** nếu dry-run 20 step vượt ~3.6GB reserved VRAM.

---

*Tài liệu thiết kế đầy đủ: [`DAT301m_Plan_Final.md`](DAT301m_Plan_Final.md). Dự án chỉ phục vụ mục
đích học tập, không thay thế tư vấn chuyên gia nông nghiệp.*
