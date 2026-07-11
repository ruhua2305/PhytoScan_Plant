# PhytoScan — DAT301m Plant Disease Classification

Hệ thống phân loại **bệnh cây trồng qua ảnh lá** cho **5 loài**: `apple`, `corn`, `grape`,
`pepper_bell`, `tomato` (tổng **20 lớp bệnh/khỏe**). Sản phẩm cuối là **web demo**
(FastAPI + frontend PhytoScan): người dùng chọn loài, upload ảnh lá, hệ thống kiểm tra ảnh hợp lệ
bằng **Leaf Gate**, chẩn đoán bệnh kèm **độ tin cậy đã hiệu chỉnh**, và có khung **từ chối** khi ảnh
không phải lá / độ tin cậy thấp / ngoài phân phối.

Toàn bộ thiết kế chi tiết nằm trong [`DAT301m_Plan_Final.md`](DAT301m_Plan_Final.md). README này
tập trung vào **cách tổ chức dữ liệu và cách chạy**.

## Đặc điểm chính

| Hạng mục | Giá trị |
|---|---|
| Phần cứng đích | RTX 3050 Ti Laptop **4GB VRAM** / **16GB RAM** / Windows |
| Backbone | EfficientNetV2-S (primary), B0 (fallback) |
| Định tuyến | Per-species: 5 model riêng, người dùng chọn loài |
| Chia dữ liệu | 70/10/10/10 = train/calib/val/holdout, manifest khoá + SHA256 |
| Lớp ít mẫu nhất | `Tomato/Spider_mites` (**55 ảnh**) — vẫn train head thường; không còn lớp <30 |
| Chống 4GB OOM | AMP + gradient accumulation + PEFT + `probe_vram_usage()` |
| Phục vụ | ONNX Runtime (CPU-only); bundle tải từ **Hugging Face Hub** (lazy load per-species) |
| Tư vấn | `/api/consult` gọi LLM ngoài (chỉ gửi text, cấu hình qua ENV) |

## Mục lục

1. [Cài đặt môi trường](#1-cài-đặt-môi-trường)
2. [Tổ chức thư mục dataset](#2-tổ-chức-thư-mục-dataset)
3. [Dataset Leaf Gate](#3-dataset-leaf-gate)
4. [Quy trình chạy đầu-cuối](#4-quy-trình-chạy-đầu-cuối)
5. [Serving, LLM & frontend](#5-serving-llm--frontend)
6. [Cross-Test (so với nghiên cứu công bố)](#6-cross-test-so-với-nghiên-cứu-công-bố)
7. [Test & xử lý sự cố](#7-test--xử-lý-sự-cố)

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
├── 2. FINETUNE DATASET\          # ~18.402 ảnh, 20 lớp, 5 loài — ảnh FIELD (miền đích thật)
│   ├── Apple\
│   │   ├── Apple___Apple_scab\        *.jpg
│   │   ├── Apple___Cedar_apple_rust\  *.jpg
│   │   └── Apple___healthy\           *.jpg
│   ├── Corn_(maize)\
│   │   ├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot\  Corn_(maize)___Common_rust_\
│   │   ├── Corn_(maize)___Northern_Leaf_Blight\  Corn_(maize)___healthy\
│   ├── Grape\
│   │   ├── Grape___Black_rot\  Grape___healthy\
│   ├── Pepper,_bell\
│   │   ├── Pepper,_bell___Bacterial_spot\  Pepper,_bell___healthy\
    └── Tomato\
        ├── Tomato___Bacterial_spot\ Tomato___Early_blight\ Tomato___Late_blight\
        ├── Tomato___Leaf_Mold\ Tomato___Septoria_leaf_spot\
        ├── Tomato___Spider_mites Two-spotted_spider_mite\
        ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus\ Tomato___Tomato_mosaic_virus\ Tomato___healthy\
```

> Tên thư mục loài thật trên đĩa (`Corn_(maize)`, `Pepper,_bell`) khác key nội bộ (`corn`,
> `pepper_bell`); `split_builder.py` tự ánh xạ qua `SPECIES_DIR_NAME` + dò không phân biệt hoa/thường.

> **Tên lớp** = **tên thư mục đầy đủ** (vd `Tomato___Bacterial_spot`). `class_index` do
> `split_builder.py` sinh tự động bằng cách **sort tên thư mục lớp** — bất biến, lưu vào
> `class_mapping.json` của bundle và dùng chung cho ONNX/API. Frontend dịch sang nhãn hiển thị
> song ngữ (en/vi) trong `api_compat.py`.

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
│   └── model_<species>_calibration.json   # T + ECE trước/sau (sinh ở Phase 3)
├── model_bundles\                # bundle đóng gói offline (trước khi push lên HF Hub)
│   ├── <version>\                # model.onnx, manifest.json, class_mapping.json, ...
│   └── active_version_<species>.txt    # con trỏ cho phương án chạy local (serving thật dùng HF Hub)
├── experiments\                  # (dành cho) log mỗi lần train — hiện trống
└── reports\                      # cross_test\ (kết quả Chiều 2) + (dành cho) robustness
    └── cross_test\               # direction2_*.json/jsonl + CROSS_TEST_RESULTS.md
```

> Với dữ liệu hiện tại **không lớp nào < 30 ảnh** nên `train_finetune.py` **không** sinh file
> `model_<species>_prototypes.pt` (nhánh few-shot chỉ kích hoạt khi có lớp hiếm — xem plan Mục 3.5).

### 2.3. Golden set (cho Phase 4 parity)

Tạo tay file JSON liệt kê vài ảnh đại diện mỗi loài (lấy từ split `train`/`val`, tránh holdout):

```json
// data/golden/tomato.json
{ "images": [
  "D:/DAT301m_Project/Final_Pipeline/data/2. FINETUNE DATASET/Tomato/Tomato___Early_blight/xxx.jpg",
  "D:/DAT301m_Project/Final_Pipeline/data/2. FINETUNE DATASET/Tomato/Tomato___healthy/yyy.jpg"
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
→ Mỗi loài sinh `model_<species>.pt`. Train gồm 3 bước: warmup head → mở BN + block cuối
(CB-Focal + CutMix + early stopping) → decoupled re-training với sampler cân bằng (CE thường).
Nhánh few-shot/prototype (Chế độ 2) **không chạy** vì không còn lớp <30 ảnh.

### Phase 3 — Đánh giá + calibration
```bash
python -m src.eval.evaluate    --config configs/finetune_tomato.yaml --split val
python -m src.eval.calibration --config configs/finetune_tomato.yaml --target species
# (lặp cho 5 loài + leaf_gate)
```
→ `evaluate` in classification report + confusion matrix. `calibration` sinh
`model_<species>_calibration.json` (nhiệt độ T + ECE trước/sau). **Lưu ý:** ECE hiện chưa đạt mục
tiêu ≤0.05 ở hầu hết loài (và leaf-gate còn cao) — cần siết ngưỡng reject bù lại. Ngưỡng reject
(`thresholds.json`) **chưa được build**, nên API hiện chạy với default (xem mục 5).

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
# Bundle được phục vụ từ Hugging Face Hub (Mục 5). Cần HF_TOKEN cho private repo:
export HF_TOKEN=hf_...        # Windows: set HF_TOKEN=... hoặc dùng .env
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```
→ Chi tiết ở mục 5.

---

## 5. Serving, LLM & frontend

### 5.1. Tải bundle & chạy API
API tải bundle model từ **Hugging Face Hub** (private repos khai báo trong `app.py` → `REPO_MAP`),
xác thực bằng biến môi trường `HF_TOKEN`. Startup chỉ nạp leaf-gate (~6MB); mỗi model loài được
**lazy load** khi có request đầu tiên cho loài đó.

```bash
export HF_TOKEN=hf_...        # bắt buộc để tải private repo (Windows: set HF_TOKEN=... / .env)
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

Ánh xạ loài → repo (trong `app.py`): `apple → ruhua2305/apple_effv2s_onnx`, `corn`, `grape`,
`pepper_bell`, `tomato → ruhua2305/tomato_effv2s_onnx`, leaf-gate → `ruhua2305/leaf_gate_MobileNetV3Small`.

> Muốn chạy **hoàn toàn local** (không mạng): đóng gói bundle bằng `build_bundle.py`, trỏ
> `active_version_<species>.txt` và sửa `app.py` để đọc bundle cục bộ thay vì HF Hub.

Mở trình duyệt: **http://localhost:8000** → frontend PhytoScan. Các endpoint:

| Endpoint | Ý nghĩa |
|---|---|
| `GET /health`, `GET /ready` | Liveness / readiness (`ready` chỉ true sau khi lazy-load ≥1 loài) |
| `POST /predict?species=<sp>` (form `file`) | API thô (test/script) |
| `POST /api/predict` (form `species,file,language`) | Cho frontend |
| `GET /api/species`, `GET /api/diseases?species=` | Danh mục song ngữ cho dropdown |
| `POST /api/consult` | Tư vấn LLM (mục 5.3) |

### 5.2. Chạy bằng Docker (serving CPU-only)
```bash
docker build -t phytoscan-api:latest .
docker run -p 8000:8000 \
  -e HF_TOKEN="hf_..." \
  -e LLM_API_KEY="..." -e LLM_BASE_URL="https://api.groq.com/openai/v1" \
  -e LLM_MODEL="llama-3.3-70b-versatile" \
  phytoscan-api:latest
```

> ⚠️ Dockerfile hiện `EXPOSE 8000` + healthcheck cổng 8000 nhưng `CMD` chạy `--port 8080` → không
> khớp (healthcheck sẽ fail). Cần thống nhất về một cổng trước khi dùng thật.

### 5.3. Tư vấn LLM ngoài (`/api/consult`)
Cấu hình qua biến môi trường — **chỉ gửi text** (loài/bệnh/câu hỏi), không gửi ảnh/PII. Thiếu key
hoặc lỗi mạng thì tự trả câu tĩnh an toàn.

```bash
export LLM_API_KEY="..."
export LLM_BASE_URL="https://api.groq.com/openai/v1"   # mặc định code hợp Groq; đổi sang OpenAI/DeepSeek/OpenRouter tùy ý
export LLM_MODEL="llama-3.3-70b-versatile"             # mặc định trong code
export LLM_MAX_TOKENS="2048"                            # hạ xuống để tiết kiệm chi phí
```

> ⚠️ Bảo mật: `/predict` và `/api/*` hiện **chưa** có auth/rate-limit. Trước khi mở Internet công
> khai phải thêm API key + rate-limiting + HTTPS. Riêng `/api/consult` nên có rate-limit riêng vì
> đây là endpoint tốn tiền theo lượt gọi.

### 5.4. Frontend PhytoScan
`src/deploy/static/index.html` — landing page (theme xanh lá nền tối, font Inter, song ngữ vi/en)
được `app.py` phục vụ tĩnh tại `/`. Danh sách loài lấy từ `/api/species` (5 loài trong `REPO_MAP`),
danh mục bệnh từ `/api/diseases` (dịch song ngữ trong `api_compat.py`). Không cần build gì thêm.

---

## 6. Cross-Test (so với nghiên cứu công bố)

Đặt pipeline cạnh công trình công bố để định lượng **domain gap lab→field** — model train trên ảnh
lab (PlantVillage) chạy kém trên ảnh field thật, đúng phát hiện Mohanty et al. 2016
([DOI:10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419)).

### 6.1. Chiều 2 — model công khai → holdout FIELD của mình (ĐÃ CHẠY)
Chạy model chỉ-biết-lab trên holdout field, so với model per-species đã fine-tune:

```bash
python -m src.cross_test.run_direction2 --model mine       # baseline fine-tune field (offline)
python -m src.cross_test.run_direction2 --model backbone   # backbone-38 của mình, chỉ biết lab (offline)
python -m src.cross_test.run_direction2 --model deit       # DeiT-39 ngoài (cần transformers + tải HF ~330MB)
```

Kết quả (holdout field, macro-F1 trung bình 5 loài):

| Model | Train trên | macro-F1 | Accuracy |
|---|---|---:|---:|
| **Của mình (fine-tune field)** | field | **0.595** | 0.780 |
| DeiT-39 ngoài | PlantVillage lab | 0.479 | 0.559 |
| Backbone-38 của mình | PRETRAIN lab | 0.312 | 0.236 |

→ Model lab thua rõ model field-adapted. Chi tiết + bảng per-species + kiểm chứng "backbone không
hỏng" (99.77% trên lab val): [`reports/cross_test/CROSS_TEST_RESULTS.md`](reports/cross_test/CROSS_TEST_RESULTS.md).

Nguồn model công khai: [rescu/deit-base-patch16-224-finetuned-plantvillage](https://huggingface.co/rescu/deit-base-patch16-224-finetuned-plantvillage) (39 lớp, có id2label) ·
backbone-38 = `checkpoints/backbone_pretrained.pt` (PlantVillage-38, [Hughes & Salathé 2015](https://arxiv.org/abs/1511.08060)).

### 6.2. Chiều 1 — model mình → dataset công bố (CHƯA CHẠY)
Test model mình trên PlantDoc. **Tạm gác** vì FINETUNE đã chứa 742 ảnh `PLANTDOC_` → cần **de-dup
SHA256** trước khi so (nếu không sẽ rò rỉ, số liệu lạc quan giả). Khi làm:
`git clone https://github.com/pratikkayal/PlantDoc-Dataset` (CC-BY-4.0), map 28 lớp PlantDoc ↔ 20
lớp mình, de-dup hash, rồi dùng helper trong `src/cross_test/run_cross_test.py`.

---

## 7. Test & xử lý sự cố

### 7.1. Chạy test
```bash
pytest src/tests/
```
Bộ test tách hai loại: test **logic thuần** (checksum, class-mapping, readiness/rollback,
parity-metrics, preprocessing, upload-security) chạy mọi nơi; test cần **torch/onnx/PIL** tự
`skip` nếu thiếu dependency, và sẽ chạy đầy đủ trên máy đã cài `requirements.txt`.

### 7.2. Lỗi thường gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `probe_vram ... OOM` khi train | 4GB không đủ ở batch hiện tại → giảm `batch_size` (16→8) và/hoặc tăng `accumulation_steps` để giữ effective batch. Cân nhắc backbone B0. |
| `RuntimeError: OOM` giữa chừng | Đóng app ngốn VRAM khác; đặt `img_size: 224`; đảm bảo `use_amp: true`. |
| `FileNotFoundError` split manifest | Chưa chạy Phase 0 (`split_builder`). |
| `KeyError` khi load class_mapping | class_index không liên tục → chạy lại `split_builder` và kiểm `class_mapping.json` của bundle (index sinh từ sort tên thư mục lớp). |
| Parity FAIL | ONNX lệch PyTorch → kiểm opset, tránh op động; nếu do INT8 thì đo lại macro-F1, cân nhắc FP16. |
| API `500 Lỗi tải model` | `HF_TOKEN` chưa set / sai, hoặc repo HF không truy cập được → kiểm token + `REPO_MAP`. |
| Import lỗi khi chạy `api_compat` trực tiếp | Chỉ chạy qua `uvicorn src.deploy.app:app`, không chạy `api_compat.py` riêng. |
| DataLoader treo/chậm trên Windows | Giảm `num_workers` (2 → 0) nếu gặp lỗi spawn. |

### 7.3. Lưu ý phần cứng (4GB VRAM là ràng buộc hạng nhất)
- **AMP + gradient accumulation + PEFT** bắt buộc để vừa 4GB. Không dùng batch thật 32/64.
- `probe_vram_usage()` chạy đầu mỗi script train — OOM thì dừng có thông báo, không crash giữa chừng.
- `num_workers=2` (16GB RAM); **không** dùng 8.
- `img_size=224` mặc định; `256` chỉ sau khi probe VRAM còn dư.
- Backbone bị **loại khỏi production** nếu dry-run 20 step vượt ~3.6GB reserved VRAM.

---

*Tài liệu thiết kế đầy đủ: [`DAT301m_Plan_Final.md`](DAT301m_Plan_Final.md). Dự án chỉ phục vụ mục
đích học tập, không thay thế tư vấn chuyên gia nông nghiệp.*
