# DAT301m — Tài liệu Giải thích Toàn diện Dự án (bản chi tiết)
## Phân loại Bệnh Cây Trồng qua Ảnh Lá — PhytoScan


> **Cách tổ chức:** đọc theo **đúng thứ tự quy trình huấn luyện** (data → chia → pretrain → fine-tune → đánh giá → đóng gói → phục vụ). Mỗi **file code** được mổ xẻ theo **từng thành phần bên trong** (hàm/class/biến), giải thích *nó nhận gì, làm gì từng bước, trả gì, ai gọi nó*. Mọi **thuật ngữ** được giải thích cụ thể ngay lần đầu xuất hiện, kèm ví dụ.
>
> **Ký hiệu trong tài liệu:**
> - 📁 = file code · 🔧 = hàm/class · 🧠 = lý thuyết · 💬 = câu hỏi giảng viên hay hỏi · ⚠️ = điểm cần lưu ý/chưa hoàn thiện.

---

## Mục lục

- [Phần A — Hiểu bức tranh lớn trước khi đọc code](#phần-a--hiểu-bức-tranh-lớn)
- [Phần B — Từ điển thuật ngữ (giải thích cực kỹ)](#phần-b--từ-điển-thuật-ngữ)
- [Phần C — Quy trình huấn luyện tổng thể (7 giai đoạn)](#phần-c--quy-trình-huấn-luyện-tổng-thể)
- [Phần D — Đọc code theo từng file](#phần-d--đọc-code-theo-từng-file)
  - [D0. `config.py` — đọc cấu hình](#d0-configpy)
  - [D1. `split_builder.py` — chia dữ liệu (Giai đoạn 0)](#d1-split_builderpy)
  - [D2. `datasets.py` — nạp ảnh](#d2-datasetspy)
  - [D3. `augmentation.py` — tăng cường dữ liệu](#d3-augmentationpy)
  - [D4. `domain_augmentation.py` — ghép nền field](#d4-domain_augmentationpy)
  - [D5. `backbone.py` — tạo mạng](#d5-backbonepy)
  - [D6. `memory_utils.py` — tiết kiệm VRAM](#d6-memory_utilspy)
  - [D7. `losses.py` — hàm mất mát](#d7-lossespy)
  - [D8. `samplers.py` — lấy mẫu cân bằng](#d8-samplerspy)
  - [D9. `peft.py` — mở khoá tham số dần](#d9-peftpy)
  - [D10. `train_pretrain.py` — Giai đoạn 1](#d10-train_pretrainpy)
  - [D11. `build_leaf_gate_dataset.py` + `train_leaf_gate.py` — Leaf Gate](#d11-leaf-gate)
  - [D12. `train_finetune.py` — Giai đoạn 2 (trái tim)](#d12-train_finetunepy)
  - [D13. `few_shot.py` + `domain_loss.py` — kỹ thuật phụ trợ](#d13-few_shot--domain_loss)
  - [D14. `evaluate.py` + `calibration.py` — Giai đoạn 3](#d14-evaluate--calibration)
  - [D15. `export_onnx.py` + `parity_test.py` + `build_bundle.py` — Giai đoạn 4](#d15-export--bundle)
  - [D16. `app.py` + `api_compat.py` — Giai đoạn 5 (serving)](#d16-serving)
  - [D17. Cross-test](#d17-cross-test)
- [Phần E — Câu hỏi giảng viên hay hỏi (đầy đủ)](#phần-e--câu-hỏi-giảng-viên)
- [Phần F — Tra nguồn paper/model](#phần-f--tra-nguồn)

---

## Phần A — Hiểu bức tranh lớn

### A.1. Dự án làm gì (1 đoạn)

Người dùng chụp/ tải **một ảnh lá cây**, chọn **loài** (táo/ngô/nho/ớt chuông/cà chua), hệ thống trả về **bệnh** mà chiếc lá đang mắc (hoặc "khỏe mạnh"), kèm **độ tin cậy**. Nếu ảnh không phải lá hoặc quá lạ, hệ thống **từ chối** thay vì đoán bừa. Có thêm chatbot tư vấn cách xử lý bệnh.

Tổng cộng **20 lớp** phân bố trên 5 loài (mỗi loài vài bệnh + 1 lớp "khỏe").

### A.2. "Machine learning" ở đây thực chất là gì?

Hình dung model như **một học sinh nhìn tranh đoán bệnh**:
1. **Huấn luyện (training):** đưa cho học sinh hàng nghìn ảnh lá **đã dán nhãn đúng** ("đây là bệnh ghẻ táo", "đây là lá khỏe"). Học sinh tự điều chỉnh "cách nhìn" (các con số bên trong gọi là **trọng số / weights**) để đoán ngày càng đúng.
2. **Suy luận (inference):** sau khi học xong, đưa ảnh **mới chưa từng thấy**, học sinh đoán.

Toàn bộ việc "điều chỉnh cách nhìn" là tự động, dựa trên việc **đo độ sai (loss)** rồi **giảm nó dần**.

### A.3. Vì sao dự án này khó/đặc biệt?

- **Ảnh thật ngoài đồng (field) khó hơn ảnh phòng lab rất nhiều.** Lá thật có nền lộn xộn (đất, tay, trời), ánh sáng thất thường, chụp nghiêng, mờ. Đa số nghiên cứu train trên ảnh lab sạch → đạt 99% nhưng **sập** trên ảnh thật. Dự án này train thẳng trên ảnh field và **chứng minh bằng số** điều đó (phần Cross-test).
- **Dữ liệu lệch nặng:** lớp nhiều nhất 5,555 ảnh, lớp ít nhất 55 ảnh (chênh ~100 lần). Model dễ "lười" đoán toàn lớp đông.
- **Máy yếu:** GPU laptop chỉ 4GB VRAM (bộ nhớ card đồ hoạ). Model to sẽ "tràn bộ nhớ". Phải dùng nhiều mẹo để nhét vừa.

### A.4. Ba nguyên tắc thiết kế xuyên suốt

1. **Trung thực & kiểm toán được:** chia dữ liệu khoá bằng mã hash, không cho rò rỉ đáp án, ghi rõ chỗ nào chưa làm xong.
2. **Chọn model theo macro-F1** (không phải accuracy) vì dữ liệu lệch.
3. **Tách huấn luyện khỏi phục vụ:** train cần GPU mạnh; phục vụ chỉ cần CPU (dùng ONNX) → nhẹ, rẻ.

---
## Phần B — Từ điển thuật ngữ

> Giải thích theo kiểu "cho người mới": mỗi thuật ngữ có **định nghĩa + ví dụ + vì sao cần**. Bạn có thể đọc phần này riêng như một cuốn từ điển.

### B.1. Nhóm cơ bản về mạng nơ-ron

**Tensor** — mảng số nhiều chiều (giống ma trận nhưng có thể >2 chiều). Ảnh màu 224×224 là tensor `[3, 224, 224]`: 3 kênh màu (R, G, B), mỗi kênh là lưới 224×224 giá trị độ sáng. Một *batch* 16 ảnh là tensor `[16, 3, 224, 224]`.

**Trọng số (weights / parameters)** — các con số bên trong mạng, quyết định mạng "nhìn" ảnh thế nào. Training = tìm bộ trọng số tốt. EfficientNetV2-S có ~21 triệu trọng số.

**Layer (lớp mạng)** — một phép biến đổi tensor. Xếp chồng nhiều lớp → mạng sâu (deep). Ví dụ lớp tích chập (convolution) dò các đốm/vết trên lá.

**CNN (Convolutional Neural Network)** — mạng nơ-ron tích chập, chuyên xử lý ảnh. "Tích chập" = trượt một cửa sổ nhỏ khắp ảnh để dò đặc trưng cục bộ (cạnh, đốm, texture). Rất hợp bài toán bệnh lá vì bệnh biểu hiện qua đốm/vết cục bộ.

**Logits** — đầu ra **thô** của mạng, mỗi lớp một số thực (có thể âm/dương, chưa phải xác suất). Ví dụ 3 lớp táo → logits `[2.1, -0.5, 1.3]`.

**Softmax** — hàm biến logits thành **xác suất** (các số ≥0 và cộng lại =1). Công thức: `softmax(z)_i = e^(z_i) / Σ e^(z_j)`. Logits `[2.1, -0.5, 1.3]` → xác suất `[0.63, 0.05, 0.32]`.

**argmax** — trả về **vị trí** của giá trị lớn nhất. `argmax([0.63, 0.05, 0.32]) = 0` → đoán lớp 0. Đây là "câu trả lời cuối" của model.

**Forward pass (lượt tiến)** — đẩy ảnh qua mạng để ra logits. **Backward pass (lượt lùi / backpropagation)** — tính xem mỗi trọng số cần chỉnh bao nhiêu để giảm sai số, lan ngược từ loss về đầu mạng.

**Gradient (đạo hàm)** — "độ dốc" cho biết chỉnh trọng số theo hướng nào thì loss giảm. Backward pass tính gradient; optimizer dùng gradient để cập nhật trọng số.

### B.2. Nhóm huấn luyện

**Loss (hàm mất mát)** — con số đo model **sai bao nhiêu** trên một batch. Train = làm loss nhỏ dần. Ví dụ Cross-Entropy Loss: đoán đúng tự tin → loss thấp; đoán sai tự tin → loss cao (phạt nặng).

**Cross-Entropy (CE)** — loss chuẩn cho phân loại. Công thức cho 1 mẫu: `CE = -log(p_đúng)` với `p_đúng` là xác suất model gán cho lớp đúng. Nếu model gán 0.9 cho lớp đúng → CE = -log(0.9) ≈ 0.1 (tốt); gán 0.1 → CE ≈ 2.3 (tệ).

**Epoch** — một lượt model đi qua **toàn bộ** tập train. Train thường chạy nhiều epoch (VD 25). Sau mỗi epoch, đo trên val.

**Batch & batch size** — vì không thể nạp cả chục nghìn ảnh cùng lúc, ta chia thành **lô (batch)**, VD 16 ảnh/lô. Batch size = số ảnh mỗi lô. Càng lớn càng tốn VRAM.

**Optimizer (bộ tối ưu)** — thuật toán dùng gradient để cập nhật trọng số. **AdamW** = phiên bản Adam có "weight decay" đúng cách (Adam = tự điều chỉnh bước nhảy cho từng trọng số dựa trên lịch sử gradient).

**Learning rate (LR, tốc độ học)** — độ lớn bước nhảy mỗi lần cập nhật. Quá lớn → "nhảy vọt" bỏ lỡ điểm tốt; quá nhỏ → học chậm lê thê. Dự án dùng LR khác nhau cho head (nhanh, 1e-3) và backbone (chậm, 1e-5).

**Weight decay** — phạt trọng số có giá trị lớn, giúp model đơn giản hơn, chống học vẹt. Dự án dùng 1e-4.

**Scheduler (lịch LR)** — giảm LR dần theo thời gian. **CosineAnnealingLR** giảm theo đường cong cos: nhanh lúc đầu, mượt dần về cuối → hội tụ ổn định.

**Label smoothing (làm mềm nhãn)** — thay vì dạy "chắc chắn 100% là lớp A", dạy "99% lớp A, rải 1% cho các lớp khác". Giúp model bớt tự tin thái quá, tổng quát tốt hơn. Dự án dùng 0.1 (pretrain) / 0.05 (leaf gate).

**Overfit (học vẹt / quá khớp)** — model thuộc lòng tập train (kể cả nhiễu) nhưng đoán kém trên ảnh mới. Dấu hiệu: train tốt lên nhưng val xấu đi. Chống bằng: augmentation, weight decay, early stopping, dropout.

**Dropout** — trong lúc train, ngẫu nhiên "tắt" một tỉ lệ nơ-ron → mạng không phụ thuộc quá vào vài nơ-ron → chống overfit. `drop_path_rate=0.1` trong dự án là biến thể (tắt cả nhánh).

**Early stopping (dừng sớm)** — nếu val không cải thiện sau N epoch liên tiếp (N = "patience"), dừng luôn → tiết kiệm thời gian + chống overfit.

**Checkpoint** — file `.pt` lưu trạng thái model (trọng số) tại một thời điểm. Dự án lưu checkpoint có macro-F1 val cao nhất.

### B.3. Nhóm dữ liệu

**Train / Validation (val) / Test (holdout)** — 3 vai trò dữ liệu:
- *Train*: để model học.
- *Val*: để **ta** chọn model tốt nhất, so sánh cấu hình (model có "thấy" val nhưng không học trực tiếp trên đó).
- *Holdout (test)*: **khoá kín**, chỉ mở 1 lần cuối để báo cáo con số thật. Mô phỏng "dữ liệu tương lai chưa từng thấy".
- *Calib*: tập riêng (10%) chỉ để chỉnh độ tin cậy + ngưỡng từ chối.

**Data leakage (rò rỉ dữ liệu)** — khi thông tin từ tập test lọt vào quá trình train/chọn model → con số đẹp giả tạo. Ví dụ: cùng một ảnh nằm ở cả train lẫn test. Dự án chống bằng hash SHA256.

**Stratified (phân tầng)** — chia sao cho **mỗi lớp** đều có mặt đúng tỉ lệ ở mọi phần. Nếu không, lớp hiếm có thể vắng mặt hoàn toàn ở val.

**Class imbalance (mất cân bằng lớp)** — số ảnh giữa các lớp chênh lệch lớn (dự án: 5555 vs 55). Gây model thiên vị lớp đông.

**Augmentation (tăng cường dữ liệu)** — biến đổi ảnh train (lật, xoay, đổi sáng, cắt...) để tạo "ảnh mới" từ ảnh cũ → model thấy đa dạng hơn, chống overfit. **Chỉ áp cho train**, không cho val/test (để đo khách quan).

**Domain / Domain gap (miền / khoảng cách miền)** — "miền" = loại phân phối dữ liệu. Ảnh lab (nền sạch) và ảnh field (nền lộn xộn) là 2 miền khác nhau. "Domain gap" = model học miền này chạy kém trên miền kia. Đây là vấn đề trung tâm của dự án.

### B.4. Nhóm mất cân bằng & metric

**Precision (độ chính xác)** — trong các ca model **đoán là bệnh X**, bao nhiêu % đúng? Cao = ít báo động giả.

**Recall (độ bao phủ / độ nhạy)** — trong các ca **thật sự là bệnh X**, model bắt được bao nhiêu %? Cao = ít bỏ sót.

**F1** — trung bình điều hoà của Precision và Recall: `F1 = 2·P·R/(P+R)`. Cân bằng cả hai; chỉ cao khi cả P lẫn R đều khá.

**Macro-F1** — tính F1 cho **từng lớp** rồi lấy **trung bình cộng không trọng số**. Mọi lớp đóng góp ngang nhau → lớp hiếm cũng quan trọng như lớp đông. **Đây là metric chọn model chính của dự án.**

**Accuracy (độ chính xác tổng)** — số đoán đúng / tổng. Vấn đề: với data lệch, đoán toàn lớp đông vẫn ra accuracy cao → **che giấu** lỗi ở lớp hiếm. Vì vậy dự án không dùng accuracy làm metric chính.

**Confusion matrix (ma trận nhầm lẫn)** — bảng vuông: hàng = lớp thật, cột = lớp đoán. Ô (i,j) = số ảnh lớp i bị đoán thành j. Đường chéo = đoán đúng. Nhìn ô ngoài chéo lớn → biết model hay nhầm cặp bệnh nào (VD Early_blight ↔ Late_blight).

### B.5. Nhóm mất cân bằng — kỹ thuật

**Class weight (trọng số lớp)** — cho lớp hiếm "trọng số" lớn hơn trong loss → model bị phạt nặng hơn khi sai lớp hiếm → chú ý hơn.

**Effective number of samples (số mẫu hữu hiệu)** — ý tưởng của Class-Balanced Loss (Cui 2019): thêm ảnh vào lớp đã nhiều thì ít giá trị (thông tin trùng lặp). Công thức `E_n = (1 - β^n)/(1 - β)`, trọng số ∝ `1/E_n`. β≈0.9999 điều chỉnh mức độ. Đây là cách tính class weight "thông minh" hơn nghịch đảo tần suất thuần.

**Focal Loss** — biến thể CE, nhân thêm `(1 - p_t)^γ` để **hạ trọng số mẫu dễ**: mẫu đã đoán đúng tự tin (p_t cao) thì `(1-p_t)^γ` ≈ 0 → gần như bỏ qua, dồn sức vào mẫu khó. γ (gamma) = 2 là chuẩn.

**Oversampling / WeightedRandomSampler** — khi lấy mẫu train, cho lớp hiếm cơ hội được chọn **nhiều lần hơn** (lấy có hoàn lại) → mỗi batch cân bằng hơn.

### B.6. Nhóm tiết kiệm bộ nhớ (chạy 4GB VRAM)

**VRAM** — bộ nhớ trên card đồ hoạ (GPU). Model + ảnh + gradient đều nằm ở đây khi train. 4GB là rất nhỏ.

**AMP (Automatic Mixed Precision, "độ chính xác hỗn hợp")** — bình thường số thực dùng 32-bit (FP32). AMP dùng 16-bit (FP16) cho phần lớn phép tính → **giảm ~½ bộ nhớ** và nhanh hơn, gần như không mất độ chính xác.

**GradScaler** — vấn đề của FP16: số quá nhỏ bị làm tròn về 0 ("underflow") làm mất gradient. GradScaler **nhân loss lên** một hệ số lớn trước khi tính gradient (để gradient không quá nhỏ), rồi chia lại sau → tránh underflow.

**Gradient accumulation (tích luỹ gradient)** — muốn batch 64 nhưng VRAM chỉ đủ 16. Giải pháp: chạy 4 lô 16 ảnh, **cộng dồn** gradient, sau 4 lô mới cập nhật trọng số 1 lần → hiệu quả như batch 64 mà chỉ tốn VRAM của 16. "Effective batch" = 16×4 = 64.

**Gradient checkpointing** — đánh đổi: không lưu kết quả trung gian (tiết kiệm bộ nhớ) mà tính lại khi cần (tốn thêm chút thời gian). Dự án bật khi thử model transformer.

**PEFT (Parameter-Efficient Fine-Tuning)** — chỉ huấn luyện **một phần nhỏ** trọng số (VD chỉ head + vài lớp cuối), đóng băng phần còn lại. Optimizer chỉ phải lưu trạng thái cho phần được train → tiết kiệm nhiều VRAM.

**requires_grad** — cờ trên mỗi trọng số: `True` = được cập nhật khi train; `False` = "đóng băng" (freeze). PEFT chính là bật/tắt cờ này.

### B.7. Nhóm transfer learning

**Pretrain (tiền huấn luyện)** — train model trên tập lớn trước (VD ImageNet 1.2 triệu ảnh, hoặc PRETRAIN 53k ảnh lá) để học đặc trưng chung.

**Fine-tune (tinh chỉnh)** — lấy model đã pretrain, huấn luyện tiếp trên dữ liệu đích (nhỏ hơn) để chuyên biệt hoá. Dự án fine-tune từng loài.

**Backbone (xương sống)** — phần chính của mạng đã học "cách nhìn ảnh", tái sử dụng qua các bài toán. **Head (đầu)** = lớp cuối biến đặc trưng thành số lớp cụ thể; thường thay head khi đổi bài toán.

**EfficientNetV2-S** — kiến trúc CNN (Tan & Le 2021) cân bằng tốc độ/độ chính xác, ~21M trọng số. "S" = Small. Chọn vì mạnh nhất còn vừa 4GB.

**MobileNetV3-Small** — CNN siêu nhẹ (~2.5M) thiết kế cho điện thoại. Dùng cho Leaf Gate (bài toán nhị phân đơn giản, cần nhanh).

**Discriminative learning rate** — dùng LR khác nhau cho các phần model: head (mới, chưa biết gì) học nhanh; backbone (đã giỏi) học chậm để không phá kiến thức cũ.

### B.8. Nhóm độ tin cậy & từ chối

**Calibration (hiệu chỉnh độ tin cậy)** — làm cho "độ tin cậy model nói" khớp "độ đúng thật". Model nói 90% thì phải đúng ~90% số lần.

**Temperature scaling** — chia logits cho một số T>0 trước softmax. T>1 → xác suất "mềm" ra (bớt tự tin). **Không đổi thứ hạng → không đổi accuracy**, chỉ chỉnh độ tin cậy. Học T bằng cách tối ưu trên tập calib.

**ECE (Expected Calibration Error)** — thước đo lệch giữa độ tin cậy và độ đúng. Chia dự đoán thành nhiều "thùng" theo mức tin cậy, so trong mỗi thùng "tin cậy TB" vs "đúng thật TB". ECE=0 là hoàn hảo. Mục tiêu ≤0.05.

**Brier score** — sai số bình phương giữa vector xác suất và nhãn đúng (one-hot). Càng nhỏ càng tốt. Đo chất lượng xác suất nói chung.

**one-hot** — biểu diễn nhãn dạng vector 0/1: lớp 2 trong 4 lớp → `[0, 0, 1, 0]`.

**OOD (Out-Of-Distribution, "ngoài phân phối")** — ảnh khác hẳn dữ liệu train (VD upload ảnh con mèo vào app bệnh lá). Model nên **từ chối** thay vì đoán bừa.

**Energy score (điểm năng lượng)** — cách phát hiện OOD từ logits (Liu 2020): `E = -log(Σ e^(z_i))`. Ảnh quen (in-distribution) có năng lượng thấp, ảnh lạ có năng lượng cao. Ưu điểm: không cần tập "ảnh xấu" riêng.

**Leaf Gate (cổng lá)** — model nhị phân "lá / không phải lá" đặt **trước** model bệnh, chặn ảnh rác (tay, đất, đồ vật).

**Reject taxonomy (phân loại lý do từ chối)** — các trạng thái trả về: `ok` (chẩn đoán bình thường), `not_a_leaf` (không phải lá), `unknown_or_ood` (lạ), `low_confidence` (không đủ tự tin), `ambiguous` (phân vân giữa 2 bệnh).

### B.9. Nhóm triển khai

**ONNX (Open Neural Network Exchange)** — định dạng model "trung lập", chạy được bằng ONNX Runtime **không cần PyTorch/CUDA** → serving nhẹ trên CPU.

**ONNX Runtime** — thư viện chạy model ONNX, tối ưu cho CPU.

**opset** — phiên bản tập toán tử ONNX (dự án dùng 17). Như "phiên bản ngôn ngữ".

**Parity test (kiểm tra tương đương)** — so kết quả model PyTorch gốc vs bản ONNX trên cùng ảnh, đảm bảo chuyển đổi không làm sai lệch. Yêu cầu top-1 khớp 100% và chênh logit rất nhỏ.

**Quantization (lượng tử hoá)** — nén trọng số xuống INT8 (số nguyên 8-bit) hoặc FP16 → model nhỏ+nhanh hơn, đánh đổi chút độ chính xác. Phải đo lại sau khi nén.

**Bundle (gói triển khai)** — thư mục đóng gói **mọi thứ** cần để phục vụ: `model.onnx` + `class_mapping.json` (idx→tên lớp) + `preprocessing.json` (cách xử lý ảnh) + `calibration.json` + checksums. **Bất biến** (không sửa sau khi đóng) → tránh lệch.

**Checksum (SHA256)** — mã băm của mỗi file trong bundle, để phát hiện nếu file bị sửa đổi.

**FastAPI / uvicorn** — FastAPI = thư viện tạo web API bằng Python; uvicorn = server chạy nó.

**Endpoint** — một địa chỉ URL của API, VD `POST /predict`.

**Lazy load (nạp lười)** — chỉ tải model của một loài khi có request đầu tiên cho loài đó → tiết kiệm RAM lúc khởi động.

**Hugging Face Hub** — kho lưu trữ model/dataset online. Dự án tải bundle ONNX từ đây (repo riêng tư, cần `HF_TOKEN` = khoá truy cập).

**Magic bytes** — vài byte đầu file cho biết định dạng thật (JPEG bắt đầu `FF D8 FF`, PNG bắt đầu `89 50 4E 47`). Kiểm tra để chống file giả đuôi (đổi tên `.exe` thành `.jpg`).

**EXIF** — metadata gắn trong ảnh (thiết bị, GPS, hướng xoay). `exif_transpose` = xoay ảnh về đúng chiều dựa trên metadata.

**Decompression bomb (bom giải nén)** — ảnh nén nhỏ (vài KB) nhưng giải nén ra khổng lồ (hàng trăm triệu pixel) → làm hết RAM, sập server. Chặn bằng giới hạn số pixel.

---
## Phần C — Quy trình huấn luyện tổng thể

> Đây là "bản đồ" nối các file với nhau. Đọc phần này trước, rồi vào Phần D xem chi tiết từng file.

### C.0. Trước Giai đoạn 0 — dựng dữ liệu thô thành 2 tập (upstream)

> ⚠️ **Sơ bộ.** Đây là phần **chưa** nằm trong `src/` mà ở thư mục `raw_data_processing/` — nó tạo ra hai tập `1. PRETRAIN DATASET` và `2. FINETUNE DATASET` mà toàn bộ pipeline `src/` giả định là đã có sẵn. Hiểu phần này để trả lời câu "dữ liệu field ở đâu ra?".

🧠 **Vấn đề:** ta có **11 dataset thô** tải từ nhiều nguồn (tổng ~280k ảnh), lẫn lộn ảnh **lab** (nền sạch), ảnh **field** (nền thật), và cả ảnh **quả** (không phải lá). Ta cần: (a) giữ PlantVillage làm tập **pretrain** (ảnh lab), (b) lọc ra ảnh **field-leaf sạch** từ các dataset còn lại làm tập **fine-tune**.

📁 **4 bước** (theo `raw_data_processing/`):

```text
BƯỚC 1 — datasets_grouping(for_classify_images).py
  Chọn "candidate species" = loài trong PlantVillage có ĐỦ CẢ nhãn healthy LẪN nhãn bệnh
  (loài chỉ có healthy hoặc chỉ có bệnh thì bỏ, vì không phân loại được).
  Chia 11 dataset thành 3 nhóm rồi lọc khác nhau:
    • ONFIELD (field chắc chắn)   → copy thẳng          → ONFIELD_LEAF_DETECTED
        PlantDoc, FieldPlant, Plant Pathology 2020, Plant Pathology 2021
    • LAB_SUSPEN (nghi lẫn lab)   → CLIP lab/field:
        field → ONFIELD_LEAF_DETECTED, lab → LAB_LEAF_DETECTED
        Master Plant Disease Dataset, Plant Disease Classification Merged,
        Tomato-Village-main, TomatoDataset, NGLD
    • FRUIT_SUSPEN (nghi lẫn quả) → CLIP fruit/leaf trước;
        nếu là leaf thì CLIP tiếp lab/field; nếu là quả thì bỏ
        plantseg

BƯỚC 2 — del_unrelevant_label.py  (finalize_candidates)
  • Xóa mọi nhãn bệnh trong LAB nếu nhãn đó KHÔNG có trong ONFIELD.
  • Xóa cả loài nếu (ONFIELD+LAB) không còn đủ cả healthy lẫn bệnh.

BƯỚC 3 — đổi tên (thủ công)
  ONFIELD_LEAF_DETECTED  →  2. FINETUNE DATASET   (ảnh field, miền đích)
  PlantVillage           →  1. PRETRAIN DATASET   (ảnh lab, học đặc trưng)

BƯỚC 4 — dedup (đã chạy)
  • Trùng lặp tuyệt đối (exact-dup): cả 2 tập.
  • Gần trùng (near-duplicate): chỉ tập FINETUNE.
```

🧠 **CLIP là gì ở đây:** CLIP (mô hình vision-language) chấm điểm ảnh khớp với câu mô tả nào hơn (VD "a leaf photo in a laboratory" vs "a leaf photo in the field") → dùng làm bộ phân loại **zero-shot** để tách lab/field và fruit/leaf mà không cần train riêng.

⚠️ **Code cũ (V1) đã bị thay:** `combined_all_into_a_dataset(except_PV).py` + `classify_lab_field_clip.py` + `classification_log.csv` là thế hệ đầu — gộp tất cả vào một `Field_Raw` rồi mới CLIP lab/field, **không** lọc quả. V2 (Bước 1 ở trên) lọc thông minh theo từng dataset. File `lab_field_fruit_leaf_split.py` gần như trùng `datasets_grouping(...)`.

📊 **Kết quả:** `1. PRETRAIN DATASET` = 53,599 ảnh / 38 lớp; `2. FINETUNE DATASET` = 18,354 ảnh / 20 lớp field trên 5 loài. (Sau đó lớp `Tomato/Spider_mites` được bổ sung 7 → 55 ảnh, nâng tổng FINETUNE lên **18,402** — con số dùng xuyên suốt tài liệu training.)

### C.1. Sơ đồ 7 giai đoạn

```text
GIAI ĐOẠN 0 — CHIA DỮ LIỆU (chạy 1 lần)
  split_builder.py  →  data/splits/<species>_split_v1.json  (train/calib/val/holdout + hash)

GIAI ĐOẠN 1 — PRETRAIN + LEAF GATE
  train_pretrain.py   →  checkpoints/backbone_pretrained.pt   (backbone 38 lớp, ảnh lab)
  build_leaf_gate_dataset.py  →  data/leaf_gate/...           (dựng tập leaf/not_leaf)
  train_leaf_gate.py  →  checkpoints/leaf_gate.pt             (model nhị phân lá/không lá)

GIAI ĐOẠN 2 — FINE-TUNE 5 LOÀI  (trái tim, mỗi loài chạy riêng)
  train_finetune.py  →  checkpoints/model_<species>.pt        (3 bước: warmup→full→decoupled)

GIAI ĐOẠN 3 — ĐÁNH GIÁ + CALIBRATION
  evaluate.py     →  classification report + confusion matrix (trên val)
  calibration.py  →  checkpoints/model_<species>_calibration.json  (nhiệt độ T)

GIAI ĐOẠN 4 — EXPORT + ĐÓNG GÓI
  export_onnx.py   →  model.onnx + class_mapping.json
  parity_test.py   →  parity_report.json  (chặn nếu ONNX lệch PyTorch)
  build_bundle.py  →  model_bundles/<version>/  (bundle bất biến + checksums)

GIAI ĐOẠN 4b — HOLDOUT (chạy 1 lần cuối)
  evaluate.py --split holdout  →  con số báo cáo chính thức

GIAI ĐOẠN 5 — PHỤC VỤ
  app.py + api_compat.py  →  web API + frontend PhytoScan (tải bundle từ HF Hub)

NGOÀI LỀ — CROSS-TEST (so với nghiên cứu công bố)
  cross_test/run_direction2.py  →  reports/cross_test/...
```

### C.2. Dữ liệu "chảy" như thế nào qua các file (rất quan trọng)

Hiểu được luồng này là hiểu được cả dự án:

1. **Ảnh gốc** nằm trong `data/2. FINETUNE DATASET/<Loài>/<Lớp>/*.jpg`.
2. `split_builder.py` **không copy ảnh**, chỉ ghi **đường dẫn + nhãn + hash** vào manifest JSON, đánh dấu ảnh nào thuộc train/calib/val/holdout.
3. Khi train, `datasets.py` (class `FinetuneDataset`) đọc manifest, lấy danh sách `(đường_dẫn, nhãn)` của split cần dùng.
4. Với mỗi ảnh, `datasets.py` gọi `augmentation.py` để biến đổi ảnh (lật/xoay/đổi sáng...) rồi chuẩn hoá thành tensor.
5. `DataLoader` (của PyTorch) gom các tensor thành **batch**, đẩy lên GPU.
6. `train_finetune.py` đưa batch qua `backbone.py` (model) → ra logits → tính loss bằng `losses.py` → cập nhật trọng số qua `memory_utils.py` (AMP + accumulation).
7. Sau mỗi epoch, `evaluate.py` đo macro-F1 trên val → nếu tốt hơn thì lưu checkpoint.

### C.3. Vì sao có "class mapping" và nó xuyên suốt thế nào?

**class mapping** = ánh xạ giữa **chỉ số** (0,1,2...) và **tên lớp** ("Apple_scab"...). Model chỉ làm việc với số; con người cần tên. Nếu mapping lệch ở bất kỳ khâu nào → model đoán "0" nhưng ta hiểu nhầm thành bệnh khác → **thảm hoạ thầm lặng**.

Dự án đảm bảo mapping bất biến bằng cách:
- `split_builder.py` sinh mapping bằng `sorted()` tên thư mục → luôn cùng thứ tự.
- Lưu vào manifest → `train_finetune.py` đọc từ đó (không tự suy).
- Lưu vào checkpoint (`classes`) → `export_onnx.py` ghi vào `class_mapping.json` trong bundle.
- `app.py` đọc mapping **từ bundle** (không tự suy từ folder).

→ Cùng một thứ tự chạy suốt từ chia dữ liệu đến API.

### C.4. Thứ tự lệnh chạy (tóm tắt)

```bash
# Giai đoạn 0
python -m src.data.split_builder --species all --seed 42
# Giai đoạn 1
python -m src.train.train_pretrain  --config configs/pretrain_effv2s.yaml
python -m src.data.build_leaf_gate_dataset --per-species 700 --seed 42   # + tự thêm ảnh not_leaf
python -m src.train.train_leaf_gate --config configs/leaf_gate.yaml
# Giai đoạn 2 (lặp 5 loài)
python -m src.train.train_finetune  --config configs/finetune_apple.yaml
# ... corn, grape, pepper_bell, tomato
# Giai đoạn 3
python -m src.eval.evaluate    --config configs/finetune_tomato.yaml --split val
python -m src.eval.calibration --config configs/finetune_tomato.yaml --target species
# Giai đoạn 4
python -m src.export.export_onnx  --config configs/finetune_tomato.yaml --out model_bundles/tmp/tomato/model.onnx
python -m src.export.parity_test  --bundle model_bundles/tmp/tomato/ --pt-ckpt checkpoints/model_tomato.pt --golden-set data/golden/tomato.json
python -m src.export.build_bundle --species tomato --version <ver> --onnx model_bundles/tmp/tomato/model.onnx
# Giai đoạn 4b
python -m src.eval.evaluate --config configs/finetune_tomato.yaml --split holdout
# Giai đoạn 5
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

---
## Phần D — Đọc code theo từng file

> Mỗi file: **vai trò → từng thành phần bên trong (nhận gì, làm gì từng bước, trả gì) → ai gọi nó**.

---

### D0. `config.py`
📁 [`src/config.py`](src/config.py) — **Đọc file cấu hình YAML.**

🧠 **Bối cảnh:** mọi siêu tham số (batch size, learning rate, số epoch, đường dẫn...) không viết cứng trong code mà để trong file `.yaml` (định dạng text dễ đọc). Muốn đổi thí nghiệm chỉ sửa file YAML, không sửa code. File này chỉ có nhiệm vụ **đọc YAML thành dict Python**.

🔧 **class `TrainConfig`** — một dataclass bọc đúng 1 thứ: `raw` (dict chứa toàn bộ nội dung YAML).
- `load(path)` (classmethod): mở file, gọi `yaml.safe_load` (parse an toàn — không cho YAML chèn code Python độc hại), trả về `TrainConfig(raw=...)`.
- `__getitem__(key)`: cho phép viết `cfg["train"]` thay vì `cfg.raw["train"]` — trả `self.raw[key]`.
- `get(key, default)`: truy cập an toàn khối tuỳ chọn (VD `cfg.get("domain_aug", {})` không lỗi nếu thiếu key).

💬 *"Sao không dùng Hydra/OmegaConf?"* → Dự án cá nhân quy mô nhỏ, một loader YAML tối giản là đủ, tránh phụ thuộc nặng.

**Ai gọi:** mọi script train/eval/export đều bắt đầu bằng `TrainConfig.load(config_path)`.

---

### D1. `split_builder.py`
📁 [`src/data/split_builder.py`](src/data/split_builder.py) — **Giai đoạn 0: chia dữ liệu + khoá bằng hash.**
**Chạy:** `python -m src.data.split_builder --species all --seed 42`

🧠 **Vai trò:** với mỗi loài, quét thư mục ảnh → chia mỗi lớp thành 70/10/10/10 (train/calib/val/holdout) một cách **xác định (deterministic)** → ghi manifest JSON kèm hash SHA256 mỗi ảnh. Đây là "nguồn sự thật" cho mọi bước sau.

**Các hằng số đầu file:**
- `SPECIES` = 5 loài (key nội bộ: apple, corn, grape, pepper_bell, tomato).
- `DATASET_ROOT` = đường dẫn tới `2. FINETUNE DATASET`.
- `RATIO = {train:0.70, calib:0.10, val:0.10, holdout:0.10}`.
- `SPECIES_DIR_NAME` = ánh xạ key nội bộ → tên thư mục thật (`corn` → `Corn_(maize)`), vì tên thật có dấu ngoặc/phẩy không tiện làm key.

🔧 **`resolve_species_dir(species)`** — tìm thư mục loài trên đĩa.
- Thử tên trong `SPECIES_DIR_NAME`; nếu thư mục tồn tại → trả về.
- Nếu không, hàm con `norm()` chuẩn hoá tên (bỏ `_`, `,`, `(`, `)`, khoảng trắng, hạ thường) rồi dò khớp không phân biệt hoa/thường → bền với khác biệt tên.

🔧 **`sha256_file(path, chunk=1MB)`** — băm nội dung file.
- Mở file nhị phân, đọc từng khối 1MB (tránh nạp cả file lớn vào RAM), cập nhật bộ băm SHA256, trả chuỗi hex 64 ký tự.
- 🧠 *Tác dụng:* hash là "vân tay" của ảnh — dùng để phát hiện 2 ảnh trùng nội dung (dù khác tên) và để đảm bảo tái lập.

🔧 **`list_class_images(species_dir)`** — liệt kê ảnh theo lớp.
- Lấy các thư mục con (mỗi thư mục = 1 lớp bệnh), **`sorted()`** để thứ tự ổn định.
- Với mỗi lớp, lấy mọi file `.jpg/.jpeg/.png`, cũng `sorted()`.
- Trả `{tên_lớp: [danh sách đường dẫn]}`.
- 🧠 *Vì sao sort?* Vì `class_index` sinh ra từ thứ tự này — sort đảm bảo lần nào chạy cũng ra **cùng** mapping.

🔧 **`stratified_split(paths, rng)`** — chia 1 lớp thành 4 phần.
- `rng` = bộ sinh ngẫu nhiên đã gắn seed (để tái lập). `paths[:]` copy để không sửa list gốc.
- `rng.shuffle(paths)` — xáo trộn.
- Tính số lượng: `n_calib = round(n×0.1)`, `n_val`, `n_hold` tương tự, phần còn lại là `n_train` (số lẻ dồn vào train).
- Cắt list theo thứ tự train→calib→val→holdout. Trả dict `{split: [paths]}`.

🔧 **`kfold_split(paths, rng, k=5)`** — chỉ cho lớp <30 ảnh (hiện KHÔNG có lớp nào). Chia đều vào k fold để đánh giá few-shot.

🔧 **`build_species(species, seed, rare_threshold=30)`** — hàm chính, xử lý 1 loài.
1. Tạo `rng = random.Random(seed)`.
2. `resolve_species_dir` + `list_class_images`.
3. **`class_to_idx = {tên lớp: chỉ số}`** bằng `enumerate(sorted(...))` → mapping bất biến.
4. Duyệt từng lớp: nếu `len(paths) < rare_threshold` (30) → đẩy vào `rare_folds`; ngược lại → `stratified_split`, và với mỗi ảnh ghi một dict vào `items[]` gồm: `image_id`, `relative_path`, `sha256`, `class_name`, `class_idx`, `split`, `source_dataset`.
5. Gói tất cả vào `manifest` dict (thêm `schema_version`, `created_at`, `seed`, `dataset_root`, `species_dir`, `split_ratio`, `class_to_idx`, `rare_class_threshold`, `items`, `rare_folds`).
6. Ghi ra `data/splits/<species>_split_v1.json`.

**CLI (khối `if __name__ == "__main__"`):** nhận `--species` (mặc định "all"), `--seed` (42), `--rare-threshold` (30). Lặp qua các loài, gọi `build_species`.

💬 **Câu hỏi hay gặp:**
- *"Làm sao đảm bảo không rò rỉ holdout?"* → Mỗi ảnh có hash; holdout chỉ mở 1 lần cuối; nếu ảnh holdout lỡ trùng train sẽ phát hiện qua hash.
- *"Chạy lại có ra kết quả khác không?"* → Không, vì `seed=42` cố định + `sorted()` → hoàn toàn tái lập.
- *"Hiện có lớp hiếm không?"* → Không. Lớp nhỏ nhất (Spider_mites) có 55 ảnh > 30 → `rare_folds` rỗng ở mọi loài.

---
### D2. `datasets.py`
📁 [`src/data/datasets.py`](src/data/datasets.py) — **Nạp ảnh từ đĩa thành tensor cho model.**

🧠 **Bối cảnh:** PyTorch cần một đối tượng `Dataset` biết "có bao nhiêu ảnh" (`__len__`) và "lấy ảnh thứ i ra sao" (`__getitem__`). `DataLoader` sẽ gọi chúng để gom batch.

🔧 **`imread_bgr(path)` / `imread_rgb(path)`** — đọc ảnh an toàn với đường dẫn Unicode.
- Vấn đề: `cv2.imread` (OpenCV) **không đọc được** đường dẫn chứa ký tự non-ASCII (VD tên file tiếng Nhật) trên Windows.
- Giải pháp: `np.fromfile(path)` đọc file thành mảng byte → `cv2.imdecode` giải mã thành ảnh. Có kiểm tra file rỗng/hỏng → raise lỗi rõ ràng.
- `imread_rgb` = `imread_bgr` rồi đổi BGR→RGB (OpenCV đọc theo thứ tự BGR, phần còn lại của pipeline dùng RGB).

🔧 **class `PretrainDataset(Dataset)`** — cho Giai đoạn 1 (đọc theo thư mục).
- `__init__(root_dir, transform, domain_aug)`: quét thư mục, mỗi thư mục con là 1 lớp, tạo `class_to_idx` (sorted), gom list `samples = [(đường_dẫn, nhãn)]`. `transform` = augmentation; `domain_aug` = ghép nền field (chỉ dùng cho train).
- `__len__`: trả số ảnh.
- `__getitem__(idx)`: đọc ảnh BGR → nếu có `domain_aug` thì ghép nền → đổi RGB → áp `transform` → trả `(tensor_ảnh, nhãn)`.

🔧 **class `FinetuneDataset(Dataset)`** — cho Giai đoạn 2 (đọc từ manifest).
- `__init__(manifest, split, transform)`: lấy `class_to_idx` **từ manifest** (không tự suy từ folder → đảm bảo mapping khớp), lọc `items` theo `split` cần dùng, tạo `samples = [(đường_dẫn_tuyệt_đối, class_idx)]`.
- `__getitem__(idx)`: đọc ảnh RGB → áp `transform` → trả `(tensor, nhãn)`.

💬 *"Vì sao 2 class Dataset khác nhau?"* → Pretrain đọc theo thư mục (đơn giản, 38 lớp); Finetune đọc theo manifest (để dùng đúng split train/val/holdout đã khoá và mapping bất biến).

**Ai gọi:** `train_pretrain.py` dùng `PretrainDataset`; `train_finetune.py` + `evaluate.py` + `calibration.py` dùng `FinetuneDataset`.

---

### D3. `augmentation.py`
📁 [`src/data/augmentation.py`](src/data/augmentation.py) — **Tăng cường dữ liệu (biến đổi ảnh khi train).**

🧠 **Vì sao cần:** ảnh train có hạn. Bằng cách lật/xoay/đổi sáng/thêm nhiễu, ta tạo vô số "biến thể" từ 1 ảnh → model học đa dạng hơn, **chống overfit** và **quen với ảnh field xấu** (mờ, nghiêng, thiếu sáng). Dùng thư viện **albumentations** (thư viện augmentation ảnh nhanh).

Hằng số: `IMAGENET_MEAN=(0.485,0.456,0.406)`, `IMAGENET_STD=(0.229,0.224,0.225)` — giá trị chuẩn hoá pixel theo thống kê ImageNet (vì backbone pretrain trên ImageNet nên input phải chuẩn hoá cùng cách).

🔧 **`get_finetune_transforms(train=True, img_size=224, strong=False)`** — augmentation cho fine-tune.
- Nếu `train=True`, chuỗi biến đổi (áp ngẫu nhiên theo xác suất `p`):
  - `RandomResizedCrop(scale=(0.7,1.0))` — cắt ngẫu nhiên 70-100% ảnh rồi resize về 224. Mô phỏng khoảng cách chụp khác nhau.
  - `HorizontalFlip(p=0.5)`, `VerticalFlip(p=0.3)` — lật ngang/dọc (lá không có "chiều đúng").
  - `Affine(scale, translate, rotate=±25°, shear)` — xoay/dịch/nghiêng nhẹ (mô phỏng cầm máy nghiêng).
  - `ColorJitter` + `RandomBrightnessContrast` — đổi sáng/tương phản/màu (mô phỏng ánh sáng field).
  - `GaussNoise` — thêm nhiễu hạt (mô phỏng ISO cao / máy yếu).
  - `ImageCompression(quality 50-95)` — giả lập ảnh nén JPEG khi upload web.
  - Nếu `strong=True`: thêm `CoarseDropout` (đục vài ô vuông) — chỉ cho lớp rất ít ảnh.
  - Cuối: `Normalize(ImageNet) + ToTensorV2` — chuẩn hoá + đổi thành tensor PyTorch `[C,H,W]`.
- Nếu `train=False` (val/test): **chỉ** `Resize + Normalize + ToTensor` — **không** augment (để đo khách quan).

🔧 **`get_pretrain_transforms(...)`** — nhẹ hơn (ảnh lab nền sạch không cần augment mạnh): RandomResizedCrop + Flip + ColorJitter nhẹ + Normalize.

🔧 **class `CutMixMixupCollate`** — trộn ảnh ở cấp batch (dùng làm `collate_fn` của DataLoader).
🧠 *Lý thuyết:* **Mixup** (Zhang 2017) = trộn tuyến tính 2 ảnh: `ảnh_mới = λ·ảnh_A + (1-λ)·ảnh_B`, nhãn cũng trộn `λ·nhãn_A + (1-λ)·nhãn_B`. **CutMix** (Yun 2019) = dán một miếng cắt từ ảnh B vào ảnh A, λ = tỉ lệ diện tích. Cả hai buộc model không quá tự tin và học đặc trưng bền hơn.
- `__init__(num_classes, alpha=0.2, cutmix_prob=0.5, p=0.5)`.
- `__call__(batch)`:
  1. Stack ảnh + nhãn.
  2. Với xác suất `1-p`: trả nguyên (không trộn).
  3. `λ = Beta(alpha, alpha)` (số ngẫu nhiên trong [0,1]); `idx = hoán vị` (chọn "ảnh đối tác" để trộn).
  4. Với xác suất `cutmix_prob` → **CutMix**: tính vùng cắt kích thước `sqrt(1-λ)`, dán patch từ `imgs[idx]`, rồi **hiệu chỉnh lại λ theo diện tích thật**.
  5. Ngược lại → **Mixup**: `imgs = λ·imgs + (1-λ)·imgs[idx]`.
  6. Trả `(ảnh_trộn, nhãn_mềm = λ·y1 + (1-λ)·y2)`.
- ⚠️ *Quan trọng:* chỉ trộn trong **cùng 1 head** (pretrain 38 lớp chung head → OK). Không trộn xuyên loài.

💬 *"Vì sao nhãn phải 'mềm' sau CutMix/Mixup?"* → Vì ảnh giờ là hỗn hợp 2 lớp, nhãn cũng phải là hỗn hợp → cần loss xử lý nhãn phân phối (xem `soft_target_cross_entropy` và `ClassBalancedFocalLossSoft`).

**Ai gọi:** `train_pretrain.py`, `train_finetune.py`, `train_leaf_gate.py` (transform); CutMix/Mixup dùng trong pretrain + bước 2 finetune.

---

### D4. `domain_augmentation.py`
📁 [`src/data/domain_augmentation.py`](src/data/domain_augmentation.py) — **Ghép lá lab lên nền field (giảm domain gap).** ⚠️ *Mặc định TẮT — hướng mở rộng.*

🧠 **Ý tưởng (Jeon 2025):** ảnh lab có nền sạch, model dễ "gian lận" bằng cách nhìn nền thay vì lá. Nếu cắt lá ra và dán lên nền field ngẫu nhiên, model buộc phải nhìn **lá** → quen với nền lộn xộn của ảnh thật.

🔧 **`extract_leaf_mask(lab_image_bgr)`** — tách lá khỏi nền.
- Đổi ảnh sang xám (grayscale).
- **Otsu threshold** — thuật toán tự tìm ngưỡng sáng/tối tối ưu để tách foreground (lá) khỏi background (nền sạch). Trả mặt nạ nhị phân (mask): 255 = lá, 0 = nền.
- `morphologyEx(CLOSE)` — lấp các lỗ nhỏ trong mask.

🔧 **`recompose_on_field_background(lab_image, field_bg)`** — ghép.
- Lấy mask, resize nền field về cùng kích thước.
- **Alpha-blend:** `ảnh_mới = lá×mask + nền_field×(1-mask)` → giữ nguyên lá, thay vùng nền.

🔧 **class `FieldBackgroundAugment`** — dùng trong `PretrainDataset.__getitem__`.
- `__call__(ảnh)`: với xác suất `p` (≈0.4) và có nền → ghép nền field ngẫu nhiên; ngược lại trả ảnh gốc.

**Ai gọi:** `train_pretrain.py` (nếu bật `domain_aug` trong config; hiện tắt).

---

### D5. `backbone.py`
📁 [`src/models/backbone.py`](src/models/backbone.py) — **Tạo model (backbone + head).**

🧠 **Bối cảnh:** dùng thư viện **timm** (PyTorch Image Models) — kho hàng trăm kiến trúc CNN/Transformer đã pretrain sẵn trên ImageNet. Chỉ cần gọi tên là có model + trọng số.

🔧 **`build_backbone(num_classes, arch="tf_efficientnetv2_s.in1k", pretrained=True, drop_path_rate=0.1)`**
- `timm.create_model(arch, pretrained=True, num_classes=num_classes, drop_path_rate=...)`:
  - `arch` = tên kiến trúc; mặc định EfficientNetV2-S đã pretrain ImageNet (`.in1k`).
  - `pretrained=True` → tải trọng số ImageNet (khởi đầu tốt).
  - `num_classes` → timm tự thay head cho đúng số lớp (VD 3 cho táo, 9 cho cà chua).
  - `drop_path_rate=0.1` → dropout dạng "tắt nhánh" chống overfit khi data ít.
- Trả về model PyTorch sẵn sàng train.

🔧 **`build_leaf_gate(pretrained=True)`** — tạo model Leaf Gate.
- `timm.create_model("mobilenetv3_small_100.lamb_in1k", num_classes=2)` — MobileNetV3-Small, 2 lớp (leaf/not_leaf).

💬 *"Backbone và head khác nhau chỗ nào?"* → Backbone = phần rút đặc trưng (dùng lại được); head = lớp cuối biến đặc trưng thành số lớp (thay mỗi khi đổi bài toán). timm gộp cả hai khi ta truyền `num_classes`.

**Ai gọi:** mọi script train, eval, export, cross-test.

📁 **Kèm theo — `energy_ood.py`** ([src/models/energy_ood.py](src/models/energy_ood.py)): chứa `energy_score(logits, T)` = `-T·logsumexp(logits/T)` — điểm năng lượng phát hiện OOD (ảnh lạ có năng lượng cao). Dùng trong `app.py` khi phục vụ.

---
### D6. `memory_utils.py`
📁 [`src/train/memory_utils.py`](src/train/memory_utils.py) — **Train tiết kiệm VRAM (AMP + gradient accumulation) + đo VRAM.**

🧠 **Đây là file giúp dự án chạy được trên 4GB.** Chứa vòng train chuẩn và hàm probe VRAM.

🔧 **`train_one_epoch_memory_efficient(model, loader, optimizer, criterion, device, accumulation_steps=4, use_amp=True, use_grad_checkpoint=False)`** — train đúng 1 epoch.
Giải thích từng bước trong vòng lặp:
1. `scaler = GradScaler(enabled=use_amp)` — bộ chống underflow FP16 (xem [B.6](#b6-nhóm-tiết-kiệm-bộ-nhớ-chạy-4gb-vram)).
2. `model.train()` — bật chế độ train (dropout/BatchNorm hoạt động khác lúc eval).
3. `optimizer.zero_grad()` — xoá gradient cũ.
4. Với mỗi batch `(imgs, labels)`:
   - Đưa lên GPU (`non_blocking=True` = chuyển bất đồng bộ cho nhanh).
   - `with autocast(dtype=float16)`: forward `logits = model(imgs)`, tính `loss = criterion(...) / accumulation_steps` — **chia trước** để khi cộng dồn 4 lô thì tổng gradient đúng bằng 1 batch lớn.
   - `scaler.scale(loss).backward()` — nhân loss lên rồi tính gradient (backward).
   - **Cứ mỗi `accumulation_steps` lô** mới: `scaler.step(optimizer)` (cập nhật trọng số), `scaler.update()` (điều chỉnh hệ số scale), `zero_grad()`.
   - Cộng dồn `loss.item() × accumulation_steps` để log giá trị loss thật.
5. Trả loss trung bình.

🧠 *Tóm ý gradient accumulation:* thay vì cập nhật sau mỗi 16 ảnh, ta đợi 4 lô (64 ảnh) mới cập nhật 1 lần → giống train batch 64 nhưng chỉ giữ 16 ảnh trong VRAM tại một thời điểm.

🔧 **`probe_vram_usage(model, sample_batch, device, batch_sizes=(4,8,16,32))`** — đo VRAM thật trước khi train.
- Với mỗi batch size thử:
  - `empty_cache()` + `reset_peak_memory_stats()` — dọn sạch, reset bộ đếm đỉnh.
  - Nhân bản 1 mẫu thành batch size đó, forward + backward trong autocast.
  - Ghi `max_memory_allocated / 1e9` → đỉnh VRAM (GB).
  - Nếu bắt `OutOfMemoryError` → ghi `"OOM"`, dọn cache, dừng.
- Trả `{batch_size: "x.xx GB" hoặc "OOM"}`.
- 🧠 *Tác dụng:* biết trước batch nào an toàn → train thật không bị crash giữa chừng (mất hàng giờ). Đây là bước **bắt buộc** đầu mỗi script train.

**Ai gọi:** `train_pretrain.py`, `train_leaf_gate.py` (cả 2 hàm); `train_finetune.py` (probe).

---

### D7. `losses.py`
📁 [`src/train/losses.py`](src/train/losses.py) — **Hàm mất mát xử lý mất cân bằng: Class-Balanced Focal Loss.**

🧠 **Nhắc lại vấn đề:** lớp 5555 ảnh vs lớp 55 ảnh. Loss thường (CE) sẽ để model thiên vị lớp đông. Giải pháp = kết hợp 2 ý tưởng: *cân theo số mẫu hữu hiệu* (Class-Balanced) + *tập trung mẫu khó* (Focal).

🔧 **`effective_num_weights(class_counts, beta=0.9999)`** — tính trọng số mỗi lớp.
Từng bước:
1. `counts` = tensor số ảnh mỗi lớp.
2. `eff_num = 1 - beta^counts` → **"số mẫu hữu hiệu"** phần mẫu số.
3. `weights = (1-beta)/eff_num` cho lớp có mẫu (lớp count=0 nhận 0 để tránh chia 0).
4. Chuẩn hoá: `weights / tổng × số_lớp_có_mẫu` → trung bình trọng số ≈ 1.

🧠 **Công thức & trực giác:**
```
E_n = (1 - β^n) / (1 - β)        # số mẫu hữu hiệu của lớp có n ảnh
trọng số lớp ∝ 1 / E_n = (1 - β) / (1 - β^n)
```
- β=0 → mọi lớp trọng số bằng nhau (bỏ qua số lượng).
- β→1 → trọng số ≈ 1/n (nghịch đảo tần suất).
- β=0.9999 (dùng trong dự án) → nằm giữa: lớp hiếm được ưu ái nhưng không quá đà.
- *Ý nghĩa "số mẫu hữu hiệu":* thêm ảnh vào lớp đã 5000 ảnh thì gần như trùng lặp (ít thông tin mới), nên "số hữu hiệu" bão hoà chứ không tăng tuyến tính.

🔧 **class `ClassBalancedFocalLoss(nn.Module)`** — cho nhãn nguyên (khi KHÔNG dùng CutMix/Mixup).
- `__init__`: lưu `weights` (buffer, đi theo `.to(device)` nhưng không phải tham số học), `gamma=2.0`, `label_smoothing=0.1`.
- `forward(logits, targets)`:
  1. `ce = F.cross_entropy(logits, targets, weight=w, label_smoothing=...)` — CE có class weight + label smoothing (giữ từng mẫu, chưa trung bình).
  2. `pt = exp(-ce)` — xác suất model gán cho lớp đúng (vì `ce = -log(p_t)`).
  3. `loss = ((1-pt)^gamma × ce).mean()` — nhân hệ số **Focal** `(1-pt)^γ` (mẫu dễ pt cao → hệ số nhỏ → bị hạ trọng số).

🔧 **class `ClassBalancedFocalLossSoft(nn.Module)`** — cho nhãn **mềm** (khi CutMix/Mixup bật → nhãn là phân phối).
- `forward(logits, soft_targets)`:
  1. Nếu `soft_targets` là nhãn nguyên (CutMix tắt ở batch đó) → tự chuyển thành one-hot.
  2. `log_probs = log_softmax(logits)`.
  3. `raw_ce = -(soft_targets × log_probs).sum(dim=1)` — soft cross-entropy từng mẫu.
  4. `sample_weight = (soft_targets × w).sum(dim=1)` — trọng số cho mẫu = tổ hợp trọng số các lớp theo tỉ lệ trộn.
  5. `loss = ((1-pt)^gamma × raw_ce × sample_weight).mean()`.
- 🧠 Khi nhãn mềm suy biến về one-hot thuần → công thức này trùng đúng bản gốc → nhất quán.

💬 *"Vì sao cần 2 phiên bản?"* → Vì CutMix/Mixup tạo nhãn dạng phân phối (0.7 A + 0.3 B), bản thường chỉ nhận nhãn nguyên. Bản Soft xử lý được cả hai.

**Ai gọi:** `train_finetune.py` khởi tạo `ClassBalancedFocalLossSoft` ở bước 1-2; `effective_num_weights` còn dùng trong `samplers.py`.

---

### D8. `samplers.py`
📁 [`src/train/samplers.py`](src/train/samplers.py) — **Lấy mẫu cân bằng cho bước 3 (decoupled).**

🔧 **`build_balanced_sampler(samples, class_counts, beta=0.9999)`**
1. `class_w = effective_num_weights(class_counts, beta)` — **dùng lại đúng công thức của loss** → nhất quán toàn dự án.
2. `sample_weights = [class_w[label] for (_, label) in samples]` — gán mỗi ảnh trọng số theo lớp của nó.
3. Trả `WeightedRandomSampler(sample_weights, num_samples=len, replacement=True)` — lấy mẫu **có hoàn lại** theo trọng số → lớp hiếm được rút nhiều lần hơn → mỗi batch cân bằng hơn.

🧠 *Khác biệt với class weight trong loss:* class weight (loss) = "phạt nặng khi sai lớp hiếm"; sampler = "cho lớp hiếm xuất hiện nhiều hơn". Bước 3 chỉ dùng **sampler**, không dùng class weight (tránh cân bằng gấp đôi — xem D12.4).

**Ai gọi:** `train_finetune.py` bước 3.

---

### D9. `peft.py`
📁 [`src/train/peft.py`](src/train/peft.py) — **Mở khoá tham số theo mức (tiết kiệm VRAM).**

🧠 **Nhắc lại:** full fine-tune (mở hết) tốn VRAM vì optimizer AdamW phải lưu 2 "moment" cho MỖI trọng số. PEFT chỉ mở một phần → optimizer chỉ lưu cho phần đó.

🔧 **`set_trainable_params(model, mode="head_bn_lastblock")`**
- Duyệt mọi `(name, param)` trong model, đặt `param.requires_grad` theo `mode`:
  - `"head_only"`: chỉ mở tên chứa `classifier` hoặc `fc` (chính là head).
  - `"head_bn_only"`: thêm `bn`/`norm` (các lớp BatchNorm — rẻ mà cải thiện đáng kể, tinh thần BitFit).
  - `"head_bn_lastblock"`: thêm `blocks.5`/`blocks.6` (1-2 block cuối backbone) — **mặc định bước 2**, nhiều năng lực học hơn.
  - khác: mở tất cả (`requires_grad=True`).
- Đếm và in `số tham số trainable / tổng` (%).
- Trả model.

🧠 **Vì sao mở "dần"?** Head mới khởi tạo ngẫu nhiên → nếu mở hết ngay từ đầu, gradient lớn từ head sẽ "phá" backbone tốt. Nên bước 1 chỉ warmup head, bước 2 mới mở thêm.

💬 *"blocks.5/blocks.6 là gì?"* → Tên các khối cuối trong kiến trúc timm. Block cuối học đặc trưng "cấp cao" (đặc thù bài toán) nên đáng mở nhất; block đầu học đặc trưng chung (cạnh, màu) → giữ nguyên.

**Ai gọi:** `train_finetune.py` gọi 3 lần (head_only → head_bn_lastblock → head_only) tương ứng 3 bước.

---
### D10. `train_pretrain.py`
📁 [`src/train/train_pretrain.py`](src/train/train_pretrain.py) — **Giai đoạn 1: pretrain backbone 38 lớp trên ảnh lab.**
**Chạy:** `python -m src.train.train_pretrain --config configs/pretrain_effv2s.yaml`

🧠 **Mục tiêu:** train EfficientNetV2-S trên 38 lớp bệnh (ảnh lab) → `checkpoints/backbone_pretrained.pt`. Đây là **điểm khởi đầu chung** cho cả 5 lần fine-tune sau, giúp fine-tune hội tụ nhanh và tốt hơn (vì backbone đã "biết" đặc trưng lá/bệnh).

🔧 **`set_seed(seed)`** — cố định seed cho `random`, `numpy`, `torch`, `torch.cuda` → tái lập.

🔧 **`soft_target_cross_entropy(logits, soft_targets, label_smoothing=0.0)`** — CE cho nhãn phân phối (khi CutMix/Mixup bật).
- `logp = log_softmax(logits)`.
- Nếu có label_smoothing: làm mềm target `soft_targets×(1-ls) + ls/n`.
- Trả `-(soft_targets × logp).sum(1).mean()`.

🔧 **`build_dataloaders(cfg)`** — dựng DataLoader train/val.
1. Lấy transform pretrain (train có augment, val không).
2. Nếu config bật `domain_aug` → tạo `FieldBackgroundAugment` (ghép nền field) chỉ cho train.
3. Tạo `PretrainDataset` train (có domain_aug) + val (không).
4. Nếu config bật `cutmix_mixup` → tạo `CutMixMixupCollate` làm `collate_fn`.
5. Trả `(train_loader, val_loader, soft)` với `soft` = có dùng nhãn mềm không.

🔧 **`train_pretrain(config_path)`** — hàm chính:
1. Load config, `set_seed`, chọn device.
2. `build_dataloaders`, `build_backbone(38 lớp)`.
3. **Probe VRAM (bắt buộc):** lấy 1 batch, `probe_vram_usage`; nếu OOM → raise với hướng dẫn.
4. Optimizer **AdamW** (`lr_head`, `weight_decay`), scheduler **CosineAnnealingLR** (T_max = số epoch).
5. Loss: nếu `soft` → `soft_target_cross_entropy`; nếu không → `CrossEntropyLoss` (đều có label_smoothing).
6. **Vòng epoch:** `train_one_epoch_memory_efficient(...)` → `sched.step()` → `f1 = evaluate(model, val)` → **nếu f1 > best: lưu checkpoint** `{model_state, arch, num_classes, val_macro_f1}`.

📊 **Kết quả thực tế:** backbone đạt **99.77% macro-F1 trên val lab**. Số này quan trọng ở Cross-test (chứng minh model không hỏng, chỉ sập vì đổi miền).

**Ai gọi:** chạy trực tiếp; `set_seed` được `train_finetune.py` import dùng lại.

---

### D11. Leaf Gate
📁 [`src/data/build_leaf_gate_dataset.py`](src/data/build_leaf_gate_dataset.py) + [`src/train/train_leaf_gate.py`](src/train/train_leaf_gate.py)

🧠 **Leaf Gate là "bảo vệ cửa"** — model nhị phân "lá / không phải lá" chặn ảnh rác trước khi vào model bệnh. Tránh việc upload ảnh con mèo mà model bệnh vẫn tự tin đoán "bệnh ghẻ táo 95%".

#### D11.1. Dựng dataset — `build_leaf_gate_dataset.py`
**Chạy:** `python -m src.data.build_leaf_gate_dataset --per-species 700 --seed 42`

- **`_collect_train_leaf_images(manifest, per_species, rng)`:** lấy ảnh `leaf` (positive) từ **split train** của 5 loài (chỉ train → **tránh rò rỉ** holdout/val vào leaf-gate). Round-robin qua các lớp để cân bằng, tối đa `per_species` ảnh/loài.
- **`_split_and_copy(...)`:** chia leaf thành 80/10/10 (train/calib/val), copy vào `data/leaf_gate/<split>/leaf/`.
- **`_split_not_leaf(src_dir, ...)`:** nếu bạn cung cấp thư mục ảnh `not_leaf` (tay, đất, đồ vật...), chia tương tự vào `<split>/not_leaf/`.
- ⚠️ Ảnh `not_leaf` **bạn phải tự thu thập** (dự án không có sẵn) — xem tiêu chí 7 nhóm trong [README mục 3](README.md).

#### D11.2. Train — `train_leaf_gate.py`
**Chạy:** `python -m src.train.train_leaf_gate --config configs/leaf_gate.yaml`

- **`CLASS_TO_IDX = {"not_leaf": 0, "leaf": 1}`** — ⚠️ index 1 = leaf, **cố ý** để khớp `app.py` đọc `softmax(...)[:, 1]` làm "xác suất là lá".
- **class `LeafGateDataset`:** đọc `{split}/{leaf,not_leaf}/`, gán nhãn nhị phân.
- **`eval_leaf_gate(model, loader)`:** trả 3 số — macro_f1, **leaf_recall** (mục tiêu ≥98%: không từ chối oan lá thật), **notleaf_recall** (mục tiêu ≥95%: chặn được rác).
- **`train_leaf_gate(config)`:** load config → tạo dataset (kiểm tra split không rỗng, đủ 2 lớp) → `build_leaf_gate(pretrained=True)` → probe VRAM → AdamW + CrossEntropyLoss(label_smoothing=0.05), **không scheduler** → vòng epoch, chọn checkpoint theo macro-F1 → lưu `checkpoints/leaf_gate.pt`.
- ⚠️ *Điểm code:* comment nói "cân bằng bằng class weight" nhưng **thực tế không** truyền `weight` vào loss (leaf/not_leaf thường không quá lệch nên chỉ dùng label_smoothing).

---

### D12. `train_finetune.py`
📁 [`src/train/train_finetune.py`](src/train/train_finetune.py) — **Giai đoạn 2: fine-tune từng loài. TRÁI TIM DỰ ÁN.**
**Chạy:** `python -m src.train.train_finetune --config configs/finetune_tomato.yaml`

🧠 **Chiến lược 3 bước** (dựa Kang 2019 — "tách học đặc trưng khỏi cân bằng phân loại"):

```text
Bước 1 WARMUP    : mở khoá HEAD (head_only)          → head làm quen dữ liệu field
Bước 2 FULL      : mở HEAD+BN+block cuối              → học đặc trưng sâu cho field
                   + Class-Balanced Focal + CutMix    + early stopping
Bước 3 DECOUPLED : đóng băng backbone, train lại HEAD → cân bằng lại bộ phân loại
                   + WeightedRandomSampler + CE thường
```

#### Các hàm hỗ trợ:

🔧 **`load_pretrained_backbone(model, ckpt_path)`** — nạp backbone 38 lớp vào model loài (VD 9 lớp).
- Vấn đề: head 38 lớp ≠ head 9 lớp → không khớp shape.
- Giải pháp: `filtered = {k:v for k,v in state.items() if k in model_state and v.shape khớp}` → **lọc theo shape**, tự bỏ head cũ, giữ backbone. Không hard-code tên → bền với mọi kiến trúc.

🔧 **`build_optimizer(model, lr_head, lr_backbone, weight_decay)`** — **discriminative LR**.
- Chia tham số thành 2 nhóm: head (`classifier`/`fc`) và backbone (còn lại).
- Head học nhanh (`lr_head=1e-3`), backbone học chậm (`lr_backbone=1e-5`) → không phá đặc trưng đã học.

🔧 **`build_regular_dataloaders(cfg, manifest)`** — dựng loader train/val từ manifest, **đếm `counts`** (số ảnh mỗi lớp) để tính Class-Balanced weight, và tạo CutMix/Mixup collate nếu bật.

🔧 **`train_one_epoch_finetune(...)`** — vòng train 1 epoch, có tích hợp **MMD domain loss tuỳ chọn**: nếu bật, mỗi batch lấy thêm 1 batch ảnh lab (source), tính đặc trưng 2 miền, cộng `lambda_mmd × mmd_loss` vào loss chính.

🔧 **`_train_phase(...)`** — chạy 1 "bước" (phase):
- Vòng epoch: train → `evaluate` val → nếu f1 > best (xuyên suốt cả 3 bước) → lưu checkpoint.
- **Early stopping:** biến `stall` đếm số epoch không cải thiện; nếu `stall >= patience` → dừng bước sớm.
- Trả `best_f1` (truyền tiếp sang bước sau).

#### 🔧 `train_finetune(config_path)` — hàm điều phối chính:
1. Load config, seed, device, manifest.
2. `build_regular_dataloaders` → có `classes`, `counts`.
3. `build_backbone(len(classes))` → `load_pretrained_backbone` (nạp backbone lab).
4. **Probe VRAM** → OOM thì raise.
5. Tạo `crit_cb = ClassBalancedFocalLossSoft(counts)`.
6. **Bước 1:** `set_trainable_params(model, "head_only")` → optimizer → `_train_phase(epochs_warmup)`.
7. **Bước 2:** `set_trainable_params(model, "head_bn_lastblock")` → (bật MMD nếu config) → `_train_phase(epochs_full, patience=early_stop_patience)`.
8. **Bước 3:** `set_trainable_params(model, "head_only")` (đóng băng backbone lại) → tạo `bal_loader` với `build_balanced_sampler` → `crit_ce = CrossEntropyLoss()` **THƯỜNG** → `_train_phase(epochs_decoupled)`.
9. **Chế độ 2 (few-shot):** kiểm tra `rare_folds`; hiện rỗng → in "bỏ qua".
10. Trả `best`.

💡 **D12.4 — Điểm code TINH TẾ (giảng viên hay soi):**
```python
# Bước 3: balanced sampler ĐI VỚI CE THƯỜNG (KHÔNG class weight).
crit_ce = nn.CrossEntropyLoss()
```
→ *Vì sao bước 3 không dùng Focal như bước 2?* Bước 3 đã cân bằng bằng **sampler** (oversampling lớp hiếm). Nếu thêm class weight vào loss nữa = **cân bằng gấp đôi** → head thiên về lớp hiếm quá mức → macro-F1 tụt. Đây đúng tinh thần Kang 2019: bước 3 chỉ "chỉnh lại" bộ phân loại một cách nhẹ nhàng.

💡 **D12.5 — Grape/Pepper (data nhỏ):** trong `configs/finetune_grape.yaml`, `lr_backbone=5e-6` (thấp hơn), tắt CutMix, ít epoch, patience nhỏ → chống overfit trên 204/268 ảnh. Không "đóng băng cứng" backbone mà cho học rất chậm.

---

### D13. few_shot & domain_loss
📁 [`src/train/few_shot.py`](src/train/few_shot.py) — **Prototype few-shot.** ⚠️ *Không chạy hiện tại (không còn lớp <30).*

🧠 **Ý tưởng (Snell 2017):** với lớp cực ít ảnh, không train head riêng (dễ overfit) mà tính **"đại diện" (prototype)** = trung bình đặc trưng của lớp đó, rồi phân loại ảnh mới bằng "gần prototype nào nhất".

🔧 **`compute_prototypes(support_features, support_labels, num_classes)`:**
- `F.normalize(features)` — **L2-normalize** (chia vector cho độ dài → mọi vector về "độ dài 1", để so bằng góc).
- Với mỗi lớp: prototype = normalize(trung bình các feature của lớp đó).

🔧 **`classify_by_prototype(query_features, prototypes)`:**
- Normalize query → `sim = query @ prototypes.T` = **cosine similarity** (vì đã normalize) → `softmax`.
- Prototype giống nhất = lớp dự đoán.

📁 [`src/train/domain_loss.py`](src/train/domain_loss.py) — **MMD loss.** ⚠️ *Mặc định tắt.*

🧠 **MMD (Maximum Mean Discrepancy):** đo "khoảng cách" giữa 2 phân phối (đặc trưng lab vs field). Minimize MMD → kéo 2 miền gần nhau → model bớt phụ thuộc miền.

🔧 **`mmd_loss(source_features, target_features, sigma=1.0)`:**
- Dùng **RBF kernel** `k(x,y) = exp(-||x-y||²/(2σ²))` (đo độ giống nhau, gần → gần 1, xa → gần 0).
- `MMD = mean(k(s,s)) + mean(k(t,t)) - 2·mean(k(s,t))`.
- = 0 khi 2 phân phối trùng nhau.

📁 [`src/train/checkpoints.py`](src/train/checkpoints.py) — **`save_resume_checkpoint(...)`**: lưu đầy đủ để train tiếp chính xác (trọng số + optimizer + scheduler + scaler + epoch + mapping + hash manifest/config + RNG state). ⚠️ Hiện train scripts lưu checkpoint tối giản, chưa dùng hàm này (resume là hướng mở rộng).

---
### D14. evaluate & calibration
📁 [`src/eval/evaluate.py`](src/eval/evaluate.py) — **Đo chất lượng model (macro-F1 + báo cáo).**
**Chạy:** `python -m src.eval.evaluate --config configs/finetune_tomato.yaml --split val`

🔧 **`collect_predictions(model, loader, device)`** — chạy model trên toàn tập, trả `(y_true, y_pred)` với `y_pred = logits.argmax(1)`. Dùng `torch.no_grad()` (không tính gradient → nhanh, tiết kiệm) + autocast FP16.

🔧 **`evaluate(model, loader, device, verbose=False, class_names=None)`** — hàm "thước đo" chung.
- Tính `macro_f1 = f1_score(y_true, y_pred, average="macro")`.
- Nếu `verbose=True`, in thêm:
  - **classification_report:** bảng Precision/Recall/F1 từng lớp + support (số ảnh).
  - **confusion_matrix:** ma trận nhầm lẫn (hàng=thật, cột=đoán).
- 🧠 Đây chính là hàm mà `train_pretrain`/`train_finetune` gọi mỗi epoch để chọn checkpoint.

🔧 **`evaluate_prototype(...)`** — biến thể cho nhánh few-shot (dùng chung metric macro-F1 để so công bằng).

🔧 **`_run_standalone(config, split)`** — chạy độc lập từ CLI: load checkpoint → dựng dataset đúng split → `evaluate(verbose=True)`. Có nhánh riêng cho leaf_gate (dùng ImageFolder thay manifest).

📁 [`src/eval/calibration.py`](src/eval/calibration.py) — **Hiệu chỉnh độ tin cậy (temperature scaling).**
**Chạy:** `python -m src.eval.calibration --config configs/finetune_tomato.yaml --target species`

🧠 **Vấn đề:** model nói "95%" nhưng thực ra chỉ đúng 80% → độ tin cậy "dối". Với hệ có từ chối, ngưỡng dựa trên độ tin cậy dối là vô nghĩa. Temperature scaling sửa việc này.

🔧 **class `TemperatureScaler(nn.Module)`:**
- `__init__(init_T=1.5)`: giữ tham số `log_T` (học log để đảm bảo `T = exp(log_T) > 0`).
- `forward(logits)`: trả `logits / T`.
- `fit(logits, labels, max_iter=50)`: dùng **L-BFGS** (optimizer mạnh cho ít tham số) tối ưu `T` để cross-entropy trên calib nhỏ nhất. 🧠 **Chỉ chia logits → không đổi argmax → không đổi accuracy**, chỉ chỉnh độ tin cậy.

🔧 **`expected_calibration_error(probs, labels, n_bins=15)`** — **ECE**:
- Lấy `conf, pred = probs.max(1)` (độ tin cậy + lớp đoán), `acc = (pred==labels)`.
- Chia [0,1] thành 15 thùng theo `conf`. Với mỗi thùng: cộng `(số mẫu/N) × |độ_đúng_TB - độ_tin_cậy_TB|`.
- Trả ECE (0 = hoàn hảo).

🔧 **`brier_score(probs, labels)`** — sai số bình phương giữa `probs` và one-hot. Càng nhỏ càng tốt.

🔧 **`calibrate_head(model, calib_loader, device)`:** thu logits trên calib → đo ECE/Brier **trước** → `fit` T → đo **sau** → in + trả dict `{temperature, ece_before, ece_after, brier_before, brier_after}`. Ghi ra `<ckpt>_calibration.json`.

⚠️ **Kết quả thực tế (trung thực):** ECE sau hiệu chỉnh **chưa đạt** ≤0.05 ở hầu hết loài (apple 0.094, corn 0.133 tệ hơn, tomato 0.084 tệ hơn, leaf-gate 0.74). → cần cải thiện; đặt ngưỡng từ chối bảo thủ hơn để bù. Tài liệu ghi rõ, không giấu.

📁 **Kèm — `ood_eval.py`** ([src/eval/ood_eval.py](src/eval/ood_eval.py), khung): `fpr_at_tpr` (FPR@95%TPR) + `ood_auroc` (AUROC) để đánh giá phát hiện OOD. `robustness.py` (khung): định nghĩa 5 loại nhiễu để đo model rớt bao nhiêu khi ảnh xấu.

---

### D15. export & bundle
📁 [`src/export/export_onnx.py`](src/export/export_onnx.py) — **Chuyển model PyTorch → ONNX.**

🔧 **`export_to_onnx(config, out, img_size=224, opset=17)`:**
- Load checkpoint PyTorch (trên CPU — không cần GPU khi export).
- `build_backbone(pretrained=False)` → nạp trọng số → `model.eval()` (tắt dropout/BatchNorm ở chế độ train).
- `torch.onnx.export(model, dummy_input, out, input_names=["input"], output_names=["logits"], dynamic_axes={batch linh hoạt}, opset=17)`.
- Ghi kèm `class_mapping.json` (idx→tên lớp, arch, img_size, species) cạnh model → API đọc mapping từ đây.
- 🔧 `quantize_dynamic_int8` / `convert_fp16` — tuỳ chọn nén (phải đo lại độ chính xác sau).

📁 [`src/export/parity_test.py`](src/export/parity_test.py) — **Kiểm tra ONNX == PyTorch.**

🔧 **`run_parity(pt_ckpt, onnx_path, batch, atol=1e-3)`:**
- Chạy cùng batch ảnh qua PyTorch **và** ONNX Runtime.
- `top1_match_rate` = tỉ lệ argmax giống nhau (phải =1.0).
- `max_abs_diff` = chênh logit lớn nhất (phải <atol).
- `passed = (top1 khớp 100% AND diff nhỏ)`. Nếu fail → `raise SystemExit` **chặn release**.
- 📊 Kết quả thực: tomato **pass**, max_diff ~6e-6 (cực nhỏ → ONNX chuẩn xác).

📁 [`src/export/build_bundle.py`](src/export/build_bundle.py) — **Đóng gói bundle bất biến.**

🔧 **`build_bundle(species, version, onnx_path, git_commit, ...)`:**
1. Copy bắt buộc: `model.onnx`, `class_mapping.json`. Ghi `preprocessing.json` (khoá img_size=224, resize=bilinear, mean/std ImageNet, RGB, NCHW).
2. Copy nếu có: `calibration.json`, `thresholds.json`, `prototypes.pt`, `metrics.json`, `parity_report.json`, `taxonomy.yaml`.
3. Ghi `manifest.json` (metadata: version, species, git_commit, danh sách file).
4. Ghi `checksums.sha256` (hash mọi file trong bundle → phát hiện nếu bị sửa).
5. **Gate:** nếu thiếu `model.onnx` hoặc `class_mapping.json` → raise, **chặn release**.

🧠 *Vì sao "bất biến"?* Để tránh tình huống "sửa 1 file lẻ → lệch mapping/preprocessing → model đoán sai mà không ai biết". Cả gói đi cùng nhau hoặc không đi.

---

### D16. Serving
📁 [`src/deploy/app.py`](src/deploy/app.py) — **Web API (FastAPI + ONNX Runtime).**
**Chạy:** `uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000`

🧠 **Điểm đặc biệt:** model được tải từ **Hugging Face Hub** (kho online), **lazy load** từng loài. Chỉ dùng ONNX Runtime CPU → không cần PyTorch/CUDA → nhẹ.

Hằng số: `REPO_MAP` (loài → repo HF `ruhua2305/<species>_effv2s_onnx`), `MAX_BYTES=8MB`, `MAX_PIXELS=12M`, `ALLOWED_MAGIC` (chữ ký JPEG/PNG).

🔧 **`_startup()`** (chạy khi server khởi động): chỉ tải Leaf Gate (~6MB) từ HF Hub (cần `HF_TOKEN`). Model loài chưa tải (để lazy).

🔧 **`_load_species_model(species)`** (lazy): tải bundle loài từ HF Hub lần đầu có request → nạp ONNX session + `class_mapping` + `preprocessing` + `calibration` + `thresholds` (nếu có).

🔧 **`validate_image_upload(raw)`** — kiểm tra an toàn:
- Dung lượng ≤8MB → nếu vượt: lỗi 413.
- **Magic bytes** khớp JPEG/PNG → nếu không: lỗi 415 (chống file giả đuôi).
- Số pixel ≤12M → chống **decompression bomb** (ảnh nén nhỏ nhưng giải nén khổng lồ).
- `ImageOps.exif_transpose` → xoay ảnh về đúng chiều theo EXIF.

🔧 **`predict(species, file)`** (endpoint `POST /predict`):
1. Lazy load model loài nếu chưa có.
2. Validate + tiền xử lý ảnh (khớp `preprocessing.json`).
3. **(1) Leaf Gate:** `p_leaf = softmax(leaf_logits/T)[1]`; nếu `< binary_leaf` → `{status: rejected, reason: not_a_leaf}`. Rồi tính `energy_score`; nếu `> T_energy` → `unknown_or_ood`.
4. **(2) Model loài:** `probs = softmax(logits/T)`; sắp xếp giảm dần. Nếu `top1 < T_low_confidence` → `low_confidence`; nếu `top1-top2 < T_margin` → `ambiguous`; ngược lại → `{status: ok, prediction, confidence, top3}`.

⚠️ **Trạng thái:** `thresholds.json` **chưa build** → chạy default (`binary_leaf=0.5`, `T_energy=1e9` ≈ tắt energy, `T_low_confidence=0`, `T_margin=0`). Nghĩa là nhánh `low_confidence`/`ambiguous`/energy-OOD **hiện chưa thực sự hoạt động** — là việc còn lại.

🔧 **`/health`, `/ready`** — kiểm tra server sống / đã nạp ≥1 model.

📁 [`src/deploy/api_compat.py`](src/deploy/api_compat.py) — **Lớp API cho frontend PhytoScan.**
- `SPECIES_I18N`, `DISEASE_I18N` — từ điển dịch tên loài/bệnh **song ngữ vi/en**.
- `/api/species`, `/api/diseases` — trả danh mục kèm nhãn dịch (cho dropdown).
- `/api/predict` — bọc `/predict`, đổi format + thêm `confidence_level` (High/Medium/Low).
- `/api/consult` — **chatbot tư vấn:** gọi LLM ngoài (mặc định Groq `llama-3.3-70b`) qua HTTP. Phân loại câu hỏi theo từ khoá (thuốc/phòng ngừa/triệu chứng/lây lan) để trả lời đúng trọng tâm. **Chỉ gửi text** (loài/bệnh/câu hỏi), không gửi ảnh/PII. Thiếu `LLM_API_KEY` → trả câu tĩnh an toàn.

⚠️ **Bảo mật:** `/predict` và `/api/*` **chưa có** auth/rate-limit → chỉ demo local, chưa mở Internet.

---

### D17. Cross-test
📁 [`src/cross_test/`](src/cross_test/) — **So model với nghiên cứu công bố.** Kết quả đầy đủ: [`reports/cross_test/CROSS_TEST_RESULTS.md`](reports/cross_test/CROSS_TEST_RESULTS.md)

🧠 **Câu hỏi:** "Ảnh field của mình khó tới đâu, fine-tune có đáng không?" **Chiều 2 (đã chạy):** lấy model chỉ-biết-lab chạy trên holdout field của mình.

🔧 **`label_maps.py`:** `canon(label)` chuẩn hoá tên (bỏ dấu ngoặc/phẩy, hạ thường) → map 20 lớp của mình sang index model công khai. `build_species_index_map` khớp theo tên chuẩn hoá (không hardcode số) → đúng cho cả model 38 lẫn 39 lớp.

🔧 **`public_models.py`:** 2 adapter cùng interface `predict_logits(batch) → logits`:
- `BackbonePVModel` — backbone 38 lớp của mình (offline).
- `DeiTPVModel` — model DeiT ngoài từ HF ([rescu/deit-...-plantvillage](https://huggingface.co/rescu/deit-base-patch16-224-finetuned-plantvillage)) qua `transformers`.

🔧 **`run_direction2.py`:** đọc holdout 5 loài → với mỗi model, đánh giá **closed-set per-species** (mỗi loài chỉ xét K lớp của loài đó → công bằng với head per-species của mình) → tính macro-F1 + accuracy + tỉ lệ đoán "lạc loài" → ghi report JSON.

📊 **Kết quả (macro-F1 trên holdout field):**

| Model | Train trên | macro-F1 | Accuracy |
|---|---|---:|---:|
| **Của mình (fine-tune field)** | field | **0.595** | 0.780 |
| DeiT-39 ngoài | PlantVillage lab | 0.479 | 0.559 |
| Backbone-38 của mình | PRETRAIN lab | 0.312 | 0.236 |

→ Model lab **thua rõ** model field → chứng minh domain gap + giá trị fine-tune. **Kiểm chứng:** backbone-38 đạt 99.77% trên val lab (không hỏng), chỉ sập vì đổi miền.

⚠️ **Chiều 1 (model mình → PlantDoc): chưa chạy** — vì FINETUNE đã chứa 742 ảnh PlantDoc → cần de-dup hash trước (tránh rò rỉ).

---

## Phần E — Câu hỏi giảng viên

| Câu hỏi | Trả lời |
|---|---|
| **Macro-F1 vs accuracy?** | Data lệch ~101 lần; accuracy bị lớp đông chi phối, macro-F1 (trung bình F1 không trọng số) phạt việc bỏ rơi lớp hiếm. |
| **Per-species vs 1 model 20 lớp?** | User đã chọn loài → không cần phân biệt xuyên loài; tách riêng giảm nhầm lẫn + mapping đơn giản. |
| **Chạy 4GB VRAM cách nào?** | AMP (FP16 giảm ½ bộ nhớ) + gradient accumulation (16×4=64 hiệu dụng) + PEFT (mở khoá một phần) + probe VRAM trước train. |
| **Class-Balanced Loss?** | Trọng số lớp theo "số mẫu hữu hiệu" (1−β^n)/(1−β) — thêm ảnh vào lớp lớn ít giá trị hơn lớp nhỏ. β=0.9999. |
| **Focal Loss?** | Nhân (1−p_t)^γ vào CE → hạ trọng số mẫu dễ (p_t cao), tập trung mẫu khó. γ=2. |
| **3 bước fine-tune?** | Kang 2019: tách học đặc trưng (bước 1-2) khỏi cân bằng bộ phân loại (bước 3); bước 3 dùng sampler + CE thường (không class weight, tránh cân bằng gấp đôi). |
| **Discriminative LR?** | Head học nhanh (1e-3), backbone học chậm (1e-5) để không phá đặc trưng pretrain. |
| **Temperature scaling giảm accuracy?** | Không. Chỉ chia logits cho T → không đổi argmax, chỉ chỉnh độ tin cậy. |
| **ECE là gì?** | Độ lệch giữa "độ tin cậy nói" và "độ đúng thật", chia theo thùng. Mục tiêu ≤0.05 (hiện chưa đạt — đã ghi rõ). |
| **Leaf Gate?** | Model nhị phân chặn ảnh không phải lá trước model bệnh, tránh đoán bừa tự tin. |
| **Energy OOD?** | Điểm E=-log Σe^z từ logits; ảnh lạ năng lượng cao → từ chối, không cần tập ảnh xấu riêng. |
| **Chia 70/10/10/10, khoá holdout?** | Tách calib (chỉnh ngưỡng) và holdout (test cuối) chống rò rỉ; hash SHA256 chống trùng + tái lập. |
| **ONNX để làm gì?** | Chạy không cần PyTorch/CUDA → serving nhẹ CPU; parity test đảm bảo không sai lệch. |
| **CutMix/Mixup?** | Trộn 2 ảnh + nhãn → model bớt tự tin, học đặc trưng bền; cần loss xử lý nhãn mềm. |
| **Cross-test chứng minh gì?** | Model lab (0.31-0.48) thua model field (0.60) trên ảnh thật → domain gap có thật + fine-tune đáng giá. |
| **Điểm chưa hoàn thiện?** | thresholds.json chưa build (reject chưa đủ), ECE chưa đạt ≤0.05, cross-test chiều 1 chưa chạy, cổng Dockerfile lệch (8080/8000). Đã ghi rõ. |

---

## Phần F — Tra nguồn

| Kỹ thuật/dữ liệu | Nguồn |
|---|---|
| EfficientNetV2 | [Tan & Le 2021 — arXiv:2104.00298](https://arxiv.org/abs/2104.00298) |
| MobileNetV3 | [Howard 2019 — arXiv:1905.02244](https://arxiv.org/abs/1905.02244) |
| Class-Balanced Loss | [Cui 2019 — arXiv:1901.05555](https://arxiv.org/abs/1901.05555) |
| Focal Loss | [Lin 2017 — arXiv:1708.02002](https://arxiv.org/abs/1708.02002) |
| Label smoothing | [Müller 2019 — arXiv:1906.02629](https://arxiv.org/abs/1906.02629) |
| Decoupling (3 bước) | [Kang 2019 — arXiv:1910.09217](https://arxiv.org/abs/1910.09217) |
| CutMix | [Yun 2019 — arXiv:1905.04899](https://arxiv.org/abs/1905.04899) |
| Mixup | [Zhang 2017 — arXiv:1710.09412](https://arxiv.org/abs/1710.09412) |
| RandAugment | [Cubuk 2019 — arXiv:1909.13719](https://arxiv.org/abs/1909.13719) |
| Mixed Precision (AMP) | [Micikevicius 2017 — arXiv:1710.03740](https://arxiv.org/abs/1710.03740) |
| Prototypical Networks | [Snell 2017 — arXiv:1703.05175](https://arxiv.org/abs/1703.05175) |
| Energy OOD | [Liu 2020 — arXiv:2010.03759](https://arxiv.org/abs/2010.03759) |
| Temperature scaling | [Guo 2017 — arXiv:1706.04599](https://arxiv.org/abs/1706.04599) |
| Field-bg recomposition | [Jeon 2025 — SSRN 5217081](https://dx.doi.org/10.2139/ssrn.5217081) |
| MMD domain adaptation | [Wu 2023 — Plant Phenomics](https://doi.org/10.34133/plantphenomics.0038) |
| PlantVillage | [Hughes & Salathé 2015 — arXiv:1511.08060](https://arxiv.org/abs/1511.08060) |
| Domain gap 99%→31% | [Mohanty 2016 — DOI](https://doi.org/10.3389/fpls.2016.01419) |
| PlantDoc | [Singh 2020 — arXiv:1911.10317](https://arxiv.org/abs/1911.10317) |
| Model DeiT cross-test | [rescu/deit-...-plantvillage](https://huggingface.co/rescu/deit-base-patch16-224-finetuned-plantvillage) |

---

*Tài liệu đồng bộ với code repo 2026-07-12. Xem thêm: thiết kế [`DAT301m_Plan_Final.md`](DAT301m_Plan_Final.md) · chạy [`README.md`](README.md) · cross-test [`reports/cross_test/CROSS_TEST_RESULTS.md`](reports/cross_test/CROSS_TEST_RESULTS.md).*
