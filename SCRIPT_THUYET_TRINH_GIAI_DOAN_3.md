# SCRIPT THUYẾT TRÌNH — GIAI ĐOẠN 3: PREPROCESSING AND TRAINING
## Dự án phân loại bệnh lá cây — PhytoScan

> Thành viên: Nguyễn Trọng Nhật Hoàng · Dương Minh Cường — GVHD: Trần Văn Hạ
>
> **Cách dùng script:** Mỗi phần có **[Lời nói]** để đọc gần như nguyên văn, và **[Ghi chú]** là số liệu/điểm phòng thủ khi thầy hỏi thêm. Nguyên tắc viết lại: **mọi thuật ngữ kỹ thuật đều được giải thích ngay khi xuất hiện lần đầu** — nói rõ *nó là gì, làm gì, vì sao cần* — để người nghe không có nền tảng vẫn theo kịp. Thời lượng gợi ý ghi ở đầu mỗi phần (tổng ~13–15 phút).

---

## Slide mở đầu — Giới thiệu (≈45 giây)

**[Lời nói]**
Em xin chào thầy và các bạn. Nhóm em gồm Nhật Hoàng và Minh Cường, hôm nay trình bày **Giai đoạn 3** của dự án phân loại bệnh cây trồng qua ảnh lá — sản phẩm tên là **PhytoScan**.

Nói ngắn gọn, mục tiêu của cả dự án là: người dùng chụp một chiếc lá bằng điện thoại, hệ thống trả về **cây đó đang bị bệnh gì**. Giai đoạn 3 là giai đoạn biến ý tưởng đó thành mô hình chạy được, gồm hai khối lớn:

1. **Tiền xử lý dữ liệu (preprocessing)** — chúng em có 11 bộ dữ liệu thô tải từ nhiều nguồn, rất lộn xộn. Khối này làm sạch, thống nhất nhãn, và tách ra đúng loại ảnh cần dùng.
2. **Huấn luyện mô hình (training)** — dạy máy tính nhận diện bệnh, theo chiến lược hai pha: học cái chung trước, rồi chuyên biệt hóa cho từng loài cây sau.

Em sẽ đi qua 7 phần theo đúng thứ tự dữ liệu chảy trong pipeline, và sẽ giải thích rõ từng thuật ngữ khi nó xuất hiện.

---

## Phần 1 — TỔNG QUAN PIPELINE (≈1.5 phút)

**[Lời nói]**
Trước khi đi vào chi tiết, em xin phác bức tranh tổng thể để thầy và các bạn dễ theo dõi. "Pipeline" ở đây nghĩa là **một dây chuyền xử lý**: dữ liệu đi vào ở một đầu, qua từng công đoạn, và ra sản phẩm ở đầu kia.

Điểm xuất phát là **dữ liệu thô: 11 bộ dataset bệnh cây** thu từ nhiều nguồn khác nhau. Vấn đề là mỗi nguồn đặt tên lớp bệnh một kiểu, cấu trúc thư mục mỗi nơi một khác — nên không thể dùng thẳng. Dữ liệu sẽ đi qua 4 công đoạn (stage):

- **Stage 1 — Gom và ánh xạ nhãn.** "Ánh xạ nhãn" nghĩa là quy mọi cách đặt tên khác nhau về **một chuẩn thống nhất**. Chuẩn chúng em chọn là kiểu PlantVillage, dạng `Loài___Bệnh`, ví dụ `Tomato___Late_blight` (cà chua — bệnh mốc sương). Để làm việc này dùng 3 chiến lược, em sẽ nói ở Phần 2.

- **Stage 2 — Phân loại lab/field bằng CLIP.** "Lab" là ảnh chụp trong phòng thí nghiệm, nền sạch trơn; "field" là ảnh chụp ngoài đồng ruộng thật, nền lộn xộn. Sản phẩm cuối phải chạy trên ảnh field, nên công đoạn này **tách riêng ảnh field ra**. Công cụ là CLIP — em sẽ giải thích ở Phần 3.

- **Stage 3 — Khử trùng lặp và chia dữ liệu.** "Khử trùng lặp (dedup)" là loại bỏ những ảnh bị lặp giữa các nguồn. Sau đó **chia dữ liệu** thành 4 phần theo tỷ lệ **70/10/10/10**, và "khóa" lại bằng một file manifest — em sẽ giải thích vì sao phải khóa.

- **Stage 4 — Huấn luyện 2 pha.**
  - **Pha 1 (pretrain):** học đặc trưng chung trên ảnh lab — 38 lớp, hơn 53.000 ảnh — tạo ra một "backbone" dùng chung.
  - **Pha 2 (finetune):** tinh chỉnh trên ảnh field, tạo ra **5 mô hình riêng cho 5 loài** (tổng cộng 20 lớp bệnh).

Cuối dây chuyền là **đánh giá, hiệu chỉnh độ tin cậy (calibration), và xuất mô hình sang định dạng ONNX** để triển khai, kèm một **Leaf Gate** — bộ lọc chặn ảnh không phải lá.

**[Ghi chú]**
- 5 loài đích: **apple (táo), corn (ngô), grape (nho), pepper_bell (ớt chuông), tomato (cà chua)**.
- Ràng buộc phần cứng xuyên suốt: card đồ họa **RTX 3050 Ti Laptop chỉ 4GB VRAM**. VRAM là bộ nhớ trên card đồ họa — 4GB là rất nhỏ để train. Đây là lý do gần như **mọi lựa chọn kỹ thuật** trong dự án đều nhằm tiết kiệm bộ nhớ. Nếu thầy hỏi "sao không dùng model to hơn" thì câu trả lời luôn quay về ràng buộc 4GB này.
- Triết lý xuyên suốt: ưu tiên một pipeline **có thể kiểm chứng (auditable)** hơn là chạy theo con số accuracy đẹp.

---

## Phần 2 — PREPROCESSING: GOM & ÁNH XẠ NHÃN (≈2 phút)

**[Lời nói]**
Phần này giải quyết công đoạn đầu tiên: từ 11 dataset hỗn loạn, làm sao có được nhãn thống nhất.

**Vấn đề cụ thể:** 11 nguồn, mỗi nguồn đặt tên lớp riêng và tổ chức dữ liệu khác nhau — có nguồn sắp xếp theo thư mục, có nguồn kèm file CSV mô tả nhãn, có nguồn dùng "bounding box" (khung chữ nhật đánh dấu vị trí lá trong ảnh). Cùng một bệnh có thể được nguồn A gọi "scab", nguồn B gọi "apple scab", nguồn C viết hoa khác đi. Nếu không thống nhất, máy sẽ hiểu nhầm là các bệnh khác nhau.

**Bước 1 — Chọn loài mục tiêu.** Chúng em không lấy tất cả loài, mà định nghĩa "candidate species" (loài ứng viên) là loài mà trong bộ chuẩn PlantVillage có **cả lớp khỏe mạnh (healthy) lẫn ít nhất một lớp bệnh**. Lý do: một mô hình phân loại bệnh chỉ có ý nghĩa khi phân biệt được "lá này khỏe hay bị bệnh". Lọc theo tiêu chí này ra đúng **5 loài**: táo, ngô, nho, ớt chuông, cà chua.

**Bước 2 — Ánh xạ nhãn về chuẩn `Loài___Bệnh` bằng 3 chiến lược**, tùy độ sạch của từng nguồn:

- **Chiến lược 1 — Hard-coded dict (từ điển gán cứng).** Với các nguồn tên rõ ràng, đáng tin như PlantDoc, Tomato-Village, NGLD, Merged: chúng em viết sẵn một bảng tra "tên gốc → tên chuẩn". Độ tin cậy cao nhất vì con người kiểm soát trực tiếp.

- **Chiến lược 2 — Đọc CSV / annotation (file mô tả nhãn).** Với Plant Pathology 2020–21, PlantSeg, Master: nhãn nằm trong file CSV kèm theo, chúng em đọc cột nhãn. Riêng ảnh "multi-label" (một ảnh gắn nhiều bệnh cùng lúc) thì **bỏ đi**, vì bài toán của chúng em là mỗi ảnh một nhãn duy nhất.

- **Chiến lược 3 — Keyword-based (dựa trên từ khóa).** Với FieldPlant, Master khi tên lộn xộn: chúng em tách tên loài bằng cách ghép 1–2 từ đầu, rồi **dò từ khóa bệnh** trong chuỗi tên như "scab", "rust", "blight"... để suy ra bệnh.

**Hai xử lý đặc biệt cần nhấn:**
- **FieldPlant** đánh dấu lá bằng bounding box trong CSV. Chúng em **cắt (crop) ảnh theo khung đó** để lấy đúng vùng lá, rồi resize về **224×224 pixel** — kích thước chuẩn đầu vào của mô hình.
- Lớp nào **không khớp** với bộ 38 lớp của PlantVillage thì chúng em **in cảnh báo và bỏ qua**, thay vì cố nhét vào — để tránh làm nhiễu nhãn.

**[Ghi chú]**
- Kết quả bước này: dữ liệu được gom về 2 nhánh (`Loài` và `Loài___Bệnh`); còn việc lọc lab/field để dành cho Stage 2.
- Nếu thầy hỏi "vì sao lấy chuẩn PlantVillage": vì đó là bộ nhãn đã chuẩn hóa, phổ biến trong giới nghiên cứu, và **khớp với backbone pretrain 38 lớp** mà chúng em dùng ở Pha 1 — thống nhất được đầu-cuối.
- Nếu hỏi "vì sao chỉ 5 loài": ràng buộc dữ liệu — chỉ 5 loài này có đủ cả healthy lẫn bệnh sau khi lọc.

---

## Phần 3 — DEDUP & PHÂN LOẠI LAB/FIELD, FRUIT/LEAF + AUGMENTATION (≈3.5 phút)

Phần này gồm 3 khối: (a) dùng CLIP tách ảnh field, (b) khử trùng lặp và chia dữ liệu, (c) tăng cường dữ liệu. Em xin đi từng khối.

### 3a. CLIP zero-shot — tách lab / field / fruit

**[Lời nói]**
Bài toán ở đây: chúng em cần tách ảnh chụp ngoài đồng (field) ra khỏi ảnh phòng lab, **mà không muốn ngồi gán nhãn tay cho hàng chục nghìn ảnh**. Giải pháp là dùng **CLIP**.

CLIP là một mô hình của OpenAI, đã được huấn luyện trên hàng trăm triệu cặp "ảnh — câu mô tả" từ internet. Nhờ đó nó hiểu được mối liên hệ giữa **hình ảnh và ngôn ngữ**: đưa cho nó một ảnh và vài câu mô tả, nó biết câu nào hợp với ảnh nhất. Bản chúng em dùng là **CLIP ViT-B/32**, khoảng 151 triệu tham số.

Điểm mấu chốt là dùng CLIP theo kiểu **"zero-shot"** — nghĩa là **không huấn luyện thêm gì cả**, dùng luôn kiến thức sẵn có của nó. Cách làm gọi là **Prompt Ensembling** (gộp nhiều câu mô tả):
- Chúng em soạn **5 câu mô tả bối cảnh lab**, **5 câu mô tả bối cảnh field**, và **4 câu mô tả ảnh quả (fruit)**. Lưu ý: các câu này mô tả **bối cảnh chụp**, không mô tả bệnh.
- CLIP biến mỗi câu thành một **vector** (một dãy số đại diện cho ngữ nghĩa). Chúng em lấy **trung bình các vector** trong mỗi nhóm — gọi là **centroid** (tâm nhóm). Vậy là có 3 centroid: tâm "lab", tâm "field", tâm "fruit".
- Với mỗi ảnh, CLIP cũng biến nó thành vector, rồi đo xem vector ảnh **gần centroid nào nhất** (bằng độ tương đồng cosine), và gán ảnh vào nhóm đó.

**Tối ưu quan trọng — định tuyến theo nguồn để đỡ tốn tính toán.** Không phải ảnh nào cũng cần chạy CLIP. Chúng em chia 11 dataset thành 3 nhóm xử lý:
- **ONFIELD** (PlantDoc, FieldPlant, Plant Pathology): đã chắc chắn là ảnh field → **copy thẳng, bỏ qua CLIP**.
- **LAB_SUSPEN** (Tomato-Village, NGLD, Master, Merged, TomatoDataset): nghi ngờ lẫn ảnh lab → **chạy CLIP để phân lab/field**.
- **FRUIT_SUSPEN** (PlantSeg): nghi lẫn ảnh quả → **loại ảnh quả trước**, phần lá còn lại mới xét tiếp lab/field.

Vì card chỉ có 4GB VRAM, khi chạy CLIP chúng em bật thêm 3 kỹ thuật tiết kiệm bộ nhớ: **FP16 autocast** (dùng số 16-bit thay vì 32-bit để giảm nửa bộ nhớ), **tự động dò batch size** (thử để tìm số ảnh xử lý cùng lúc lớn nhất mà không tràn), và **OOM recovery** — "OOM" là lỗi hết bộ nhớ, khi gặp thì hệ thống tự **chia đôi batch** rồi thử lại thay vì sập.

Đầu ra công đoạn này: thư mục `ONFIELD_LEAF_DETECTED` chứa ảnh field, `LAB_LEAF_DETECTED` chứa ảnh lab; ảnh quả bị loại bỏ hoàn toàn.

### 3b. Dedup & Split — khử trùng lặp và chia dữ liệu

**[Lời nói]**
Tiếp theo là **khử trùng lặp (dedup)**. Giữa các nguồn có thể có cùng một ảnh xuất hiện nhiều lần. Chúng em tính **hash SHA256** cho mỗi ảnh — hiểu đơn giản đó là một "dấu vân tay" dạng chuỗi ký tự, hai ảnh giống hệt sẽ cho cùng một hash. Nhờ đó phát hiện và loại ảnh trùng.

Sau đó là **chia dữ liệu (split)** thành 4 phần theo tỷ lệ **70/10/10/10**, mỗi phần một vai trò:
- **Train (70%)** — để mô hình học.
- **Calib (10%)** — tập riêng chỉ dùng để **hiệu chỉnh độ tin cậy và ngưỡng từ chối** sau này, không dùng để học.
- **Val (10%)** — để **chúng em chọn mô hình tốt nhất** và so sánh các cấu hình. Mô hình có "thấy" val nhưng không học trực tiếp trên đó.
- **Holdout (10%)** — tập **khóa kín, chỉ mở đúng một lần cuối** để báo cáo con số thật. Nó mô phỏng "dữ liệu tương lai mà mô hình chưa từng thấy".

Việc chia là **deterministic** — nghĩa là dùng một "seed" (số khởi tạo ngẫu nhiên) cố định là **42**, nên chạy lại bao nhiêu lần cũng ra đúng cùng một cách chia, đảm bảo tái lập được.

Vì sao phải **khóa bằng manifest**? Manifest là file ghi lại chính xác ảnh nào thuộc phần nào, kèm hash SHA256 của từng ảnh. Mục đích là chống **"data leakage" (rò rỉ dữ liệu)** — tình huống một ảnh ở tập holdout lỡ lọt vào tập train, khiến con số đánh giá đẹp giả tạo. Có manifest + hash, chúng em luôn kiểm được điều đó không xảy ra, và **holdout chỉ được chạy đúng một lần**.

### 3c. Augmentation — tăng cường dữ liệu

**[Lời nói]**
"Augmentation" là kỹ thuật **biến đổi ảnh train** (lật, xoay, đổi độ sáng, cắt...) để tạo ra nhiều "ảnh mới" từ ảnh cũ. Mục đích: cho mô hình thấy dữ liệu đa dạng hơn, chống hiện tượng **"overfit"** — tức là mô hình học vẹt tập train nhưng đoán kém trên ảnh mới. Lưu ý augmentation **chỉ áp cho tập train**, không áp cho val/holdout để việc đánh giá còn khách quan.

Chúng em dùng hai nhóm:
- **CutMix / Mixup** (với tham số alpha = 0.2): trộn hai ảnh lại với nhau và trộn nhãn tương ứng. Mixup pha mờ hai ảnh chồng lên nhau; CutMix cắt một mảnh ảnh này dán vào ảnh kia. Điều này buộc mô hình học chắc hơn, không quá tự tin.
- **Domain augmentation** (chỉ ở pha pretrain, xác suất p = 0.4): **ghép nền field vào ảnh lab**. Nhớ rằng ảnh lab nền sạch còn ảnh field nền lộn xộn — đây là "domain gap" (khoảng cách miền). Bằng cách dán lá lab lên nền đồng ruộng, mô hình quen dần với nền phức tạp, thu hẹp khoảng cách này ngay từ pha pretrain.

**Policy augment khác nhau theo từng loài** (vì mỗi loài có đặc thù riêng):
- **Apple (táo):** augment hình học + quang học chuẩn; MixUp là tùy chọn thử nghiệm vì dữ liệu đã nhiều.
- **Corn (ngô):** ưu tiên cân bằng lớp; hạn chế augment làm sai lệch bối cảnh của lớp Northern_Blight vì lớp này áp đảo (chiếm ~80%).
- **Grape & Pepper (nho, ớt):** augment bảo thủ, nhẹ tay, vì tập rất nhỏ — biến đổi mạnh dễ làm val bất ổn.
- **Tomato (cà chua):** tăng cường quang học để mô hình bền với thay đổi ánh sáng; lớp hiếm không augment quá đà để tránh tạo "đa dạng giả tạo".

**[Ghi chú]**
- Nếu hỏi "vì sao calib tách riêng khỏi val": để tránh rò rỉ — nếu dùng cùng một tập vừa chọn mô hình vừa hiệu chỉnh độ tin cậy thì con số sẽ thiên lệch.
- Nếu hỏi "SHA256 có chắc không": xác suất hai ảnh khác nhau trùng hash gần như bằng 0 nên đủ tin cậy để phát hiện trùng lặp chính xác.
- Điểm mạnh cần nhấn: toàn bộ khâu này làm pipeline **có thể audit** — manifest + hash + holdout khóa một lần.

---

## Phần 4 — PHASE 1: PRETRAIN (38 CLASSES) (≈2.5 phút)

**[Lời nói — mục tiêu]**
Bây giờ sang khâu huấn luyện. Trước khi giải thích, em xin làm rõ hai khái niệm nền tảng của **transfer learning** (học chuyển giao):
- **Pretrain (tiền huấn luyện):** huấn luyện mô hình trên một tập dữ liệu lớn trước, để nó học các đặc trưng chung — ví dụ cách nhận ra cạnh lá, đốm bệnh, kết cấu bề mặt.
- **Fine-tune (tinh chỉnh):** lấy mô hình đã pretrain đó, huấn luyện tiếp trên dữ liệu đích nhỏ hơn để chuyên biệt hóa cho từng loài.

Mục tiêu của Pha 1 là học đặc trưng chung về bệnh cây trên **38 lớp** của bộ PlantVillage — toàn bộ là ảnh lab (nền sạch) — để tạo ra một **backbone** dùng chung. "Backbone" (xương sống) là phần chính của mạng đã học "cách nhìn ảnh", có thể tái sử dụng qua nhiều bài toán; còn "head" (đầu) là lớp cuối cùng biến đặc trưng thành số lớp cụ thể.

**[Lời nói — mô hình & dữ liệu]**
Mô hình chúng em chọn là **EfficientNetV2-S**. Đây là một kiến trúc CNN — tức **mạng nơ-ron tích chập**, loại mạng chuyên xử lý ảnh bằng cách trượt các cửa sổ nhỏ khắp ảnh để dò đặc trưng cục bộ như cạnh, đốm, vết. Nó rất hợp với bài toán bệnh lá vì bệnh biểu hiện qua các đốm/vết cục bộ. Chữ "S" là Small, mô hình có khoảng **21 triệu tham số** (tham số = các con số bên trong mạng, huấn luyện chính là đi tìm bộ tham số tốt). Chúng em chọn EfficientNetV2-S vì nó mạnh nhất trong nhóm mô hình còn vừa được 4GB VRAM. Input ảnh là 224×224, train trên **53.599 ảnh lab, 38 lớp**.

**[Lời nói — kỹ thuật train để vừa 4GB VRAM]**
Vì chỉ có 4GB VRAM (bộ nhớ card đồ họa, nơi chứa mô hình, ảnh và gradient khi train), chúng em dùng ba kỹ thuật tiết kiệm bộ nhớ:

- **AMP — Automatic Mixed Precision (độ chính xác hỗn hợp):** bình thường số thực dùng 32-bit; AMP dùng 16-bit cho phần lớn phép tính, giảm khoảng một nửa bộ nhớ và nhanh hơn, gần như không mất độ chính xác.
- **Gradient accumulation (tích lũy gradient):** chúng em muốn batch 64 nhưng VRAM chỉ đủ batch 16. Giải pháp: chạy 4 lô nhỏ 16 ảnh, **cộng dồn gradient** qua cả 4 lô rồi mới cập nhật trọng số một lần. Kết quả tương đương batch 64 — gọi là "effective batch 64" — nhưng chỉ tốn bộ nhớ của batch 16. ("Gradient" là độ dốc cho biết cần chỉnh trọng số theo hướng nào để mô hình bớt sai.)
- Trước mỗi lần train, chúng em chạy hàm `probe_vram_usage()` để đo trước bộ nhớ, nếu sắp tràn thì dừng có cảnh báo thay vì crash giữa chừng.

Về cấu hình tối ưu: dùng **optimizer AdamW** (thuật toán cập nhật trọng số, tự điều chỉnh bước nhảy cho từng trọng số), **learning rate 1e-3** (learning rate là độ lớn bước nhảy mỗi lần cập nhật — quá lớn thì nhảy vọt, quá nhỏ thì học chậm), scheduler **CosineAnnealingLR** (giảm dần learning rate theo đường cong cos để hội tụ ổn định), train **30 epoch** (mỗi epoch là một lượt đi qua toàn bộ tập train), weight decay 1e-4 (phạt trọng số quá lớn, chống học vẹt).

Về nhãn, chúng em dùng **CutMix / Mixup** (alpha 0.2) kèm **Label Smoothing 0.1**. Label smoothing là kỹ thuật "làm mềm nhãn": thay vì dạy mô hình "chắc chắn 100% là lớp A", ta dạy "99% lớp A, rải 1% cho các lớp còn lại", giúp mô hình bớt tự tin thái quá. Vì nhãn giờ đã ở dạng phân phối (không còn là 0/1) nên phải dùng **soft-target cross-entropy** — phiên bản hàm mất mát xử lý được nhãn mềm.

**[Lời nói — lưu ý quan trọng]**
Một điểm cần nhấn: pha pretrain này **KHÔNG dùng class weight** (trọng số lớp cho lớp hiếm). Việc xử lý mất cân bằng lớp chúng em để dành hoàn toàn cho Pha 2 finetune, vì đó mới là nơi dữ liệu field thật sự lệch.

**[Lời nói — kết quả]**
Kết quả: **val macro-F1 đạt khoảng 0.995**, hội tụ rất nhanh — chỉ khoảng 7 epoch đã vượt 0.99. Em giải thích "macro-F1": ta tính chỉ số F1 cho **từng lớp** rồi lấy trung bình cộng — nghĩa là mọi lớp đóng góp ngang nhau, lớp hiếm cũng quan trọng như lớp đông. Đây là lý do chúng em dùng macro-F1 thay vì accuracy: với dữ liệu lệch, accuracy có thể cao giả tạo do đoán toàn lớp đông. Chúng em chọn checkpoint (bản lưu trọng số) có macro-F1 val cao nhất, lưu thành `backbone_pretrained.pt`.

**[Lời nói — vai trò backbone]**
Backbone này là **điểm khởi tạo chung** cho cả 5 model finetune. Sang Pha 2, chúng em nạp lại nó, **bỏ head 38 lớp cũ**, và thay head mới theo số lớp của từng loài. Việc backbone đạt tới 99.77% macro-F1 trên lab val chứng minh **backbone không hỏng** — nên nếu điểm số có tụt trên ảnh field thì nguyên nhân là **domain gap lab→field**, chứ không phải mô hình học kém.

**[Ghi chú]**
- Câu chốt phòng thủ: điểm 0.995 trên lab **không** phải điểm production. Nó chỉ là bằng chứng backbone đã học tốt đặc trưng lá/bệnh. Điểm thật của sản phẩm là điểm holdout field ở Pha 2.

---

## Phần 5 — PHASE 2: FINETUNE (5 MODELS) (≈3.5 phút)

**[Lời nói — mục tiêu & vì sao 5 model]**
Pha 2 là trọng tâm. Mục tiêu: tạo ra **5 model riêng biệt, mỗi loài một model**, chuyên biệt hóa cho ảnh field. Vì sao không dùng một model 20 lớp chung mà tách 5 model?
- Mỗi loài có tập bệnh riêng, tách model giúp mô hình **không bị nhầm chéo giữa các loài** — ví dụ không thể đoán một ảnh táo thành bệnh của cà chua.
- Mỗi model nhỏ, khi triển khai có thể **lazy-load** (nạp lười — chỉ tải model của loài nào khi có người dùng chọn loài đó), nhẹ và dễ triển khai.

Trong ứng dụng, người dùng chọn loài trước, nên hệ thống biết ngay phải dùng model nào.

**[Lời nói — quy trình 3 bước]**
Với mỗi loài, chúng em nạp `backbone_pretrained.pt`, thay head mới, rồi finetune theo **3 bước**. Xuyên suốt cả 3 bước, chúng em luôn lưu lại model tốt nhất theo macro-F1 trên val. Một chi tiết kỹ thuật: chúng em dùng **discriminative learning rate** — tức learning rate khác nhau cho từng phần: head (mới toanh, chưa biết gì) học nhanh với LR 1e-3; backbone (đã giỏi sẵn) học chậm với LR 1e-5 để không phá kiến thức cũ.

- **Bước 1 — Warm-up head:** chỉ train riêng head trong 3 epoch, backbone đóng băng. Mục đích để head làm quen trước, tránh gradient hỗn loạn ban đầu làm hỏng backbone. ("Đóng băng" = đặt cờ requires_grad = False, trọng số không được cập nhật.)
- **Bước 2 — Mở head + BN + block cuối:** mở khóa thêm các lớp BatchNorm và một hai khối cuối của backbone (đây là ý tưởng **PEFT — Parameter-Efficient Fine-Tuning**: chỉ train một phần nhỏ trọng số, phần còn lại đóng băng, để tiết kiệm VRAM). Ở bước này chúng em dùng **CB-Focal Loss** — tức Class-Balanced Focal Loss (beta 0.9999, gamma 2) — cùng CutMix và **early stopping** (patience 6, tức nếu val không cải thiện sau 6 epoch liên tiếp thì dừng). Em giải thích CB-Focal:
  - **Class-Balanced** cho lớp hiếm trọng số lớn hơn trong loss, dựa trên "số mẫu hữu hiệu" — ý tưởng là thêm một ảnh vào lớp đã đông thì ít giá trị hơn thêm vào lớp hiếm.
  - **Focal** hạ trọng số các mẫu dễ (đã đoán đúng tự tin) để mô hình dồn sức vào mẫu khó.
- **Bước 3 — Decoupled re-training (Kang 2019):** tách riêng việc học đặc trưng và việc cân bằng lại. Ở bước này chúng em dùng **balanced sampler** (WeightedRandomSampler — cho lớp hiếm cơ hội được lấy mẫu nhiều lần hơn để mỗi batch cân bằng) kết hợp **CE thường, không dùng CutMix**. Lý do dùng CE thường mà không thêm class weight: nếu vừa dùng balanced sampler vừa dùng class weight thì thành "cân bằng gấp đôi", head lệch mạnh về lớp thiểu số và macro-F1 tụt.

**[Lời nói — kết quả]**
Kết quả finetune 5 model:

| Loài | Val macro-F1 | Holdout macro-F1 (khóa 1 lần) |
|---|---|---|
| Apple | 0.86 | 0.84 |
| Corn | 0.61 | 0.59 |
| Grape | 0.64 | 0.49 |
| Pepper bell | 0.70 | 0.67 |
| Tomato | 0.53 | 0.39 |

Trung bình holdout 5 loài: **macro-F1 = 0.595**, accuracy nhóm (grouped accuracy) khoảng 0.78. Nhắc lại, "holdout" là tập khóa kín, chỉ mở đúng một lần ở cuối để báo con số thật — nó mô phỏng dữ liệu tương lai chưa từng thấy, nên đây mới là điểm số đáng tin của sản phẩm.

**[Lời nói — nhận xét]**
Nhận xét:
- **Apple tốt nhất** vì có nhiều ảnh field nhất; **grape và tomato thấp** vì ít mẫu hoặc bệnh khó phân biệt (các bệnh cà chua như Early_blight và Late_blight rất giống nhau về thị giác).
- Đáng chú ý nhất là **cross-test** (kiểm tra chéo): trên cùng tập holdout field, model finetune field của chúng em đạt **0.595**, trong khi một model chỉ biết ảnh lab — DeiT — chỉ đạt **0.479**, còn backbone lab của chính chúng em chỉ **0.312**. Con số này **định lượng rõ giá trị của việc finetune trên ảnh field thật**: nó chứng minh domain gap là có thật, và việc chúng em bỏ công tách ảnh field ở khâu tiền xử lý là xứng đáng.

**[Ghi chú]**
- Nếu thầy hỏi "sao tomato thấp thế": vì tomato có 9 lớp, nhiều bệnh nhìn gần giống nhau, và lớp Spider_mites chỉ có 55 ảnh. Chúng em báo per-class recall riêng thay vì giấu sau con số trung bình.
- Nếu hỏi "DeiT là gì": là một mô hình transformer thị giác được train sẵn trên PlantVillage — chúng em dùng nó làm mốc "model chỉ biết lab" để so sánh công bằng.
- Điểm trung thực cần giữ: chúng em **không thổi phồng** — báo cả điểm thấp (tomato 0.39) vì mục tiêu là pipeline audit được, không phải con số đẹp.

---

## Phần 6 — ĐÁNH GIÁ & XUẤT MODEL (≈2 phút)

**[Lời nói]**
Sau khi có 5 model, chúng em làm 4 việc để đánh giá và đóng gói cho triển khai.

**1. Đánh giá (evaluation).** Chúng em xuất **classification report** (bảng liệt kê precision, recall, F1 cho từng lớp) và **confusion matrix** (ma trận nhầm lẫn) trên tập val. Em giải thích nhanh:
- **Precision** trả lời: trong các ca mô hình đoán là bệnh X, bao nhiêu phần trăm đúng.
- **Recall** trả lời: trong các ca thật sự là bệnh X, mô hình bắt được bao nhiêu phần trăm.
- **Confusion matrix** là bảng vuông, hàng là lớp thật, cột là lớp đoán; nhìn các ô ngoài đường chéo lớn là biết mô hình hay nhầm cặp bệnh nào.

Tập **holdout vẫn chỉ chạy đúng một lần** ở khâu này để không rò rỉ.

**2. Calibration (hiệu chỉnh độ tin cậy).** Vấn đề: mô hình thường **tự tin thái quá** — nói "chắc 99%" nhưng thực tế chỉ đúng 90%. Calibration làm cho độ tin cậy khớp với độ đúng thật. Chúng em dùng **temperature scaling**: chia logits (đầu ra thô của mạng) cho một số T trước khi tính xác suất. T > 1 làm xác suất "mềm" ra, bớt tự tin. Ưu điểm: **không đổi thứ hạng nên không đổi accuracy**, chỉ chỉnh độ tin cậy. Chúng em học T trên tập calib và lưu ra `model_<species>_calibration.json`, kèm chỉ số **ECE** (Expected Calibration Error — thước đo độ lệch giữa tin cậy và đúng thật, càng gần 0 càng tốt) trước và sau khi hiệu chỉnh.

**3. Export ONNX + parity test.** **ONNX** là một định dạng model trung lập, chạy được bằng ONNX Runtime **mà không cần PyTorch hay CUDA** — nên triển khai nhẹ trên CPU. Nhưng khi chuyển đổi có thể phát sinh sai lệch nhỏ, nên chúng em chạy **parity test** (kiểm tra tương đương): so kết quả model PyTorch gốc với bản ONNX trên cùng bộ ảnh golden set. Chỉ **release nếu top-1 khớp 100%**.

**4. Build bundle (đóng gói triển khai).** "Bundle" là một thư mục gom **mọi thứ cần để phục vụ**: file `model.onnx`, `class_mapping.json` (bảng tra chỉ số lớp → tên bệnh), `manifest.json`, và `checksums.sha256` (mã băm từng file để phát hiện nếu bị sửa). Bundle này **bất biến** — đóng xong không sửa — và quá trình build sẽ **chặn lại nếu thiếu file**, tránh triển khai nhầm.

**[Ghi chú]**
- Phân biệt golden set và holdout khi thầy hỏi: **holdout** đo hiệu năng khoa học của mô hình; **golden set** là bộ ảnh nhỏ để smoke-test khâu triển khai — bắt lỗi mapping sai, ONNX lệch, load nhầm bundle.
- Trạng thái trung thực (nên chủ động nói): hiện **ECE chưa đạt** mục tiêu ≤ 0.05 ở hầu hết loài, và **ngưỡng reject** (`thresholds.json`) vẫn là việc còn lại. Chúng em ghi rõ điều này thay vì trình bày như đã hoàn thành.

---

## Phần 7 — TÓM TẮT & KẾT LUẬN (≈1.5 phút)

**[Lời nói]**
Em xin tóm tắt lại toàn bộ Giai đoạn 3.

**Các kỹ thuật chính đã dùng:**
- Ánh xạ nhãn **đa chiến lược** (từ điển gán cứng / đọc CSV / dò từ khóa) để thống nhất nhãn từ 11 dataset.
- **CLIP zero-shot** (ViT-B/32) tách ảnh lab / field / fruit bằng centroid prompt ensembling, không cần gán nhãn tay.
- **Khử trùng lặp + chia dữ liệu khóa bằng SHA256**, chống rò rỉ, holdout chỉ chạy một lần.
- **AMP + gradient accumulation + probe VRAM** để huấn luyện vừa trong 4GB.
- **Transfer learning 2 pha:** pretrain 38 lớp lab → finetune 5 loài trên ảnh field (quy trình 3 bước).
- **Xử lý mất cân bằng:** CB-Focal Loss + balanced sampler (decoupled) + CutMix + label smoothing.

**Kết quả đạt được:**
- Pipeline tiền xử lý hoàn chỉnh: gom → CLIP → dedup → split.
- Pretrain: val macro-F1 ≈ **0.995** trên 38 lớp lab.
- Finetune 5 model: holdout macro-F1 trung bình **0.595**, apple cao nhất **0.84**.
- Cross-test định lượng rõ domain gap; đã export ONNX + đóng bundle + có web demo chạy được.

**Hướng phát triển tiếp theo:**
- Thu thêm ảnh field cho các loài ít mẫu (grape, tomato) để nâng điểm.
- Cải thiện calibration (giảm ECE) và xây ngưỡng reject cho API.
- Chạy cross-test Chiều 1 trên PlantDoc sau khi de-dup SHA256.

Đó là toàn bộ phần trình bày của nhóm em. Em xin cảm ơn thầy và các bạn đã lắng nghe, và nhóm em sẵn sàng nhận câu hỏi ạ.

---

## PHỤ LỤC — Chuẩn bị hỏi đáp (không đưa lên slide)

> Câu trả lời viết ngắn để nhớ nhanh; khi trả lời nên diễn đạt tự nhiên.

**Q: Vì sao không dùng Hydra/OmegaConf để quản lý cấu hình?**
Dự án cá nhân quy mô nhỏ, một loader YAML tối giản là đủ, tránh phụ thuộc thư viện nặng.

**Q: Vì sao dùng 2 class Dataset khác nhau cho pretrain và finetune?**
Pretrain đọc ảnh theo thư mục cho đơn giản (38 lớp chung một head). Finetune đọc theo manifest để dùng đúng tập train/val/holdout đã khóa và class mapping bất biến — đảm bảo không lẫn dữ liệu.

**Q: Vì sao nhãn phải "mềm" sau CutMix/Mixup?**
Vì sau khi trộn, ảnh là hỗn hợp của hai lớp, nên nhãn cũng phải là hỗn hợp (ví dụ 0.7 lớp A + 0.3 lớp B). Cần hàm loss xử lý được nhãn phân phối — đó là soft-target cross-entropy.

**Q: Backbone và head khác nhau chỗ nào?**
Backbone là phần rút đặc trưng, tái sử dụng được qua nhiều bài toán. Head là lớp cuối biến đặc trưng thành số lớp cụ thể, phải thay mỗi khi đổi bài toán. Đây chính là cơ chế cho phép transfer learning: giữ backbone, thay head.

**Q: blocks.5 / blocks.6 trong PEFT là gì?**
Là tên các khối cuối của EfficientNetV2 trong thư viện timm. Khối cuối học đặc trưng cấp cao, đặc thù bài toán, nên đáng mở khóa nhất; khối đầu học đặc trưng chung (cạnh, màu) nên giữ nguyên — vừa tiết kiệm VRAM vừa hiệu quả.

**Q: Vì sao CLIP tách được lab/field mà không cần train?**
CLIP đã học chung một không gian ảnh–text trên hàng trăm triệu cặp, nên nó "hiểu" khái niệm "ảnh chụp trong lab" so với "ảnh ngoài đồng" thông qua câu prompt. Đây là zero-shot: không train thêm, chỉ so độ tương đồng cosine giữa vector ảnh và centroid của các nhóm prompt.

**Q: Vì sao chọn EfficientNetV2-S mà không phải mô hình lớn hơn?**
Ràng buộc 4GB VRAM. EffV2-S (~21M tham số) là backbone cho độ chính xác cao nhất mà vẫn chạy được trên 4GB nhờ AMP + gradient accumulation + PEFT. Model lớn hơn như EffV2-M hay Swin-S sẽ gây tràn bộ nhớ (OOM).

**Q: Holdout 0.595 có phải là thấp không?**
Đây là điểm trên ảnh field thật, khóa và chỉ chạy một lần — khắt khe hơn nhiều so với accuracy kiểu PlantVillage (thường >0.99 nhưng chỉ trên ảnh lab nền sạch). Cross-test cho thấy model của nhóm (0.595) vượt xa model chỉ biết lab (0.31–0.48), chứng minh finetune trên field có giá trị thật.

**Q: Làm sao chắc chắn không rò rỉ dữ liệu giữa train và holdout?**
Mọi ảnh được hash SHA256 và ghi trong manifest; việc chia là deterministic với seed 42. Nếu phát hiện một ảnh holdout xuất hiện trong train/val thì experiment đó bị vô hiệu và phải chạy lại với manifest mới.

**Q: Vì sao Bước 3 dùng CE thường mà không dùng class weight?**
Vì Bước 3 đã dùng balanced sampler để cân bằng lại dữ liệu. Nếu cộng thêm class weight thì thành "cân bằng gấp đôi", head sẽ lệch quá mức về lớp thiểu số và làm macro-F1 tụt.

**Q: Macro-F1 khác accuracy chỗ nào, vì sao chọn macro-F1?**
Accuracy là tỉ lệ đoán đúng tổng thể — với dữ liệu lệch, đoán toàn lớp đông vẫn ra accuracy cao, che giấu lỗi ở lớp hiếm. Macro-F1 tính F1 từng lớp rồi lấy trung bình cộng, mọi lớp quan trọng như nhau, nên phản ánh trung thực hơn khi dữ liệu mất cân bằng.
