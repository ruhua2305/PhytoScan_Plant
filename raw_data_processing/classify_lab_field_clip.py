#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_lab_field_clip.py
============================================================================
Pipeline CLIP zero-shot để phân loại ảnh bệnh cây "lab" (nền đơn sắc, chụp
trong studio/phòng thí nghiệm) và "field" (chụp ngoài đồng, nền tự nhiên
phức tạp), sau đó copy ảnh field vào thư mục đích, giữ nguyên cấu trúc
Loài/Bệnh/. Không sửa/xóa file gốc.

Toàn bộ các lựa chọn thiết kế (model, prompt, threshold, batch size...)
được giải thích chi tiết trong báo cáo đi kèm (bao_cao_pipeline_CLIP_lab_field.md).
Ở đây chỉ tóm tắt các điểm quan trọng nhất trong comment.

Yêu cầu cài đặt:
    pip install torch --index-url https://download.pytorch.org/whl/cu121   (bản có CUDA, tùy driver)
    pip install open_clip_torch tqdm pillow

Chạy thử nhanh (20 ảnh đầu, không copy, chỉ xem log):
    python classify_lab_field_clip.py --limit 20 --dry-run

Chạy thật:
    python classify_lab_field_clip.py ^
        --src "D:\\DAT301m_Project\\data\\Field_Raw" ^
        --dst-field "D:\\DAT301m_Project\\data\\Field_Clean" ^
        --dst-lab "D:\\DAT301m_Project\\data\\Lab_Detected"
============================================================================
"""

import argparse
import csv
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ảnh trong các bộ dữ liệu bệnh cây thu thập từ nhiều nguồn thường có một số
# file JPEG bị cắt cụt (truncated) do lỗi tải/nén. Nếu không bật cờ này,
# Pillow sẽ raise lỗi và làm dừng cả tiến trình chỉ vì một ảnh lỗi.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# 1. CẤU HÌNH PROMPT (xem lý giải chi tiết trong báo cáo, mục 3)
# ---------------------------------------------------------------------------
# Nguyên tắc thiết kế: mỗi prompt mô tả TRỰC TIẾP thuộc tính nền/bối cảnh
# (background/style) chứ không mô tả loại bệnh - vì đây là bài toán phân
# loại "kiểu chụp ảnh", không phải phân loại đối tượng. CLIP được huấn
# luyện trên caption tự nhiên trên web, vốn thường mô tả phong cách chụp
# ("studio shot", "close-up", "in the field", "outdoors"...), nên các cụm
# từ này nằm trong phân bố mà encoder văn bản của CLIP đã học tốt.
LAB_PROMPTS = [
    "a photo of a single plant leaf placed on a plain, solid-colored background",
    "a close-up macro photograph of a leaf against a uniform white or gray studio background",
    "a laboratory specimen photo of a detached leaf on a flat, plain surface, no soil or other plants visible",
    "a plant disease dataset image with a clean, monochromatic background and studio lighting",
    "a top-down photo of an isolated leaf on a table, with no natural outdoor scenery in the background",
]

FIELD_PROMPTS = [
    "a photo of a plant growing outdoors in a farm field",
    "a photo of crop leaves still attached to the plant, surrounded by soil, other leaves, and natural vegetation",
    "an outdoor agricultural photo with a complex, cluttered natural background",
    "a field photograph of a crop plant with visible ground, weeds, sky, or other plants in the background",
    "a photo taken in a garden or farm showing foliage among natural surroundings, not a studio background",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# 2. DATASET / DATALOADER
# ---------------------------------------------------------------------------
class ImagePathDataset(Dataset):
    """
    Dataset chỉ lưu ĐƯỜNG DẪN ảnh (chuỗi ký tự), KHÔNG load ảnh vào RAM lúc
    khởi tạo. Việc đọc + resize + chuẩn hóa ảnh chỉ diễn ra "lazily" bên
    trong __getitem__, được các worker process của DataLoader thực hiện
    song song. Đây là cách để thỏa yêu cầu "không tải toàn bộ ảnh vào RAM;
    xử lý tuần tự/streaming" khi làm việc với 93k+ ảnh trên máy chỉ có 16GB RAM
    (đôi khi bị các tab trình duyệt chiếm dụng).
    """

    def __init__(self, samples: List[Tuple[str, str, str]], preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, species, disease = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")  # theo đúng bước "_convert_image_to_rgb" trong preprocess gốc của OpenAI CLIP
                tensor = self.preprocess(img)
            return tensor, path, species, disease, True
        except Exception as e:  # ảnh lỗi/hỏng: không raise, đánh dấu ok=False để lọc ra ở collate_fn
            return None, path, species, disease, str(e)


def collate_skip_corrupt(batch):
    """
    Lọc bỏ các ảnh lỗi (ok=False) ra khỏi batch trước khi đưa vào GPU, để
    không lãng phí compute cho tensor rác và không làm crash torch.stack().
    Ảnh lỗi được trả về riêng để ghi log corrupt_images.csv.
    """
    good_tensors, good_paths, good_species, good_disease = [], [], [], []
    bad_records = []  # (path, error_message)

    for tensor, path, species, disease, ok in batch:
        if ok is True:
            good_tensors.append(tensor)
            good_paths.append(path)
            good_species.append(species)
            good_disease.append(disease)
        else:
            bad_records.append((path, ok))  # ok ở đây là error message (string)

    if good_tensors:
        tensors = torch.stack(good_tensors, dim=0)
    else:
        tensors = torch.empty(0, 3, 224, 224)

    return tensors, good_paths, good_species, good_disease, bad_records


# ---------------------------------------------------------------------------
# 3. THIẾT BỊ, BATCH SIZE TỰ ĐỘNG, FP16
# ---------------------------------------------------------------------------
def select_device(preferred: Optional[str], logger: logging.Logger) -> torch.device:
    """
    Tự động chọn cuda nếu khả dụng, fallback sang cpu nếu không, có thông báo
    rõ ràng để người dùng biết pipeline đang chạy ở chế độ nào.
    """
    if preferred is not None:
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("Yêu cầu dùng CUDA nhưng không phát hiện GPU khả dụng. Chuyển sang CPU.")
            device = torch.device("cpu")
        return device

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Phát hiện GPU: {name} ({total_vram_gb:.1f} GB VRAM). Sẽ dùng CUDA + FP16 (autocast).")
        return torch.device("cuda")

    logger.warning(
        "Không phát hiện GPU CUDA khả dụng. Chuyển sang chạy trên CPU (chậm hơn đáng kể). "
        "Với ~93k ảnh, hãy cân nhắc dùng --limit để test trước, hoặc chạy qua đêm."
    )
    return torch.device("cpu")


def autotune_batch_size(
    model,
    device: torch.device,
    logger: logging.Logger,
    candidates: Tuple[int, ...] = (64, 48, 32, 24, 16, 8, 4, 2, 1),
) -> int:
    """
    Dò batch size lớn nhất mà VRAM hiện có (tại THỜI ĐIỂM chạy, đã trừ đi
    phần bị các ứng dụng/tab trình duyệt khác chiếm dụng) có thể xử lý được,
    bằng cách thử forward pass với tensor giả (dummy) kích thước
    (batch, 3, 224, 224) - đúng kích thước input thật của CLIP ViT-B/32.

    Vì sao dò từ lớn xuống nhỏ thay vì cố định ở 16 như đề xuất ban đầu:
    ViT-B/32 là model rất nhẹ (151M tham số, ~0.3GB ở FP16), với batch 64
    ước tính hoạt động (activation) chỉ chiếm vài trăm MB, vẫn nằm sâu
    trong ngưỡng an toàn của 4GB VRAM khi không có ứng dụng khác cạnh
    tranh bộ nhớ. Dò từ lớn xuống nhỏ tận dụng tối đa phần VRAM đang rảnh,
    thay vì luôn cố định ở mức bảo thủ dù máy có nhiều bộ nhớ trống hơn.
    Nếu VRAM đang bị chiếm dụng (nhiều tab Chrome, app khác...), vòng lặp
    sẽ tự lùi dần xuống 16, 8... và thấp nhất là 1 - đúng theo yêu cầu
    "bắt đầu từ 16, giảm dần nếu OOM", chỉ khác là có thử các mức cao hơn
    16 trước khi lùi, để không lãng phí tài nguyên khi VRAM còn dư.
    """
    if device.type != "cuda":
        # Trên CPU không bị giới hạn cứng bởi VRAM, nhưng batch quá lớn sẽ
        # làm tăng độ trễ trước khi thấy kết quả đầu tiên và tăng áp lực RAM
        # hệ thống (vốn cũng bị chia sẻ với trình duyệt). 32 là điểm cân bằng.
        logger.info("Chạy trên CPU: dùng batch size cố định = 32 (không cần dò VRAM).")
        return 32

    for bs in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            dummy = torch.zeros(bs, 3, 224, 224, device=device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model.encode_image(dummy)
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.empty_cache()
            logger.info(f"Batch size an toàn được chọn: {bs} (đỉnh VRAM đo được: {peak_gb:.2f} GB).")
            return bs
        except RuntimeError as err:
            if "out of memory" in str(err).lower():
                logger.warning(f"Batch size {bs} bị tràn VRAM (OOM), thử mức nhỏ hơn...")
                torch.cuda.empty_cache()
                continue
            raise

    logger.warning("Không tìm được batch size an toàn nào >1, dùng batch size = 1.")
    return 1


def encode_images_safe(model, batch: torch.Tensor, device: torch.device, logger: logging.Logger) -> torch.Tensor:
    """
    Bọc quanh model.encode_image() với cơ chế tự phục hồi khi OOM giữa
    chừng (ví dụ người dùng mở thêm tab Chrome/app khác trong lúc script
    đang chạy, làm VRAM đột ngột bị chiếm dụng thêm). Khi gặp OOM, tự động
    chia đôi batch và thử lại đệ quy - vẫn dùng autocast FP16 để giữ hiệu
    năng, cho tới khi batch chỉ còn 1 ảnh. Việc chia đôi (thay vì bỏ ảnh)
    đảm bảo không mất dữ liệu.
    """
    try:
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = model.encode_image(batch)
            else:
                # PyTorch không hỗ trợ tốt FP16 thuần trên CPU (nhiều kernel
                # chưa được tối ưu / không tồn tại), nên trên CPU luôn chạy
                # FP32 - đúng theo hành vi mặc định của repo CLIP gốc của OpenAI.
                feats = model.encode_image(batch)
        return feats.float()
    except RuntimeError as err:
        msg = str(err).lower()
        if device.type == "cuda" and "out of memory" in msg and batch.shape[0] > 1:
            torch.cuda.empty_cache()
            mid = batch.shape[0] // 2
            logger.warning(
                f"[VRAM] Hết bộ nhớ GPU giữa chừng ở batch size {batch.shape[0]}. "
                f"Tự động chia thành 2 lô ({mid} + {batch.shape[0]-mid}) và thử lại với batch nhỏ hơn..."
            )
            first = encode_images_safe(model, batch[:mid], device, logger)
            second = encode_images_safe(model, batch[mid:], device, logger)
            return torch.cat([first, second], dim=0)
        raise RuntimeError(
            "Hết bộ nhớ GPU ngay cả với batch size = 1. Hãy chạy lại với --device cpu, "
            "hoặc đóng bớt ứng dụng/tab trình duyệt đang chiếm VRAM."
        ) from err


# ---------------------------------------------------------------------------
# 4. XÂY DỰNG EMBEDDING VĂN BẢN (PROMPT ENSEMBLING)
# ---------------------------------------------------------------------------
def build_class_centroid(model, tokenizer, prompts: List[str], device: torch.device) -> torch.Tensor:
    """
    "Prompt ensembling" theo Radford et al. (2021): mã hóa từng prompt riêng
    lẻ, chuẩn hóa L2 từng vector, lấy TRUNG BÌNH các vector đã chuẩn hóa rồi
    chuẩn hóa lại lần nữa thành 1 vector centroid duy nhất đại diện cho cả
    lớp. Vì sao lấy trung bình: không gian embedding của CLIP được huấn
    luyện để các cặp ảnh-văn bản khớp nhau nằm gần nhau trên một siêu cầu
    (theo cosine similarity); trung bình của nhiều cách diễn đạt khác nhau
    cho cùng một khái niệm sẽ tiến gần "tâm ngữ nghĩa" của khái niệm đó,
    triệt tiêu bớt nhiễu đặc thù của từng cách dùng từ riêng lẻ - về bản
    chất tương tự kỹ thuật ensembling/bagging trong ML cổ điển giúp giảm
    phương sai. Bài báo gốc báo cáo mức tăng ~3.5 điểm phần trăm độ chính
    xác zero-shot trên ImageNet khi ensembling 80 prompt so với dùng 1
    prompt đơn "a photo of a {label}." (vốn đã tăng 1.3 điểm so với dùng
    tên nhãn trần trụi không có ngữ cảnh câu).
    """
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_feats = model.encode_text(tokens).float()
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        centroid = text_feats.mean(dim=0)
        centroid = centroid / centroid.norm()
    return centroid


# ---------------------------------------------------------------------------
# 5. THU THẬP DANH SÁCH ẢNH (chỉ lưu đường dẫn, không load ảnh)
# ---------------------------------------------------------------------------
def collect_samples(src_root: Path, logger: logging.Logger) -> List[Tuple[str, str, str]]:
    """
    Duyệt Field_Raw theo đúng cấu trúc Loài/Bệnh/ảnh.jpg. Chỉ lưu 3 chuỗi
    (đường dẫn, loài, bệnh) cho mỗi ảnh - với 93,574 ảnh, tổng bộ nhớ cho
    danh sách này chỉ khoảng 15-20MB, không đáng kể so với RAM 16GB.
    """
    samples: List[Tuple[str, str, str]] = []
    if not src_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {src_root}")

    species_dirs = sorted(p for p in src_root.iterdir() if p.is_dir())
    for species_dir in species_dirs:
        disease_dirs = sorted(p for p in species_dir.iterdir() if p.is_dir())
        for disease_dir in disease_dirs:
            for f in disease_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((str(f), species_dir.name, disease_dir.name))

    logger.info(f"Tìm thấy {len(samples)} ảnh trong {len(species_dirs)} thư mục loài.")
    return samples


# ---------------------------------------------------------------------------
# 6. COPY FILE (idempotent - có thể resume an toàn)
# ---------------------------------------------------------------------------
def copy_prediction(src_path: str, species: str, disease: str, dst_root: Path, dry_run: bool) -> str:
    filename = Path(src_path).name
    dst_dir = dst_root / species / disease
    dst_path = dst_dir / filename

    if dry_run:
        return str(dst_path)

    # Idempotent: nếu file đích đã tồn tại và cùng kích thước, bỏ qua (hỗ trợ
    # chạy lại/resume sau khi bị gián đoạn mà không copy trùng lặp).
    if dst_path.exists():
        try:
            if dst_path.stat().st_size == Path(src_path).stat().st_size:
                return str(dst_path)
        except OSError:
            pass

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)  # copy2 giữ metadata; KHÔNG động vào file gốc
    return str(dst_path)


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phân loại ảnh bệnh cây lab/field bằng CLIP zero-shot.")
    p.add_argument("--src", type=str, default=r"D:\DAT301m_Project\data\Field_Raw")
    p.add_argument("--dst-field", type=str, default=r"D:\DAT301m_Project\data\Field_Clean")
    p.add_argument("--dst-lab", type=str, default=None, help="Nếu chỉ định, copy ảnh lab vào thư mục này (tùy chọn).")
    p.add_argument("--model", type=str, default="ViT-B-32", help="Tên kiến trúc CLIP (open_clip naming).")
    p.add_argument("--pretrained", type=str, default="openai", help="Nguồn trọng số pretrained.")
    p.add_argument("--batch-size", type=int, default=None, help="Bỏ trống để tự động dò batch size an toàn.")
    p.add_argument("--num-workers", type=int, default=None, help="Số worker DataLoader. Mặc định: min(8, số nhân-2).")
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng prob_field để gán nhãn 'field'.")
    p.add_argument("--review-margin", type=float, default=0.10,
                    help="Nếu |prob_field-0.5| < margin, đánh dấu 'uncertain' để review thủ công.")
    p.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N ảnh đầu (để test nhanh).")
    p.add_argument("--dry-run", action="store_true", help="Chỉ phân loại + ghi log, KHÔNG copy file.")
    p.add_argument("--resume", action="store_true", help="Bỏ qua các ảnh đã có trong log CSV từ lần chạy trước.")
    p.add_argument("--log-csv", type=str, default="classification_log.csv")
    p.add_argument("--corrupt-log", type=str, default="corrupt_images.csv")
    return p


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("clip_lab_field")


def load_already_processed(log_csv: Path) -> set:
    done = set()
    if log_csv.exists():
        with open(log_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["src_path"])
    return done


def main():
    args = build_arg_parser().parse_args()
    logger = setup_logging()

    src_root = Path(args.src)
    dst_field_root = Path(args.dst_field)
    dst_lab_root = Path(args.dst_lab) if args.dst_lab else None

    # ---- Import open_clip ở đây để thông báo lỗi rõ ràng nếu chưa cài ----
    try:
        import open_clip
    except ImportError:
        logger.error("Chưa cài open_clip_torch. Chạy: pip install open_clip_torch")
        sys.exit(1)

    device = select_device(args.device, logger)
    num_workers = args.num_workers
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        # Để lại ít nhất 2 luồng cho OS/main process/trình duyệt đang mở,
        # và không vượt quá 8 vì việc decode/resize ảnh không scale tuyến
        # tính với hyperthreads ở mức cao (thường bị giới hạn bởi I/O đĩa
        # và bộ nhớ cache CPU nhiều hơn là số luồng logic).
        num_workers = max(1, min(8, cpu_count - 2))
    logger.info(f"Số DataLoader worker: {num_workers}")

    # ---- Load model ----
    logger.info(f"Đang tải model {args.model} (pretrained={args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()
    model.to(device)

    # ---- Xây dựng centroid embedding cho 2 lớp lab/field ----
    lab_centroid = build_class_centroid(model, tokenizer, LAB_PROMPTS, device)
    field_centroid = build_class_centroid(model, tokenizer, FIELD_PROMPTS, device)
    text_features = torch.stack([lab_centroid, field_centroid], dim=0)  # [2, D] thứ tự: [lab, field]

    # logit_scale là tham số ĐƯỢC HỌC trong lúc huấn luyện CLIP (không phải
    # hằng số tự chọn) - lấy exp() để ra hệ số nhân thực tế (~100 với
    # checkpoint OpenAI gốc), đúng theo công thức chính thức của CLIP:
    # logits = 100 * cosine_similarity(image_embed, text_embed).
    logit_scale = model.logit_scale.exp().item()
    logger.info(f"logit_scale (đã exp) = {logit_scale:.2f}")

    # ---- Batch size ----
    batch_size = args.batch_size or autotune_batch_size(model, device, logger)

    # ---- Thu thập ảnh ----
    samples = collect_samples(src_root, logger)
    if args.limit:
        samples = samples[: args.limit]
        logger.info(f"--limit được bật: chỉ xử lý {len(samples)} ảnh đầu tiên.")

    if args.resume:
        done = load_already_processed(Path(args.log_csv))
        before = len(samples)
        samples = [s for s in samples if s[0] not in done]
        logger.info(f"--resume: bỏ qua {before - len(samples)} ảnh đã xử lý trước đó, còn lại {len(samples)}.")

    if not samples:
        logger.info("Không có ảnh nào cần xử lý. Kết thúc.")
        return

    dataset = ImagePathDataset(samples, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_skip_corrupt,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # ---- Mở file log (append nếu resume, ghi mới nếu không) ----
    log_path = Path(args.log_csv)
    corrupt_path = Path(args.corrupt_log)
    write_header = not (args.resume and log_path.exists())
    log_file = open(log_path, "a" if args.resume else "w", encoding="utf-8", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(
            ["src_path", "species", "disease", "predicted_label", "prob_field", "prob_lab", "confidence", "dst_path"]
        )

    corrupt_file = open(corrupt_path, "a", encoding="utf-8", newline="")
    corrupt_writer = csv.writer(corrupt_file)
    if corrupt_file.tell() == 0:
        corrupt_writer.writerow(["src_path", "error"])

    # ---- Vòng lặp inference chính ----
    n_field = n_lab = n_uncertain = n_corrupt = n_total = 0
    t_start = time.time()

    pbar = tqdm(loader, total=len(loader), unit="batch", desc="Phân loại lab/field")
    for tensors, paths, species_list, disease_list, bad_records in pbar:
        for bad_path, err in bad_records:
            corrupt_writer.writerow([bad_path, err])
            n_corrupt += 1

        if tensors.shape[0] == 0:
            continue

        tensors = tensors.to(device, non_blocking=True)
        image_features = encode_images_safe(model, tensors, device, logger)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.T  # [N, 2] -> cột 0: lab, cột 1: field
        probs = logits.softmax(dim=-1)
        prob_field = probs[:, 1].detach().cpu()

        for i, path in enumerate(paths):
            p_field = prob_field[i].item()
            p_lab = 1.0 - p_field
            label = "field" if p_field >= args.threshold else "lab"
            confidence = "uncertain" if abs(p_field - 0.5) < args.review_margin else "certain"

            dst_path = ""
            if label == "field":
                n_field += 1
                dst_path = copy_prediction(path, species_list[i], disease_list[i], dst_field_root, args.dry_run)
            else:
                n_lab += 1
                if dst_lab_root is not None:
                    dst_path = copy_prediction(path, species_list[i], disease_list[i], dst_lab_root, args.dry_run)

            if confidence == "uncertain":
                n_uncertain += 1

            log_writer.writerow([path, species_list[i], disease_list[i], label, f"{p_field:.4f}", f"{p_lab:.4f}", confidence, dst_path])
            n_total += 1

        log_file.flush()  # flush thường xuyên để không mất tiến trình nếu bị gián đoạn
        elapsed = time.time() - t_start
        img_per_sec = n_total / elapsed if elapsed > 0 else 0
        pbar.set_postfix(field=n_field, lab=n_lab, unsure=n_uncertain, corrupt=n_corrupt, img_s=f"{img_per_sec:.1f}")

    log_file.close()
    corrupt_file.close()

    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info("HOÀN TẤT")
    logger.info(f"Tổng ảnh xử lý thành công : {n_total}")
    logger.info(f"  -> field (đã copy)      : {n_field}")
    logger.info(f"  -> lab                  : {n_lab}" + (" (đã copy)" if dst_lab_root else " (không copy, chỉ log)"))
    logger.info(f"  -> uncertain (nên review): {n_uncertain}")
    logger.info(f"Ảnh lỗi/không đọc được     : {n_corrupt}")
    logger.info(f"Thời gian chạy             : {elapsed/60:.1f} phút ({n_total/max(elapsed,1e-9):.1f} ảnh/giây)")
    logger.info(f"Log chi tiết               : {log_path.resolve()}")
    if n_corrupt:
        logger.info(f"Danh sách ảnh lỗi          : {corrupt_path.resolve()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
