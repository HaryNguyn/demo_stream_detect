# Kết quả triển khai 19/11/2026

## Đã có

1. Manifest detector/OCR, trích frame, split theo group, kiểm tra rò rỉ và nhãn.
2. Chạy baseline bằng CLI, xuất JSONL + JPEG + cấu hình/hash model.
3. Package nhận diện tách khỏi Streamlit; session feed/finalize.
4. Train detector YOLO và OCR CRNN/CTC riêng; chọn checkpoint bằng validation.
5. Export ONNX; đánh giá detector, OCR và sự kiện.
6. Dockerfile CPU và Compose cho CLI inference/UI.

Hướng dẫn vận hành: [PHAT_TRIEN_AI.md](PHAT_TRIEN_AI.md).

## Kiểm tra đã thực hiện

- 15 unit/integration tests đạt.
- Streamlit khởi động thành công với SQLite tạm.
- Nạp model xe/logo/biển/CCT hiện có và chạy clip kiểm thử 5 frame:
  tạo được sự kiện và ảnh.
- Trích frame qua CLI thành công.
- CRNN train 1 epoch trên dữ liệu tổng hợp, lưu/nạp checkpoint thành công.
- CRNN export ONNX và đối chiếu logits với PyTorch đạt tolerance kiểm thử.
- Detector train 1 epoch trên dữ liệu tổng hợp và export ONNX thành công.
- `docker compose config --quiet` đạt.
- Docker Engine đang tắt: chưa build/chạy container thực tế.

Báo cáo smoke test:
`runtime/verification/modular-001/verification.json`.
Các checkpoint trong thư mục verification chỉ phục vụ kiểm thử, không dùng
thay model chính và không phải bằng chứng độ chính xác trên camera thực tế.

## Chưa thể thực hiện khi thiếu dữ liệu

- Chưa có bộ train/val/test gán nhãn của dự án.
- Chưa đo baseline trên video camera mục tiêu có ground truth.
- Chưa train/đánh giá model riêng đủ điều kiện triển khai.

Backup trước khi tách module:
`../../backup-before-modular-20261119-205134`.
