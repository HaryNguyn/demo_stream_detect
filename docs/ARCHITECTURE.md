# Kiến trúc mã nguồn hiện tại

## Cấu trúc

```text
app.py                      # điểm vào UI nhận diện, gọi ui.detection.main()
review_app.py               # điểm vào UI duyệt nhãn/ROI, gọi ui.review.main()
src/vehicle_ai/
├── ui/                     # Streamlit: hiển thị, input và callback
├── persistence/            # EventRepository, SQLite và MariaDB
├── settings.py             # .env, biến môi trường, đường dẫn ứng dụng
├── config.py               # PipelineConfig + kiểm tra YAML
├── model_loader.py         # dựng OCR adapter, dùng chung CLI/UI
├── capture.py              # nguồn video/FFmpeg
├── inference.py            # nhận diện trên crop xe
├── ocr.py, crnn.py          # CCT và CRNN
├── aggregation.py          # tổng hợp kết quả nhiều frame, ảnh tốt
├── session.py              # một phiên tracking riêng theo nguồn
├── pipeline.py             # vòng đọc frame và callback
├── domain.py               # schema event JSON độc lập UI
├── evidence.py             # JPEG + metadata cho event lưu DB
├── storage.py              # JSONL + ảnh cho lần chạy CLI
├── annotation.py           # gợi ý nhãn, duyệt và xuất manifest
├── roi.py                  # hình học ROI, dùng cho inference
├── roi_editor.py           # adapter UI canvas
├── roi_component/          # HTML/JS canvas được đóng gói cùng package
├── datasets.py             # kiểm tra, split và chuẩn bị dataset
├── training.py             # train/export/evaluate OCR
├── evaluation.py           # đánh giá sự kiện
└── cli.py                  # phân tích đối số và điều phối lệnh
configs/                    # cấu hình và taxonomy
models/                     # trọng số baseline, nguồn model
data/                       # video, ảnh, nhãn duyệt, ROI
runtime/                    # database, ảnh, kết quả thực nghiệm
scripts/                    # kiểm thử tích hợp/so sánh model
tests/                      # kiểm thử Python và canvas JS
docs/                       # hướng dẫn và báo cáo lịch sử
```

## Ranh giới trách nhiệm

- UI và CLI gọi core; core không import Streamlit hoặc CLI.
- `RecognitionSession` quản lý track, nhận `feed(frame, timestamp)` và `finalize()`.
- `model_loader` dựng model, không phụ thuộc cách hiển thị kết quả.
- `EventRepository` có `append`, `update_images`, `recent`; backend không biết UI.
- `evidence` lưu ảnh, không ghi DB; UI điều phối ghi event rồi cập nhật đường dẫn ảnh.
- `storage` chỉ lưu thí nghiệm JSONL; không ghi vào DB vận hành.
- Nhãn đã duyệt, schema SQLite/MariaDB và cấu trúc JSONL được giữ tương thích.

Các file cầu nối `events.py`, `sqlite_events.py`, `pipeline.py`, `recognition.py`,
`sources.py` ở root đã được loại bỏ; code mới import trực tiếp từ `vehicle_ai`.

## Cấu hình đường dẫn

- Checkout source tự nhận thư mục chứa `pyproject.toml` và `configs/`.
- Package cài dạng wheel dùng thư mục làm việc làm mặc định.
- `VEHICLE_AI_HOME` có thể chỉ định root rõ ràng; Docker đặt `/app`.
- Thứ tự thiết lập: môi trường tiến trình → `.env` của project → môi trường
  người dùng Windows → mặc định.
- Đọc `.env` không tự ghi các biến vào `os.environ`.

## Các thư mục đã dọn

**Cập nhật:** ba thư mục backup và `reference-archive-20261119` bên ngoài
dự án đã được xóa theo yêu cầu người dùng. Các đường dẫn backup/archive
bên dưới chỉ ghi lại lịch sử thao tác.

- `third_party/` chuyển nguyên vẹn ra
  `../reference-archive-20261119/third_party/` (tính từ root dự án).
- `runs/` cũ là đầu ra smoke test lạc vị trí, chuyển vào `runtime/legacy_runs/`.
- Mã trước khi dọn được sao lưu ở `../backup-before-cleanup-20261119-225447/`.
- Thông tin nguồn: [THIRD_PARTY.md](../THIRD_PARTY.md).

## Kiểm tra khi phát triển

```powershell
.\.venv312\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv312\Scripts\python.exe -m ruff check .
.\.venv312\Scripts\python.exe -m unittest discover -s tests -v
node tests/roi_canvas.test.cjs
docker compose config --quiet
```

Test UI/database dùng thư mục tạm. Không cần kết nối MariaDB thật để chạy suite.
