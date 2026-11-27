# Đo hàng loạt: ảnh và ma trận độ phân giải video

Bộ công cụ này phục vụ hai tập kiểm tra: ảnh theo số lượng (ví dụ 50/500/5000/10000)
và video 10/30 phút ở 144p → 1080p. Mọi kết quả ghi vào thư mục **mới**, không ghi đè.

## 1. Cấu hình riêng cho từng độ phân giải

Các ngưỡng pixel mặc định (`min_vehicle_width: 120`, `min_plate_width: 32`,
`quality_target_width/height: 100/32`) được hiệu chỉnh cho ~720p. Ở 144p chúng sẽ
loại hầu hết xe/biển trước khi OCR. Lệnh sau scale các ngưỡng đó theo chiều cao ảnh,
lấy 720p làm gốc, và ghi rõ từng giá trị đã đổi vào `configs.json`:

```powershell
.\.venv312\Scripts\python.exe scripts/measure_video_matrix.py configs `
  --base configs/vietnam-ocr.yaml --output-dir configs/resolutions
```

- `detector_imgsz` / `plate_imgsz` **không đổi** vì là kích thước input model, không
  phải ngưỡng pixel của ảnh nguồn.
- Giá trị có sàn (8 hoặc 20) để tránh ngưỡng quá nhỏ gây nhiễu.
- Mỗi YAML được nạp thử bằng `load_config` trước khi dùng.
- Nhớ tạo config mới khi đổi config gốc; dùng `--force` nếu muốn ghi đè.

## 2. Ảnh 50 / 500 / 5000 / 10000

Ground truth ảnh là JSONL, mỗi dòng một ảnh:

```jsonl
{"filename":"img001.jpg","plate":"30A12345","vehicle_type":"Ô tô con","brand":"toyota","tags":["144p","day"]}
{"filename":"img002.jpg","plate":null,"vehicle_type":"Xe máy","tags":["144p","hard"]}
```

`filename` khớp theo đường dẫn tương đối (ví dụ `144p/img001.jpg`) hoặc tên file nếu ảnh
nằm trực tiếp trong thư mục gốc. `plate: null` vẫn tính vào recall
nhưng không tính vào CER/độ đúng biển, giống `evaluate-events`.

```powershell
.\.venv312\Scripts\python.exe scripts/measure_images.py `
  --images data/images `
  --truth data/images.truth.jsonl `
  --config configs/resolutions/res144p.yaml `
  --option min_vehicle_width=24 `
  --counts 50 500 5000 10000 `
  --output runtime/measure-images-144p
```

- Script chạy **một lượt duy nhất** tới số ảnh lớn nhất rồi tính các prefix 50/500/...,
  nên không OCR lại từng ngưỡng. Dùng `--shuffle --seed 42` để lấy mẫu ngẫu nhiên cố định.
- `--warmup 2` (mặc định) chỉ loại ảnh đầu khỏi thống kê độ trễ; độ chính xác vẫn dùng
  tất cả ảnh.
- `--option KEY=VALUE` nhận các khóa trong `plate_image.DEFAULT_OPTIONS`
  (`min_vehicle_width`, `vehicle_conf`, `plate_conf`, `detector_imgsz`, ...).
- `results.jsonl` đã **bỏ base64** (`annotated`, `plate_image`), chỉ giữ nhãn/điểm.
- `summary.json` có `images_per_second`, `median_ms`, `p95_ms`, `max_ms`, số lỗi,
  `groups` và `tag_groups` ( recall biển/CER/độ đúng loại xe-hãng theo tag).

## 3. Video 10 phút và 30 phút × 6 độ phân giải

Đặt tên file theo mẫu `<số>min_<độ phân giải>.mp4`, ví dụ
`data/videos/10min_144p.mp4`, `data/videos/30min_1080p.mp4`. Mỗi video có một file
ground truth tại `data/truth/<tên video>.truth.jsonl` (hoặc `<tên>.jsonl`) theo
[GROUND_TRUTH.md](GROUND_TRUTH.md). Gắn tag `144p` và `10min` vào từng dòng để báo cáo
tách được theo độ phân giải/thời lượng.

```powershell
# Tạo manifest + commands.ps1 + commands.sh
.\.venv312\Scripts\python.exe scripts/measure_video_matrix.py plan `
  --videos data/videos --truth-dir data/truth `
  --config-dir configs/resolutions --output runtime/matrix

# Chạy 12 lệnh run + evaluate-events (merged và raw)
powershell -File runtime/matrix/commands.ps1

# Gộp kết quả
.\.venv312\Scripts\python.exe scripts/measure_video_matrix.py report `
  --manifest runtime/matrix/manifest.csv --output runtime/matrix/report
```

`matrix.csv` gồm `frame_size`, `source_fps`, `content_seconds`, thông lượng, recall,
CER và các cột `tag_144p_*`, `tag_10min_*`. Kiểm tra `frame_size` phải đúng độ phân giải
đã đặt: nếu luôn là `1280x720` thì nguồn đã bị fallback FFmpeg ép khung hình.

Không dùng `--realtime` khi so sánh ngưỡng: chế độ này bỏ frame nên kết quả phụ thuộc
tải CPU. Script `plan` đã bỏ cờ này. 30 phút ở 30 FPS là ~54.000 frame; nên chạy thử
với `--max-seconds` ngắn (sửa trong `commands.ps1`) trước khi chạy full.

## 4. Đọc kết quả

- Giảm dần độ phân giải mà recall/CER tệ dần là kỳ vọng; nếu tụt đột ngột ở 144p,
  xem `rejection_reasons` trong `events.merged.jsonl` (`vehicle_too_small`,
  `plate_too_small`, `plate_too_blurry`) và đối chiếu ngưỡng trong `configs.json`.
- So hai cấu hình cùng một video bằng `--baseline` của `evaluate-events`, như
  [EVALUATE_CONFIDENCE.md](EVALUATE_CONFIDENCE.md).
- Vài chục lượt xe chưa đủ kết luận; ghi số tuyệt đối (`"41/52"`) thay vì chỉ tỷ lệ.
