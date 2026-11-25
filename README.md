# Vehicle AI

Nhận diện xe, loại xe, biển số và logo từ video/camera thử nghiệm.
Gồm pipeline theo dõi xe, UI nhận diện web và CLI train, đánh giá.

## Chạy

Python 3.12, PowerShell tại thư mục dự án:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -e .
.\.venv312\Scripts\python.exe app.py
```

- Nhận diện: **http://127.0.0.1:8000**, giao diện HTML/CSS/JavaScript và API Flask. Xem [hướng dẫn web local](docs/LOCAL_WEB.md).
- CLI train và đánh giá: `.\.venv312\Scripts\python.exe -m vehicle_ai --help`.

Máy đã có môi trường thì bỏ bước tạo venv. Linux/macOS dùng `python3.12` và
`.venv312/bin/python`. FFmpeg cần nằm trong PATH cho nguồn stream; yt-dlp/Node.js
chỉ phục vụ nguồn trang web được hỗ trợ.

## Model và dữ liệu

Tất cả trọng số nằm trong `models/`:

| File | Nhiệm vụ |
|---|---|
| `yolo11n.pt` | Phát hiện xe/loại xe |
| `best_stripped.onnx` | Phát hiện biển số (mặc định, chạy ở 640px) |
| `ocr_model.onnx` + YAML | OCR biển Việt Nam (mặc định) |
| `logo.pt` | Nhận diện 26 hãng xe |

Web mặc định dùng `ocr_model.onnx` (OCR Việt Nam fine-tuned), `best_stripped.onnx` và
`logo.pt` 26 hãng; UI không còn lựa chọn OCR CCT cũ. Chế độ YAML dùng `configs/vietnam-ocr.yaml`.
`configs/baseline.yaml` vẫn dùng được cho pipeline baseline. Giữ bộ nhãn đã duyệt và báo cáo
trong `data/`, `runtime/evaluation/`, `docs/`.
CRNN riêng có công cụ train/export; chưa được chọn làm OCR mặc định.

## Cấu hình

- Chép `.env.example` thành `.env` nếu chưa có; giữ `.env` riêng của máy.
- MariaDB là database duy nhất. Tạo database trước rồi điền các biến `MARIADB_*`
  trong `.env`; ứng dụng tự tạo bảng `events`. Xem [cấu hình database](docs/LOCAL_WEB.md#7-database-và-ảnh).
- Ảnh bằng chứng: `runtime/images/`, đổi bằng `VDTC_IMAGE_ROOT`.
- ROI là file JSON `{"points": [[x,y], ...]}` với tọa độ chuẩn hóa 0–1; chọn ở UI
  nhận diện hoặc truyền `--roi-file` cho CLI. Danh sách kết quả và CSV chỉ trả xe
  có chuỗi biển số; xe chưa đọc được vẫn có trong nhật ký chẩn đoán.

Ví dụ baseline:

```powershell
.\.venv312\Scripts\python.exe -m vehicle_ai run `
  --config configs/baseline.yaml --source data/raw/dataset1.mp4 `
  --source-id dataset1 --camera CAM-01 --output runtime/runs/new-baseline
```

Output phải là thư mục mới. CLI lưu JSONL/ảnh độc lập với database của UI.

## Cấu trúc

```text
app.py                 Web nhận diện Flask, cổng 8000
src/vehicle_ai/web.py   API, tác vụ nền, Stop, xuất CSV
src/vehicle_ai/web_static/ HTML/CSS/JavaScript
src/vehicle_ai/        Core, web, database và công cụ dữ liệu/train/evaluation
configs/              Cấu hình pipeline và lớp nhãn
models/               Trọng số và thông tin nguồn
data/                 Video, ảnh, nhãn đã duyệt, ROI
runtime/              Ảnh bằng chứng, báo cáo đánh giá và kết quả chạy (không nằm trong git)
scripts/, tests/      Công cụ kiểm tra và kiểm thử
docs/                 Hướng dẫn và báo cáo
```

Các checkout `third_party` và archive bên ngoài đã được xóa; thông tin nguồn
vẫn được lưu trong [THIRD_PARTY.md](docs/THIRD_PARTY.md).

## Docker CPU

```powershell
docker compose build
docker compose --profile ui up -d ui
```

Model/config/data mount riêng; `.env` không đóng vào image.
Mở http://127.0.0.1:8000 sau khi container khởi động.
MariaDB trên Windows host dùng `host.docker.internal` khi truy cập từ Docker.

## Kiểm tra

```powershell
.\.venv312\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv312\Scripts\python.exe -m ruff check .
.\.venv312\Scripts\python.exe -m unittest discover -s tests -v
```

## Phạm vi hiện tại

Mỗi nguồn có tracker riêng; mặc định chốt event sau 2 giây mất dấu hoặc cuối
nguồn. Một xe có thể tạo nhiều track. ROI lọc theo tâm bbox, detector vẫn chạy
toàn ảnh. Chưa triển khai trigger, webhook/outbox và đồng bộ thời gian camera.

- Hướng dẫn dữ liệu/train: [PHAT_TRIEN_AI.md](docs/PHAT_TRIEN_AI.md).
- Chạy và sử dụng web: [LOCAL_WEB.md](docs/LOCAL_WEB.md).
