# Vehicle AI

Nhận diện xe, loại xe, biển số và logo từ video/camera thử nghiệm.
Gồm pipeline theo dõi xe, UI nhận diện, duyệt nhãn/ROI, CLI train và đánh giá.

## Chạy

Python 3.12, PowerShell tại thư mục dự án:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -e .
.\.venv312\Scripts\python.exe app.py
```

- Nhận diện: **http://127.0.0.1:8000**, giao diện HTML/CSS/JavaScript và API Flask.
- Không cần Streamlit để chạy nhận diện. Xem [hướng dẫn web local](docs/LOCAL_WEB.md).
- Công cụ duyệt nhãn/ROI riêng vẫn dùng Streamlit; chỉ cài thêm khi cần:

```powershell
.\.venv312\Scripts\python.exe -m pip install -e ".[review]"
.\.venv312\Scripts\python.exe -m streamlit run review_app.py --server.port 8502
.\.venv312\Scripts\python.exe -m vehicle_ai --help
```

Máy đã có môi trường thì bỏ bước tạo venv. Linux/macOS dùng `python3.12` và
`.venv312/bin/python`. FFmpeg cần nằm trong PATH cho nguồn stream; yt-dlp/Node.js
chỉ phục vụ nguồn trang web được hỗ trợ.

## Model và dữ liệu

Tất cả trọng số nằm trong `models/`:

| File | Nhiệm vụ |
|---|---|
| `yolo11n.pt` | Phát hiện xe/loại xe — model gốc đang dùng |
| `DetectLisence_YOLO11.pt` | Phát hiện biển số |
| `cct_s_v1_vn.onnx` + YAML | OCR biển Việt Nam |
| `logo.pt` | Nhận diện logo/hãng |

Web mặc định chọn model thủ công với OCR CCT-VN, giống giao diện nhận diện cũ.
Chế độ YAML gợi ý `configs/vietnam-ocr.yaml` dùng `ocr_model.onnx`.
`configs/baseline.yaml` vẫn dùng được cho pipeline baseline. Bản fine-tune v001 đã được loại
bỏ sau so sánh; giữ bộ nhãn đã duyệt và báo cáo trong `data/`, `runtime/`, `docs/`.
CRNN riêng có công cụ train/export; chưa được chọn làm OCR mặc định.

## Cấu hình

- Chép `.env.example` thành `.env` nếu chưa có; giữ `.env` riêng của máy.
- MariaDB là database duy nhất. Tạo database trước rồi điền các biến `MARIADB_*`
  trong `.env`; ứng dụng tự tạo bảng `events`. Xem [cấu hình database](docs/LOCAL_WEB.md#7-database-và-ảnh).
- Ảnh bằng chứng: `runtime/images/`, đổi bằng `VDTC_IMAGE_ROOT`.
- ROI vẽ tại UI duyệt nhãn, sau đó chọn file JSON ở UI nhận diện hoặc
  truyền `--roi-file` cho CLI.

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
review_app.py          Công cụ duyệt nhãn tùy chọn, cổng 8502
src/vehicle_ai/web.py   API, tác vụ nền, Stop, xuất CSV
src/vehicle_ai/web_static/ HTML/CSS/JavaScript
src/vehicle_ai/        Core, UI, database và công cụ dữ liệu/train/evaluation
configs/              Cấu hình pipeline và lớp nhãn
models/               Trọng số và thông tin nguồn
data/                 Video, ảnh, nhãn đã duyệt, ROI
runtime/              Database, ảnh và kết quả chạy
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
node tests/roi_canvas.test.cjs
```

## Phạm vi hiện tại

Mỗi nguồn có tracker riêng; mặc định chốt event sau 2 giây mất dấu hoặc cuối
nguồn. Một xe có thể tạo nhiều track. ROI lọc theo tâm bbox, detector vẫn chạy
toàn ảnh. Chưa triển khai trigger, webhook/outbox và đồng bộ thời gian camera.

- Hướng dẫn dữ liệu/train: [PHAT_TRIEN_AI.md](docs/PHAT_TRIEN_AI.md).
- Chạy và sử dụng web: [LOCAL_WEB.md](docs/LOCAL_WEB.md).
