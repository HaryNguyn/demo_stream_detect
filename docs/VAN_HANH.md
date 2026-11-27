# Cài đặt và vận hành

Chạy lệnh từ thư mục chứa `app.py`. Cần Python 3.12, trọng số trong `models/` và MariaDB cho web. Với nguồn stream, cài FFmpeg vào `PATH`; nguồn trang web được hỗ trợ dùng yt-dlp và có thể cần Node.js. CLI chạy file video không cần MariaDB.

## 1. Cài đặt và chạy web

PowerShell:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env  # Chỉ chạy nếu chưa có .env
```

Khởi động MariaDB. Dùng tài khoản quản trị để tạo database và tài khoản ứng dụng (thay mật khẩu mẫu):

```sql
CREATE DATABASE IF NOT EXISTS lane_demo CHARACTER SET utf8mb4;
CREATE USER IF NOT EXISTS 'vehicle_ai'@'localhost' IDENTIFIED BY 'change-me';
GRANT SELECT, INSERT, UPDATE, CREATE, ALTER ON lane_demo.* TO 'vehicle_ai'@'localhost';
```

Điền `MARIADB_HOST`, `MARIADB_PORT`, `MARIADB_DATABASE`, `MARIADB_USER`, `MARIADB_PASSWORD` trong `.env`. Web tự tạo bảng `events`, không tự tạo database. `VDTC_IMAGE_ROOT` đổi nơi lưu ảnh bằng chứng (mặc định `runtime/images/`). Không đưa `.env`, ảnh hoặc video thật lên kho công khai.

```powershell
.\.venv312\Scripts\python.exe app.py
```

Mở <http://127.0.0.1:8000>. Thêm `--port 8001` nếu cổng bị chiếm; tắt máy chủ bằng `Ctrl+C`. Trên Linux/macOS, dùng `python3.12` để tạo venv và `.venv312/bin/python` để chạy.

## 2. Nhận diện trên web

1. Chọn video MP4/AVI/MOV/MKV hoặc nhập URL luồng RTSP/HTTP. Có thể chọn nhiều file; URL chạy sau cùng. Với YouTube hoặc trang được yt-dlp hỗ trợ, chọn chế độ nguồn phù hợp. Camera IP phải dùng URL luồng video, không dùng trang quản trị HTML.
2. Nếu dùng camera IP, mở cấu hình camera để tạo URL Hikvision/Dahua. Có thể điền sẵn `IP_CAM_URL` hoặc các trường `IP_CAM_*` trong `.env` theo [mẫu](../.env.example). Ưu tiên dùng các trường rời và nhập mật khẩu trong UI; không chia sẻ URL có chứa mật khẩu.
3. Giữ model mặc định hoặc chọn **Pipeline YAML** (ví dụ `configs/vietnam-ocr.yaml`), lúc đó các ô model thủ công không được dùng. Đường dẫn trọng số trong YAML tương đối với file YAML; đường dẫn nhập trên UI là đường dẫn của máy chủ, tương đối với thư mục chạy lệnh.
4. Nếu cần ROI, chọn trên giao diện hoặc cung cấp JSON như `{"points":[[0.1,0.2],[0.9,0.2],[0.9,0.8],[0.1,0.8]]}`. Tọa độ chuẩn hóa từ 0 đến 1; ROI chỉ định trên web ghi đè ROI trong YAML.
5. Bấm **Phân tích video**. Giới hạn mặc định là 60 giây mỗi nguồn; `0` xử lý tới hết hoặc tới khi bấm **Stop**. Đóng tab không dừng phiên. **Stop** chờ bước hiện tại kết thúc, chốt track đang có và bỏ qua các nguồn còn lại.

Trang chính hiển thị khung hình đã chú thích, tiến độ, thông tin OCR và lịch sử có ảnh bằng chứng. Có thể tải tối đa 200 sự kiện gần nhất qua CSV. Các API chính: `/api/status`, `/api/frame`, `/api/stream`, `/api/events`, `/api/events.csv`; `/api/analyze` và `/api/stop` nhận `POST`.

Để thử ảnh, mở <http://127.0.0.1:8000/plate> và chọn ảnh hoặc thư mục (JPG/JPEG/PNG/BMP/WebP). Ảnh chạy lần lượt; dừng hàng đợi ảnh không ảnh hưởng phiên video.

## 3. CLI và Docker

Ví dụ chạy với video có sẵn; thay `--source` bằng đường dẫn thật. Thư mục `--output` phải chưa tồn tại. Thêm `--roi-file` nếu cần ghi đè ROI trong YAML.

```powershell
.\.venv312\Scripts\python.exe -m vehicle_ai run `
  --config configs/vietnam-ocr.yaml --source data/raw/cam01.mp4 `
  --source-id video-001 --camera CAM-01 --output runtime/runs/demo-001
```

Docker CPU: chuẩn bị `.env`, rồi chạy `docker compose --profile ui up -d --build ui`. Nếu MariaDB ở Windows host, dùng `host.docker.internal` và cấp quyền cho tài khoản kết nối từ container. Các đường dẫn model/ROI phải thấy được từ container; service `ui` mặc định mount `models/`, `configs/` và `runtime/`, không mount `data/`. Compose công bố cổng 8000; chỉ dùng trong môi trường tin cậy.

## 4. Xử lý lỗi nhanh

- **Web không khởi động:** kiểm tra MariaDB, database, mật khẩu và quyền theo host kết nối.
- **Không mở được stream:** kiểm tra FFmpeg và thử URL luồng bằng trình phát; tránh URL trang quản trị.
- **Không thấy biển hoặc hãng:** kiểm tra trọng số, ROI, ngưỡng kích thước và phần chẩn đoán. Xe không đọc được biển không hiện trong danh sách mặc định.
- **Có sự kiện nhưng ảnh trả 404:** kiểm tra `VDTC_IMAGE_ROOT` và file trên đĩa. Xóa ảnh không xóa sự kiện trong database.
