# Hướng dẫn web nhận diện xe local

## 1. Kiến trúc

`app.py` chạy máy chủ Flask; trình duyệt hiển thị HTML/CSS/JavaScript.
Pipeline AI chạy trong luồng nền, giao diện cập nhật trạng thái mỗi giây.
Không cần cài hoặc chạy Streamlit để nhận diện xe.

- `src/vehicle_ai/web.py`: upload, hàng đợi nguồn, API trạng thái/ảnh/CSV/Stop.
- `src/vehicle_ai/web_models.py`: chọn model thủ công, YAML và ROI.
- `src/vehicle_ai/web_static/`: giao diện.
- `src/vehicle_ai/pipeline.py`: đọc video và điều phối nhận diện.
- MariaDB lưu sự kiện; thư mục ảnh lưu bằng chứng.

## 2. Cài đặt và chạy trên Windows

Mở PowerShell tại thư mục chứa `app.py` và `pyproject.toml`:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -e .
.\.venv312\Scripts\python.exe app.py
```

Nếu đã có môi trường, bỏ bước tạo venv. Mở **http://127.0.0.1:8000**.
Trước lần chạy đầu tiên, cấu hình MariaDB theo mục 7.

Đổi cổng:

```powershell
.\.venv312\Scripts\python.exe app.py --port 8001
```

Linux/macOS dùng `python3.12` và `.venv312/bin/python`.
Nguồn stream cần FFmpeg trong PATH. YouTube/trang web dùng yt-dlp,
có thể cần Node.js tùy nguồn.

## 3. Chọn nguồn

1. Chọn một hoặc nhiều video MP4/AVI/MOV/MKV.
2. Có thể nhập thêm URL YouTube, RTSP hoặc HTTP.
   Nếu đặt `IP_CAM_URL` trong `.env`, URL camera IP sẽ được điền sẵn khi mở giao diện;
   vẫn có thể thay bằng URL khác trước khi chạy.
3. Với camera IP, mở **Cấu hình IP camera / RTSP**, nhập IP, cổng, tài khoản,
   mật khẩu, kênh và chọn Hikvision/Dahua rồi bấm **Tạo URL RTSP**. Có thể điền
   sẵn các trường rời bằng `IP_CAM_HOST`, `IP_CAM_PORT`, `IP_CAM_USER`,
   `IP_CAM_CHANNEL`, `IP_CAM_STREAM`, `IP_CAM_BRAND`, `IP_CAM_PATH` trong `.env`.
   Mật khẩu camera chỉ nhập trên UI, không trả về qua `/api/config`.
4. **Lưu ý:** URL như `http://10.248.166.48/doc/page/preview.asp` là trang quản
   trị HTML, không phải luồng video; ứng dụng sẽ báo lỗi. Dùng URL RTSP, ví dụ
   Hikvision `rtsp://user:pass@IP:554/Streaming/Channels/101` (main) hoặc
   `/102` (sub), Dahua `.../cam/realmonitor?channel=1&subtype=0`.
5. Loại URL **Tự động / luồng trực tiếp** tự nhận YouTube.
   Chọn **Trang web (yt-dlp)** cho các trang khác được yt-dlp hỗ trợ.
6. Giới hạn mặc định 60 giây **cho mỗi nguồn**; nhập 0 để xử lý đến hết.
7. Nhấn **Phân tích video**. Các video chạy lần lượt, URL chạy cuối.

Tổng request upload tối đa 4 GiB. Mỗi máy chủ chạy một phiên phân tích tại
một thời điểm. Đóng tab không dừng tác vụ; mở lại trang để xem trạng thái.

## 4. Model và OCR

Mở **Cấu hình model nhận diện / OCR / ROI**.

### Chọn model thủ công (mặc định)

| Trường | Giá trị mặc định |
|---|---|
| Loại xe | `models/yolo11n.pt` |
| Logo/hãng | `models/logo.pt` |
| Phát hiện biển | `models/best_stripped.onnx` |
| OCR | OCR Việt Nam (character slots) |
| Model OCR | `models/ocr_model.onnx` |
| YAML OCR | `models/ocr_model.yaml` |

Mặc định đã dùng OCR fine-tuned; chỉ cần sửa đường dẫn nếu đổi model.
Các model logo/biển là tùy chọn theo cơ chế pipeline hiện tại; kiểm tra
đường dẫn nếu không có kết quả hãng hoặc biển.

### Pipeline YAML

Chọn **Pipeline YAML** và nhập đường dẫn cấu hình, ví dụ
`configs/vietnam-ocr.yaml` hoặc `configs/baseline.yaml`.
Chế độ này lấy model và tham số OCR từ YAML, bỏ qua các ô model thủ công.
Đường dẫn model bên trong YAML được giải quyết tương đối với file YAML.

### ROI

Nhập đường dẫn JSON có trường `points` theo định dạng ROI của dự án.
ROI nhập tại đây ghi đè ROI trong cấu hình. Bỏ trống để dùng cấu hình gốc.
Các đường dẫn nhập trên web là đường dẫn trên máy chạy Python;
đường dẫn tương đối tính từ thư mục chạy lệnh.

### Nhận diện ảnh đơn hoặc cả thư mục

Mở `http://127.0.0.1:8000/plate`, rồi chọn một ảnh, nhiều ảnh hoặc chọn cả thư
mục ảnh. Các định dạng hỗ trợ: JPG, JPEG, PNG, BMP và WebP. Kết quả được hiển thị
riêng cho từng ảnh. Trang xếp ảnh vào hàng đợi và gửi từng ảnh trong một request
riêng để tránh request quá lớn; có thể theo dõi tiến độ hoặc dừng sau ảnh hiện tại.
Trang này không dừng phiên video hoặc ghi sự kiện vào database. Nếu detector xe
bỏ sót, trang quét biển trực tiếp toàn ảnh ở nhiều tỉ lệ, suy luận vùng xe rồi
thử OCR trên vài biến thể crop (đệm, phóng to, ngưỡng sáng) và chọn kết quả được
nhiều biến thể ủng hộ nhất.

## 5. Xem kết quả

- Bên trái: khung hình đã chú thích, tiến độ, nguồn đang xử lý.
- Chẩn đoán: số mẫu, số lần phát hiện biển, OCR đạt, chuỗi OCR và điểm.
- Bên phải: biển số, Track ID, loại xe, hãng, thời điểm và ảnh bằng chứng.
- **Copy** sao chép biển số; **Xem ảnh** mở ảnh lớn; **Tải ảnh** lưu ảnh.
- **CSV** xuất tối đa 200 sự kiện gần nhất, có BOM UTF-8 để mở bằng Excel.

Thẻ xuất hiện khi xe mất dấu quá ngưỡng hoặc phiên kết thúc/được dừng.
Điểm OCR hiển thị là điểm đọc tốt nhất của biển đã chọn, không phải độ tin
cậy từng ký tự. Ảnh xe là ảnh bằng chứng đã lưu, không nhất thiết là toàn frame.
Lịch sử gồm cả các phiên trước. Một xe mất dấu rồi xuất hiện lại có thể tạo track mới.

## 6. Dừng và chạy lại

**Stop** yêu cầu dừng phiên, chốt các xe đang theo dõi và bỏ qua nguồn còn lại.
Giao diện chuyển **Đang dừng…** rồi **Đã dừng**. Chờ bước đọc frame,
inference hoặc tải model hiện tại hoàn tất; nút không tắt máy chủ.
Sau đó có thể chọn nguồn và chạy phiên mới.

Để tắt máy chủ và giải phóng cổng, nhấn **Ctrl+C** trong terminal chạy Python.
Khi cập nhật code, dừng máy chủ, chạy lại `python app.py`, rồi tải lại trang
(Ctrl+F5 nếu trình duyệt còn giữ CSS/JavaScript cũ).

## 7. Database và ảnh

Giữ cấu hình riêng trong `.env` (tham khảo `.env.example`):

- MariaDB là database duy nhất; không còn lựa chọn `EVENT_STORE`.
- Điền `MARIADB_HOST`, `MARIADB_PORT`, `MARIADB_DATABASE`,
  `MARIADB_USER`, `MARIADB_PASSWORD`.
- `VDTC_IMAGE_ROOT`: thư mục bằng chứng; mặc định `runtime/images`.

Cài/chạy MariaDB rồi dùng tài khoản quản trị tạo database và user, ví dụ:

```sql
CREATE DATABASE IF NOT EXISTS lane_demo CHARACTER SET utf8mb4;
CREATE USER IF NOT EXISTS 'vehicle_ai'@'localhost' IDENTIFIED BY 'change-me';
GRANT SELECT, INSERT, UPDATE, CREATE, ALTER ON lane_demo.* TO 'vehicle_ai'@'localhost';
```

Thay mật khẩu ví dụ bằng mật khẩu của bạn, rồi điền `.env`:

```dotenv
MARIADB_HOST=127.0.0.1
MARIADB_PORT=3306
MARIADB_DATABASE=lane_demo
MARIADB_USER=vehicle_ai
MARIADB_PASSWORD="change-me"
```

Ứng dụng tự tạo bảng `events` trong database đã có. Khi chạy Docker, cấu hình
host và quyền truy cập MariaDB phù hợp với địa chỉ container; database trên
Windows host dùng `host.docker.internal`.
Nếu MariaDB không kết nối được, máy chủ sẽ báo lỗi lúc khởi động.

Kết quả được lưu ngay khi sự kiện chốt. Nếu trang báo lỗi database, kiểm tra
cấu hình và kết nối trước khi chạy lại. Nếu ảnh đã bị xóa khỏi ổ đĩa,
API ảnh trả 404 nhưng thông tin sự kiện vẫn còn trong database.

## 8. Docker

```powershell
docker compose build
docker compose --profile ui up -d ui
```

Web dùng cổng 8000. Đường dẫn model/ROI phải tồn tại trong container.

## 9. Kiểm tra

```powershell
.\.venv312\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv312\Scripts\python.exe -m unittest discover -s tests -p test_web.py
.\.venv312\Scripts\python.exe -m unittest discover -s tests -p test_pipeline_stop.py
.\.venv312\Scripts\python.exe -m ruff check .
```

API chính: `GET /api/status`, `GET /api/frame`, `GET /api/events`,
`GET /api/events.csv`, `POST /api/analyze` (multipart), `POST /api/stop`.
