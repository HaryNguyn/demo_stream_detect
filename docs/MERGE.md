# So sánh và gộp mã

Nguồn: https://github.com/HaryNguyn/demo_stream_detect
Commit: `e7a5231` (Initial commit: lane video detection demo).
Thư mục local không có Git; đây là gộp mã theo nội dung, không phải three-way merge.
Mã trước merge được sao lưu tại `../../backup-before-merge-20261119`.

| Thành phần | Local trước merge | GitHub | Kết quả gộp |
|---|---|---|---|
| Lưu sự kiện | SQLite | MariaDB bắt buộc | SQLite mặc định, MariaDB tùy chọn |
| Ảnh | Chỉ preview | JPEG/base64, top 3 | Lưu JPEG + metadata, tối đa 30 ứng viên/loại/track |
| Thư mục ảnh | Không có | `D:\VDTC11` | `runtime/images` hoặc `VDTC_IMAGE_ROOT` |
| Nguồn video | OpenCV/FFmpeg | Proxy, retry, báo lỗi | Tích hợp; reconnect HTTP chỉ áp dụng HTTP |
| OCR, model | CCT-VN, YOLO | Giống local | Giữ nguyên |
| Tài liệu | SQLite | MariaDB, có thông tin riêng máy tác giả | Giữ hướng dẫn local và bổ sung tài liệu này |

`.gitignore`, `recognition.py`, `docs/THUAT_TOAN_XU_LY.md` chỉ khác xuống dòng.
Các trọng số và YAML model giống nhau theo SHA-256.
Giữ thư mục `third_party` local; các mục tương ứng trên GitHub là gitlink.

## Chạy mặc định

Trong thư mục chứa `app.py`:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pip install -e .
$env:EVENT_STORE = "sqlite"
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Dữ liệu SQLite hiện có vẫn được đọc. Sự kiện mới có `Xe ID`, điểm hãng,
điểm biển và `images` chứa đường dẫn JPEG. CSV hỗ trợ cả sự kiện cũ và mới.
Ảnh ghi sau khi ghi sự kiện; nếu ghi ảnh lỗi, sự kiện vẫn tồn tại.
Xóa sự kiện không tự xóa ảnh; sao lưu cả database và thư mục ảnh khi cần.

## MariaDB tùy chọn

Có thể điền cấu hình vào `.env` cạnh `app.py`, rồi đổi `EVENT_STORE=mariadb`.
Ứng dụng tự đọc file này; biến môi trường PowerShell có ưu tiên cao hơn.
Khởi động lại ứng dụng sau khi sửa cấu hình. `.env` được loại khỏi Git.

Tạo database trước bằng tài khoản quản trị:

```sql
CREATE DATABASE IF NOT EXISTS lane_demo CHARACTER SET utf8mb4;
```

Cấu hình tài khoản có quyền tạo/đọc/ghi bảng trong database này:

```powershell
$env:EVENT_STORE = "mariadb"
$env:MARIADB_HOST = "127.0.0.1"
$env:MARIADB_PORT = "3306"
$env:MARIADB_DATABASE = "lane_demo"
$env:MARIADB_USER = "<tai-khoan>"
$env:MARIADB_PASSWORD = "<mat-khau>"
$env:VDTC_IMAGE_ROOT = "D:\VDTC11"
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Ứng dụng tự tạo bảng `events`; bổ sung `xe_id` nếu bảng theo tài liệu upstream
thiếu cột này. Chuyển backend không tự chuyển lịch sử SQLite sang MariaDB.
Không áp dụng các tuyên bố migration dữ liệu riêng máy tác giả trong tài liệu upstream.

## Các sửa lỗi khi tích hợp

- Giới hạn bộ nhớ ứng viên ảnh cho track dài; xuất tối đa 3 ảnh mỗi loại.
- Vẫn lấy ảnh xe khi không bật model logo/biển.
- Đọc stderr nền có bộ đệm giới hạn để FFmpeg/yt-dlp không nghẽn pipe.
- FFmpeg fallback cho file local không dùng các cờ HTTP hoặc `nobuffer`.
- Dùng đường dẫn ảnh phù hợp máy local và xử lý tên dành riêng trên Windows.
- Giữ API dataframe tương thích Streamlit 1.40 như requirements hiện có.
- SQLite không lưu base64 vào payload; ảnh nằm trên đĩa.

## Kiểm tra

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s tests -v
```

Kiểm thử dùng database tạm và tiến trình giả lập; cần MariaDB/nguồn stream thực
để kiểm tra kết nối tích hợp trên môi trường triển khai.
