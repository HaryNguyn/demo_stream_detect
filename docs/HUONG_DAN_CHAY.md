# Hướng dẫn chạy và xóa dữ liệu trên Windows

Các lệnh dưới đây dùng trong **PowerShell**.

## 1. Mở đúng thư mục

```powershell
cd C:\Users\ADMIN\Downloads\lane_video_demo\lane_video_demo
```

Đây là thư mục chứa `app.py` và `requirements.txt`.

## 2. Cài đặt lần đầu

Máy hiện tại đã có Python 3.12 và môi trường `.venv312` đã cài thư viện.
Chỉ thực hiện bước này khi thiết lập lại hoặc cài trên máy mới.

Kiểm tra các phiên bản Python:

```powershell
py --list-paths
```

Tạo môi trường bằng Python 3.12:

```powershell
py -3.12 -m venv .venv312
```

Cài thư viện và chờ đến khi lệnh hoàn tất thành công:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pip install -e .
```

Đọc URL stream cần FFmpeg trong `PATH`. Link trang web dùng `yt-dlp`;
YouTube có thể cần thêm Node.js. Kiểm tra:

```powershell
ffmpeg -version
node --version
```

## 3. Chạy ứng dụng

```powershell
cd C:\Users\ADMIN\Downloads\lane_video_demo\lane_video_demo
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Mở địa chỉ Streamlit in trong terminal, thường là:

```text
http://localhost:8501
```

- Tải video lên hoặc nhập URL.
- Với link trang web, chọn **Trang web (yt-dlp)**. Chỉ hỗ trợ các nguồn mà yt-dlp truy cập được.
- Mặc định phân tích 60 giây nội dung. Nhập `0` để xử lý đến hết nguồn; stream liên tục sẽ không tự kết thúc.
- Bấm **Phân tích video**.
- Dùng `.venv312`, không dùng đường dẫn `.venv` cũ.

## 4. Dừng ứng dụng

Tại cửa sổ PowerShell đang chạy ứng dụng, nhấn:

```text
Ctrl+C
```

Chờ ứng dụng dừng và PowerShell hiện lại dấu nhắc lệnh.

## 5. Dữ liệu được lưu ở đâu?

Sự kiện được lưu trong SQLite:

```text
C:\Users\ADMIN\Downloads\lane_video_demo\lane_video_demo\runtime\events.sqlite3
```

Database lưu lịch sử qua các lần chạy. Giao diện và CSV hiện hiển thị/xuất
tối đa 200 sự kiện gần nhất; database giữ toàn bộ sự kiện đã ghi.

Video tải lên được xử lý trong thư mục tạm và được dọn khi phiên xử lý kết thúc bình thường.
Ảnh xe/biển số được chọn và lưu tại `runtime/images`, kèm `metadata.json`.
Biến `VDTC_IMAGE_ROOT` cho phép đổi thư mục. Xem [cấu hình MariaDB và bản merge](MERGE.md).

## 6. Xóa toàn bộ lịch sử sự kiện

**Thao tác này xóa toàn bộ sự kiện đã lưu.** Dừng ứng dụng bằng `Ctrl+C` trước khi xóa.

```powershell
cd C:\Users\ADMIN\Downloads\lane_video_demo\lane_video_demo
Remove-Item -LiteralPath .\runtime\events.sqlite3 -ErrorAction SilentlyContinue
```

Kiểm tra:

```powershell
Test-Path .\runtime\events.sqlite3
```

Kết quả `False` nghĩa là file đã được xóa hoặc chưa tồn tại.
Nếu xuất hiện lỗi file đang được sử dụng, đóng các tiến trình ứng dụng/công cụ đang mở database rồi thử lại.

Chạy lại:

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Ứng dụng tự tạo database rỗng. Refresh trang trình duyệt hoặc mở phiên mới nếu còn thấy bảng cũ.
Việc xóa database không xóa model, mã nguồn, video gốc hay CSV đã tải xuống.

## 7. Sao lưu trước khi xóa (tùy chọn)

Dừng ứng dụng rồi chạy:

```powershell
Copy-Item -LiteralPath .\runtime\events.sqlite3 -Destination ".\runtime\events-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss').sqlite3"
```

## 8. Lỗi thường gặp

### `No module named streamlit`

Cài thư viện vào đúng môi trường và đợi lệnh hoàn tất:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pip install -e .

Remove-Item .\runtime\events.sqlite3
```

### `No suitable Python runtime found`

Kiểm tra bằng `py --list-paths`. Các lệnh trong tài liệu dùng Python 3.12,
không yêu cầu Python 3.11.

### `No Python at ... VDTC-WHO ...`

Đó là môi trường cũ được chép từ máy khác. Dùng `.venv312` đã tạo trên máy hiện tại.

### Không tìm thấy `app.py` hoặc `requirements.txt`

Chạy lại lệnh `cd` ở mục 1. Thư mục dự án có hai cấp cùng tên `lane_video_demo`.
