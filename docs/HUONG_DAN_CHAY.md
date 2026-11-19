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

Sự kiện được lưu trong MariaDB database `lane_demo`, bảng `events`.

Giao diện và CSV hiển thị/xuất tối đa 200 sự kiện gần nhất. MariaDB giữ
toàn bộ sự kiện đã ghi.

Ứng dụng mặc định đọc cấu hình từ các biến môi trường `MARIADB_*`. Các giá trị
Windows đã lưu cho tài khoản hiện tại cũng được ứng dụng đọc tự động.

Video tải lên được xử lý trong thư mục tạm và được dọn khi phiên xử lý kết thúc bình thường.
Ảnh crop chẩn đoán hiện chỉ hiển thị trên giao diện, chưa được lưu thành ảnh bằng chứng.

## 6. Xóa toàn bộ lịch sử sự kiện

**Thao tác này xóa toàn bộ sự kiện đã lưu.** Dừng ứng dụng bằng `Ctrl+C` trước khi xóa.

```powershell
& "C:\Program Files\MariaDB 12.3\bin\mariadb.exe" `
  -u lane_app -p1234567 -h 127.0.0.1 -D lane_demo `
  -e "DELETE FROM events;"
```

Chạy lại:

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Refresh trang trình duyệt hoặc mở phiên mới để cập nhật bảng rỗng.
Việc xóa database không xóa model, mã nguồn, video gốc hay CSV đã tải xuống.

## 7. Sao lưu trước khi xóa (tùy chọn)

Dừng ứng dụng rồi chạy:

```powershell
$backup = "events-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss').sql"
& "C:\Program Files\MariaDB 12.3\bin\mariadb-dump.exe" `
  -u lane_app -p1234567 -h 127.0.0.1 lane_demo events > $backup
```

## 8. Lỗi thường gặp

### `No module named streamlit`

Cài thư viện vào đúng môi trường và đợi lệnh hoàn tất:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

### `No suitable Python runtime found`

Kiểm tra bằng `py --list-paths`. Các lệnh trong tài liệu dùng Python 3.12,
không yêu cầu Python 3.11.

### `No Python at ... VDTC-WHO ...`

Đó là môi trường cũ được chép từ máy khác. Dùng `.venv312` đã tạo trên máy hiện tại.

### Không tìm thấy `app.py` hoặc `requirements.txt`

Chạy lại lệnh `cd` ở mục 1. Thư mục dự án có hai cấp cùng tên `lane_video_demo`.
