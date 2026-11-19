# Demo nhận diện xe từ video

Hướng dẫn PowerShell chi tiết: [Chạy ứng dụng, sao lưu và xóa dữ liệu](HUONG_DAN_CHAY.md).

## Bộ tài liệu dự án

- [Tài liệu dự án](docs/TAI_LIEU_DU_AN.md): phạm vi, kiến trúc, model, dữ liệu sự kiện, kiểm thử và lộ trình AI Box.
- [Thuật toán xử lý](docs/THUAT_TOAN_XU_LY.md): luồng video, ByteTrack, lấy mẫu, OCR, tổng hợp kết quả và chốt sự kiện.
- [Hướng dẫn vận hành](HUONG_DAN_CHAY.md): lệnh cài đặt, chạy, sao lưu và xóa dữ liệu.

## AI Box — bản thử nghiệm sự kiện

- Tải video hoặc nhập URL RTSP/HTTP/HLS. Với URL trang web, chọn **Trang web (yt-dlp)**; chỉ các nền tảng được yt-dlp hỗ trợ và truy cập được mới hoạt động.
- Giữ nguyên bộ model demo, sửa giải mã CCT theo vị trí ký tự để giữ số lặp.
- Sự kiện được chốt sau 2 giây video mất dấu track hoặc cuối phiên, hiển thị ngay và lưu bền vững trong MariaDB.
- Giao diện hiển thị/xuất tối đa 200 sự kiện gần nhất, gồm cả phiên trước. MariaDB giữ toàn bộ sự kiện.
- Mặc định xử lý 60 giây nội dung mỗi nguồn; đặt 0 để xử lý đến hết nguồn. Đây là thời lượng nội dung, không phải thời gian thực thi. Stream liên tục không tự kết thúc.
- Thời điểm xuất hiện/kết thúc là thời gian tương đối trong nguồn; `recorded_at` là thời điểm ghi sự kiện UTC.
- Bản thử nghiệm chưa có vạch đếm, chống trùng giữa các track, webhook, reconnect hay nút dừng tác vụ nền. Xe không đọc được biển vẫn có sự kiện.
- Nếu `.venv` được chép từ máy khác, hãy tạo môi trường mới bằng Python trên máy hiện tại.

Ứng dụng Streamlit chạy trên máy của bạn. Chọn video, xem khung hình có hộp nhận diện trong lúc phân tích, xem bảng theo camera và tải CSV. Mỗi video giữ một bộ ByteTrack riêng. Một xe có thể tạo thêm track nếu bị che khuất lâu.

Có thể nhập link stream ở ô **Hoặc nhập link stream**. YouTube được đọc qua pipeline `yt-dlp → FFmpeg`, không lưu thành file; RTSP/HTTP cũng được đọc qua FFmpeg. Cài `ffmpeg` và Node.js trong PATH để hỗ trợ các nguồn stream.

## Chạy trên Windows (PowerShell)

Dùng Python 3.12 và mở PowerShell trong thư mục chứa app.py.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Trên Linux/macOS, tạo môi trường bằng `python3.12 -m venv .venv312`, rồi thay đường dẫn Python trong hai lệnh cuối bằng `.venv312/bin/python`.

Mở địa chỉ Streamlit in trên terminal, thường là http://localhost:8501. Thử với video ngắn của một làn trước. Khi dùng `yolo11n.pt` mặc định, Ultralytics tải trọng số nếu chưa có sẵn. Nếu chạy offline, đặt file trọng số trên máy rồi điền đường dẫn vào ô **Model loại xe**. Các nhãn loại xe dùng ID theo COCO.

## Bật logo và biển số

- **Logo:** App mặc định dùng [models/logo.pt](./models/logo.pt). Tên nhãn trong trọng số được dùng làm tên hãng.
- **Biển số Việt Nam:** Điền đường dẫn tới `DetectLisence_YOLO11.pt` trong repo [ddkhoa2605/LicensePlateDetection](https://github.com/ddkhoa2605/LicensePlateDetection), rồi chọn engine **CCT-VN ONNX** và cấu hình `cct_s_v1_vn.onnx` cùng `cct_s_v1_vn_plate_config.yaml`.
- **PaddleOCR thay thế:** Chọn engine **PaddleOCR REC**, điền thư mục inference REC và tên model REC.
- `models/logo.pt` hiện chứa trọng số thật; khi tải thêm model qua Git LFS, cần lấy blob thật thay vì file pointer.
- Bộ model biển số Việt Nam đã được đóng gói trong thư mục `models/`: `DetectLisence_YOLO11.pt`, `cct_s_v1_vn.onnx` và `cct_s_v1_vn_plate_config.yaml`. App mặc định chọn CCT-VN ONNX.
- PaddleOCR REC là lựa chọn bổ sung, cần cài riêng PaddleOCR/PaddlePaddle tương thích. Mặc định dùng CCT-VN ONNX. Model CRNN ONNX cần adapter riêng với đúng tiền xử lý và bảng ký tự.
- Cột hãng hoặc biển số để trống khi thiếu model tương ứng hoặc không có kết quả đạt ngưỡng. OCR mẫu ưu tiên biển số có nhiều lần đọc trùng; hỗ trợ biển hai dòng phụ thuộc vào cách model REC được huấn luyện và chuẩn bị ảnh.

## Phạm vi demo

Mỗi nguồn được xử lý như một camera, tuần tự. Bảng trả một dòng cho mỗi track đã chốt. Muốn ghép cùng một lượt xe giữa các camera cần dấu thời gian đồng bộ và sự kiện loop/radar hoặc quy tắc ghép theo vị trí làn. Demo có nguồn RTSP và SQLite cục bộ, chưa kết nối LaneApp.

## Cấu trúc mã nguồn

- `app.py`: giao diện Streamlit, chọn nguồn và xuất CSV.
- `pipeline.py`: theo dõi xe, lấy mẫu nhận diện và chốt sự kiện.
- `recognition.py`: adapter CCT-VN và chuẩn hóa kết quả OCR.
- `sources.py`: nguồn video/FFmpeg/yt-dlp và quản lý tiến trình.
- `events.py`: lưu và đọc sự kiện SQLite.
