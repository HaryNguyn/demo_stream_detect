# Tài liệu kỹ thuật dự án AI Box — Lane Video Demo

Ngày cập nhật: 18/11/2026. Trạng thái: bản thử nghiệm chạy cục bộ.

## 1. Mục tiêu

Xây dựng nền thử nghiệm cho AI Box quan sát phương tiện từ video/stream,
nhận diện loại xe, hãng và biển số; chuyển kết quả thành sự kiện để tiến tới tích hợp hệ thống làn/bãi đỗ.

Giai đoạn hiện tại chưa chốt phần cứng. Ưu tiên kiểm chứng luồng xử lý trên máy tính,
video có sẵn và URL stream trước khi tối ưu cho thiết bị triển khai.

## 2. Phạm vi đã có

| Nhóm | Chức năng |
|---|---|
| Nguồn | Nhiều file video tải lên và một URL nhập trên giao diện |
| Xử lý | Các nguồn chạy tuần tự, mỗi nguồn có trạng thái tracking riêng |
| Phát hiện | Ô tô con, xe máy, xe buýt, xe tải |
| Theo dõi | ByteTrack theo xe |
| Thuộc tính | Hãng từ logo và biển số từ detector + OCR |
| Sự kiện | Chốt track mất dấu hoặc khi hết phiên bình thường |
| Lưu trữ | SQLite cục bộ, giữ dữ liệu qua các lần chạy |
| Giao diện | Xem frame, nhãn nhận diện, crop và thống kê chẩn đoán |
| Xuất dữ liệu | CSV UTF-8 BOM của tối đa 200 sự kiện gần nhất |

Chưa có API REST, webhook, worker nền nhiều camera, tự reconnect, vạch đếm,
xác định chiều di chuyển, ghép xe giữa camera hoặc ảnh bằng chứng lưu trên đĩa.

## 3. Quan hệ với VN-ANPR

Hướng đã thống nhất là giữ model của demo làm bộ nhận diện chính và phát triển
cấu trúc dịch vụ dựa trên kinh nghiệm từ `vn_anpr`.

**Hiện chưa hoàn tất hợp nhất hai codebase.** Dự án này chưa import package `anpr`,
chưa dùng API/worker của VN-ANPR, chưa có nắn phối cảnh, văn phạm hoặc bộ quyết định
`ACCEPT/UNREAD` của dự án đó.

CRNN được huấn luyện bằng ảnh tổng hợp bên VN-ANPR không phải OCR mặc định ở đây.

## 4. Kiến trúc module

```text
app.py
 ├── sources.stream_source(): phân loại URL từ giao diện
 ├── recognition.VietnamesePlateRecognizer: nạp OCR CCT
 ├── pipeline.optional_yolo(): nạp model tùy chọn
 ├── pipeline.analyze_video()
 │    ├── sources.open_capture()
 │    ├── YOLO + ByteTrack
 │    ├── recognize_vehicle()
 │    │    └── recognition.inspect_plate()
 │    ├── on_frame: cập nhật video xem trước
 │    ├── on_diagnostic: cập nhật mẫu nhận diện
 │    └── on_event: ghi sự kiện
 └── events.EventStore: SQLite
```

### Cấu trúc thư mục

```text
lane_video_demo/
├── app.py
├── pipeline.py
├── recognition.py
├── sources.py
├── events.py
├── requirements.txt
├── README.md
├── HUONG_DAN_CHAY.md
├── docs/
│   ├── TAI_LIEU_DU_AN.md
│   └── THUAT_TOAN_XU_LY.md
├── models/
│   ├── logo.pt
│   ├── DetectLisence_YOLO11.pt
│   ├── cct_s_v1_vn.onnx
│   └── cct_s_v1_vn_plate_config.yaml
├── yolo11n.pt
├── runtime/                    # tệp tạm và cấu hình runtime
└── .venv312/                  # môi trường Python cục bộ
```

Đường dẫn gốc chứa code trên máy hiện tại:

```text
C:\Users\ADMIN\Downloads\lane_video_demo\lane_video_demo
```

## 5. Công nghệ và model

| Thành phần | Công nghệ/model | Vai trò |
|---|---|---|
| UI | Streamlit | Cấu hình, xem kết quả, tải CSV |
| Xử lý ảnh | OpenCV, NumPy | Đọc file, crop, resize, vẽ |
| Xe | Ultralytics `yolo11n.pt` | Detection 4 lớp COCO được chọn |
| Tracker | ByteTrack của Ultralytics | Gán ID qua frame |
| Hãng | `models/logo.pt` | Nhận diện logo, lấy tên hãng từ nhãn |
| Biển | `models/DetectLisence_YOLO11.pt` | Phát hiện hộp biển |
| OCR chính | CCT-VN ONNX | Nhận dạng ký tự theo vị trí |
| OCR tùy chọn | PaddleOCR REC | Thay thế khi tự cấu hình model và dependencies |
| Stream trực tiếp | FFmpeg | Giải mã và xuất raw frame |
| Trang web | yt-dlp, Node.js khi cần | Truy cập nguồn video được hỗ trợ |
| Lưu trữ | SQLite chuẩn Python | Sự kiện cục bộ |

Hợp đồng tensor và giải mã model được mô tả trong [thuật toán xử lý](THUAT_TOAN_XU_LY.md).
Không thể thay bất kỳ file `.onnx` vào adapter hiện có mà không kiểm tra tensor,
alphabet, kiểu dữ liệu và tiền xử lý.

Các thư viện chính được khai báo trong `requirements.txt`; hiện là khoảng phiên bản,
chưa phải lockfile tái lập hoàn toàn. Các gói xử lý ảnh/PyTorch đi theo dependencies của Ultralytics.
PaddleOCR/PaddlePaddle không nằm trong bộ cài mặc định.

## 6. Vòng đời một lần chạy

1. Streamlit khởi tạo kho sự kiện và hiển thị lịch sử gần nhất.
2. Người dùng chọn model, video/URL và giới hạn thời lượng.
3. Khi bấm phân tích, app nạp logo, detector biển và OCR.
4. File tải lên được ghi ra thư mục tạm theo từng nguồn.
5. Mỗi nguồn tạo model xe/tracker riêng và được xử lý tới giới hạn hoặc hết nguồn.
6. Các callback hiển thị frame, chẩn đoán, ghi sự kiện ngay khi track được chốt.
7. Sau phiên thành công, app đọc 200 sự kiện gần nhất để hiển thị và xuất CSV.
8. Capture được giải phóng; các file tạm được dọn khi thoát context bình thường.

Model logo/biển/OCR được dùng chung giữa các nguồn trong cùng lần bấm,
nhưng chưa có cache model qua các lần bấm. Giao diện xử lý đồng bộ, không có nút dừng worker riêng.

Nếu nguồn đầu là stream liên tục và giới hạn là 0, các nguồn sau không được xử lý cho tới khi nguồn đầu kết thúc.

## 7. Hợp đồng sự kiện hiện tại

Ví dụ minh họa, không phải dữ liệu đo thực tế:

```json
{
  "event_id": "cfbc0e11-4563-4a4a-b423-4f67df8917bf",
  "recorded_at": "2026-11-18T15:00:00+00:00",
  "Camera": "1 · cong_vao.mp4",
  "Track ID": 12,
  "Xuất hiện (giây)": 2.4,
  "Loại xe": "Ô tô con",
  "Hãng": "toyota",
  "Biển số": "30A11234",
  "Kết thúc (giây)": 5.8
}
```

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `event_id` | string | UUID mới cho mỗi lần ghi sự kiện |
| `recorded_at` | string | ISO 8601 UTC, thời gian ghi |
| `Camera` | string | Số thứ tự nguồn và tên file/URL |
| `Track ID` | integer | ID của tracker trong nguồn |
| `Xuất hiện (giây)` | number | Thời gian nội dung lúc track được thấy lần đầu |
| `Kết thúc (giây)` | number | Thời gian nội dung lúc track được thấy lần cuối |
| `Loại xe` | string | Nhãn thắng theo tổng confidence |
| `Hãng` | string | Nhãn logo tốt nhất, có thể rỗng |
| `Biển số` | string | Chuỗi thắng trong các lần OCR đạt ngưỡng, có thể rỗng |

`Camera` chưa phải ID camera ổn định; có thể chứa URL đầy đủ do người dùng nhập.
`Track ID` không phải khóa duy nhất toàn hệ thống. Chạy lại một video tạo UUID/sự kiện mới,
không tự chống trùng với lần chạy trước.

Chưa lưu confidence, crop, trạng thái nhận diện, lý do chốt, chiều xe,
phiên bản model hoặc thống kê chẩn đoán trong payload sự kiện.

## 8. Thiết kế lưu trữ

MariaDB: database `lane_demo`, bảng `events`.

```sql
CREATE TABLE events (
    event_id CHAR(36) NOT NULL PRIMARY KEY,
    recorded_at DATETIME(6) NOT NULL,
    camera VARCHAR(255),
    track_id INT UNSIGNED,
    started_at DOUBLE,
    ended_at DOUBLE,
    vehicle_type VARCHAR(100),
    brand VARCHAR(100),
    brand_score DOUBLE,
    plate VARCHAR(20),
    plate_score DOUBLE,
    images JSON,
    INDEX ix_recorded_at (recorded_at)
) ENGINE=InnoDB;
```

- Các trường camera, track ID, thời gian, loại xe, hãng, biển số và điểm tin cậy được lưu riêng để truy vấn/filter.
- `images` lưu danh sách đường dẫn ảnh xe và ảnh biển số dưới dạng JSON.
- File JPEG được lưu trong `D:\VDTC11\<ngày>\<camera>\<Xe ID>\<event_id>`.
- Mỗi thư mục event chứa `metadata.json` và các ảnh `detected_*.jpg`, `plate_*.jpg`.
- `xe_id` là biển số nếu OCR đã nhận được; nếu chưa nhận được thì dùng `TRACK-<track_id>`.
- Mỗi track giữ tối đa 3 ảnh xe tốt nhất và 3 ảnh biển số tốt nhất.
- Điểm ảnh xe:
  `0.4 × confidence + 0.3 × size_ratio + 0.3 × sharpness_norm`.
- `size_ratio` là diện tích bbox chia diện tích khung hình/crop tương ứng.
- `sharpness_norm` là độ sắc nét Laplacian chuẩn hóa về khoảng `[0, 1]`.
- Các ảnh được chọn phải cách nhau ít nhất 5 frame; nếu track quá ngắn, bổ sung theo điểm cao nhất.
- `EventStore.append()` ghi trực tiếp bằng PyMySQL.
- `recent(limit=200)` đọc theo `recorded_at DESC`.
- 103 sự kiện SQLite cũ đã được migration sang MariaDB.
- Chưa có retention tự động hoặc phân trang.
- Không lưu video lâu dài hay ảnh bằng chứng; crop chẩn đoán ở bộ nhớ.

Lệnh chạy, sao lưu và xóa dữ liệu: [HUONG_DAN_CHAY.md](../HUONG_DAN_CHAY.md).

## 9. Hướng dẫn sử dụng nhanh

```powershell
cd C:\Users\ADMIN\Downloads\lane_video_demo\lane_video_demo
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Mở URL mà Streamlit in ra, thường `http://localhost:8501`.

Nên bắt đầu bằng video ngắn có xe và biển nhìn rõ; giữ giới hạn 60 giây.
Xem lần lượt video, ảnh crop và các bộ đếm OCR trước khi kiểm tra CSV.

Để tắt logo hoặc biển, để trống đường dẫn model tương ứng. Nếu có detector biển,
app yêu cầu cấu hình OCR đầy đủ. Sai đường dẫn hoặc model không nạp được sẽ báo lỗi,
không âm thầm bỏ qua.

## 10. Chẩn đoán biển số trống

| Quan sát | Cần kiểm tra |
|---|---|
| Không có mẫu xe | Nguồn có frame không; detector có track ID không; model tùy chọn đã bật chưa |
| Có mẫu xe, không phát hiện biển | Crop có phần đầu/đuôi xe không; biển có đủ kích thước không; điều kiện camera và detector |
| Có biển, OCR không đạt | Crop thực tế, độ mờ/góc nghiêng, chuỗi và score OCR, cấu hình CCT |
| OCR đạt nhưng chưa có sự kiện | Xe còn đang trong track; chờ mất dấu hoặc hết phiên |
| Có biển trên một xe nhưng crop đang khác | Chẩn đoán hiển thị mẫu gần nhất, có thể thuộc track khác |
| CSV có kết quả của video trước | CSV lấy lịch sử gần nhất toàn database, không chỉ lần chạy hiện tại |
| Một xe xuất hiện nhiều lần | Track bị tách hoặc video được xử lý lại |
| Không mở được URL | FFmpeg, hỗ trợ yt-dlp, khả năng truy cập nguồn; chưa có reconnect |

Không nên hạ ngưỡng OCR chỉ để tăng số biển có kết quả trước khi kiểm tra tỷ lệ đọc sai.

## 11. Quy trình kiểm thử đề xuất

### 11.1. Kiểm thử chức năng

| Tình huống | Kết quả cần kiểm tra |
|---|---|
| Video không có xe | Kết thúc không sinh sự kiện xe |
| Một xe, biển rõ | Có track, có mẫu biển; đối chiếu chuỗi với nhãn thật |
| Một xe, không nhìn thấy biển | Vẫn có sự kiện xe với biển rỗng |
| Biển có ký tự lặp | Giữ đúng các ký tự lặp, ví dụ `30A11234` |
| OCR điểm thấp | Chẩn đoán hiển thị chuỗi nhưng không dùng làm phiếu biển |
| Nhiều xe cùng frame | Thuộc tính không bị gán sang track khác |
| Xe bị che khuất | Đo số lần đổi track và sự kiện trùng |
| Hết video khi xe còn hiện diện | Track còn lại được chốt |
| Giới hạn 60 giây | Dừng theo thời lượng nội dung, không theo đồng hồ chạy |
| Nhiều file | Xử lý tuần tự, không dùng lại trạng thái tracker giữa nguồn |
| Restart ứng dụng | Lịch sử đã commit còn trong SQLite |
| Xóa database lúc đã dừng | Lần chạy sau tạo database mới |
| Stream không truy cập được | Có lỗi/kết thúc đọc; kiểm tra giải phóng tiến trình |

### 11.2. Đánh giá chất lượng

Chuẩn bị bộ video có nhãn theo lượt xe: loại xe, biển thật, thời điểm xuất hiện/rời,
điều kiện ánh sáng, camera và biển một/hai hàng.

Tránh dùng các frame gần nhau của cùng lượt xe làm cả dữ liệu tinh chỉnh và nghiệm thu.
Giữ cùng video, giới hạn xử lý và model khi so sánh các phiên bản.

Chỉ số nên báo cáo:

```text
Độ phủ biển = số lượt xe có kết quả biển / tổng lượt xe mục tiêu

Độ chính xác trong số có biển =
    số lượt xe có biển đọc đúng hoàn toàn / số lượt xe có kết quả biển

Exact match đầu-cuối =
    số lượt xe có biển đọc đúng hoàn toàn / tổng lượt xe mục tiêu
```

Cần ghép sự kiện với lượt xe có nhãn trước khi tính; không dùng số dòng CSV trực tiếp
thay số xe thực khi có track trùng. Báo cáo thêm xe bỏ sót, sự kiện thừa, độ trễ và tốc độ xử lý.

Hiện chưa có benchmark trên tập video thật có nhãn hoặc độ chính xác cam kết.
Các kiểm tra khởi động UI, nạp model, giải mã và SQLite chỉ xác nhận chức năng tương ứng.
Codebase hiện chưa đóng gói bộ kiểm thử hồi quy tự động riêng.

## 12. Hạn chế kỹ thuật hiện tại

- Đọc frame, inference, cập nhật UI và SQLite nằm trên đường xử lý đồng bộ.
- Tốc độ không theo kịp nguồn có thể tạo độ trễ; chưa có chiến lược bỏ frame cũ.
- Stream qua FFmpeg bị chuẩn hóa xuống 720p.
- Chưa ghi stderr FFmpeg/yt-dlp phục vụ phân tích lỗi.
- Chưa có timeout toàn phiên, tự reconnect hoặc quản lý tác vụ nền.
- Việc chốt sự kiện dựa trên track timeout, chưa tương đương lượt xe qua làn.
- Một mẫu OCR đạt có thể đủ để chọn biển; chưa có grammar/đồng thuận tối thiểu.
- Ngoại lệ giữa phiên có thể khiến track chưa chốt không được lưu.
- SQLite lưu sự kiện nhưng không phải outbox gửi backend.
- Phần cứng, số camera đồng thời và mục tiêu độ trễ chưa xác định.

## 13. Lộ trình phát triển AI Box

Các mục dưới đây là kế hoạch, chưa phải chức năng hiện có.

### Giai đoạn A — Kiểm chứng nhận diện

1. Thu video thực và nhãn chuẩn.
2. Dùng chẩn đoán để tách lỗi detector xe, detector biển và OCR.
3. Đo ảnh hưởng của độ phân giải stream, tần suất lấy mẫu và crop.
4. Chọn ngưỡng bằng kết quả đúng/sai, bổ sung kiểm thử hồi quy.

### Giai đoạn B — Chuẩn hóa sự kiện làn

1. Cấu hình camera ID, ROI và vạch đếm.
2. Xác định hướng, thời điểm qua vạch và quy tắc chốt.
3. Thêm session ID, lý do chốt, confidence, trạng thái và model version.
4. Lưu ảnh bằng chứng, giới hạn dung lượng và xử lý sự kiện trùng.

### Giai đoạn C — Dịch vụ tích hợp

1. Tách UI khỏi worker xử lý.
2. Thêm API cấu hình/tra cứu và webhook.
3. Dùng outbox bền vững, retry và `event_id` để backend xử lý idempotent.
4. Tách đọc camera, suy luận và gửi sự kiện; bổ sung reconnect/health/metrics.
5. Tích hợp các thành phần phù hợp từ VN-ANPR qua adapter cho model demo.

### Giai đoạn D — Tối ưu thiết bị

1. Chọn phần cứng dựa trên số camera, độ phân giải và độ trễ yêu cầu.
2. Benchmark CPU/GPU, bộ nhớ và thời gian chạy liên tục.
3. Cân nhắc batch, xuất model hoặc tăng tốc sau khi có baseline chính xác.
4. Đóng gói phiên bản dependencies/model và quy trình cập nhật.

## 14. Quy tắc khi thay đổi code/model

- Thay OCR: kiểm tra tensor, alphabet, padding, kiểu giải mã và ký tự lặp.
- Thay detector xe: kiểm tra mapping class ID và tracking.
- Thay model biển: kiểm tra nhãn, crop và phân bố confidence.
- Thay ngưỡng: chạy lại cùng bộ video có nhãn.
- Thay schema sự kiện: xem xét tương thích dữ liệu SQLite cũ và xuất CSV.
- Thay đọc stream: kiểm tra EOF, mất kết nối và việc dọn tiến trình con.

## 15. Tài liệu liên quan

- [README tổng quan](../README.md)
- [Thuật toán xử lý chi tiết](THUAT_TOAN_XU_LY.md)
- [Câu lệnh chạy, sao lưu và xóa dữ liệu](../HUONG_DAN_CHAY.md)
