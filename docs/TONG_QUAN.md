# Vehicle AI — Tổng quan và kiến trúc

Vehicle AI là ứng dụng thử nghiệm nhận diện xe, loại xe, biển số Việt Nam và hãng xe từ video, luồng camera IP hoặc ảnh. Web dùng để xem trực tiếp và tra lịch sử; CLI dùng để chạy thí nghiệm, huấn luyện và đánh giá.

- [Cài đặt và vận hành](VAN_HANH.md)
- [Dữ liệu, huấn luyện và đánh giá](PHAT_TRIEN_VA_DANH_GIA.md)

## Luồng xử lý

1. **Đọc nguồn.** OpenCV đọc video cục bộ; FFmpeg hỗ trợ luồng trực tiếp, yt-dlp lấy luồng từ các trang được hỗ trợ. Trên web, phát hình và suy luận chạy riêng: hàng đợi chỉ giữ frame mới nhất, nên AI có thể bỏ frame nếu không theo kịp. CLI mặc định không bỏ frame theo tải CPU; dùng `--realtime` để thử chế độ tương tự web.
2. **Phát hiện và theo dõi.** YOLO phát hiện ô tô con, xe máy, xe buýt và xe tải; ByteTrack giữ ID riêng trong từng nguồn. ROI là đa giác tọa độ chuẩn hóa: hệ thống lọc theo tâm hộp xe. Nếu bật `crop_vehicle_roi`, detector chạy trên hộp bao ROI có nới lề rồi quy tọa độ về ảnh gốc.
3. **Đọc biển và nhận diện hãng.** Hệ thống lấy mẫu theo track, giới hạn số xe xử lý mỗi frame và ưu tiên xe lâu chưa lấy mẫu. Detector biển và logo chạy trên ảnh cắt của xe. Sau khi lọc kích thước/độ nét, OCR đọc các crop biển phù hợp và gom các crop cùng frame thành batch khi backend hỗ trợ.
4. **Tổng hợp và lưu.** Biển số, loại xe và hãng được tổng hợp qua nhiều mẫu; biển chỉ được xác nhận khi đủ phiếu và có ảnh đạt ngưỡng bằng chứng. Biển đã xác nhận không cần OCR lại trên cùng track. Sự kiện chốt khi xe mất dấu quá ngưỡng (2 giây trong cấu hình Việt Nam), hết nguồn hoặc dừng phiên. Trong cùng nguồn và lần chạy, các track đọc cùng biển chuẩn hóa có thể được gộp; track không có biển vẫn được giữ riêng khi bật fallback.

Web lưu sự kiện vào MariaDB và ảnh bằng chứng ra đĩa; danh sách mặc định và CSV chỉ trả sự kiện có chuỗi biển, kể cả khi chuỗi mang nhãn tin cậy thấp. CLI lưu cả sự kiện thô và sự kiện sau gộp dưới dạng JSONL, độc lập với MariaDB. Trang `/plate` xử lý ảnh riêng và không ghi sự kiện video.

## Model và cấu hình

- `models/yolo11n.pt`: phát hiện và phân loại xe.
- `models/best_stripped.onnx`: phát hiện biển; `models/ocr_model.onnx` + `models/ocr_model.yaml`: đọc ký tự biển Việt Nam bằng OCR kiểu CCT.
- `models/logo.pt`: nhận diện logo của 26 hãng đã định nghĩa, không bao phủ mọi hãng.
- `configs/vietnam-ocr.yaml`: cấu hình mẫu cho bộ model trên, mặc định CPU; `configs/baseline.yaml` và `configs/candidate.example.yaml` phục vụ thử nghiệm. Không nên giả định các cấu hình có cùng ngưỡng.

Trong `vietnam-ocr.yaml`, detector xe và biển dùng đầu vào 640 px; ngân sách nhận diện là 2 xe mỗi frame AI, mẫu cách nhau tối thiểu 0,5 giây, crop xe/biển rộng tối thiểu 120/32 px. Chỉnh ngưỡng theo video thực tế và đo lại; nới ngưỡng có thể tăng số xe đọc được nhưng cũng tăng đọc sai.

## Tìm mã nguồn

- `app.py`, `src/vehicle_ai/web.py`, `web_static/`: máy chủ Flask, API và giao diện HTML/CSS/JS.
- Trong `src/vehicle_ai/`: `capture.py`, `pipeline.py`, `session.py` điều phối nguồn và track; `inference.py`, `ocr.py`, `aggregation.py`, `evidence.py` xử lý model, bỏ phiếu và bằng chứng; `persistence/mariadb.py` và `storage.py` lưu kết quả web và CLI.
- `data/` chứa dữ liệu/nhãn/ROI; `runtime/` chứa dữ liệu sinh ra khi chạy; `scripts/` và `tests/` chứa công cụ đo và kiểm thử.

## Giới hạn và nguồn tham khảo

Đây là bản demo local, không phải hệ thống giám sát hoàn chỉnh. Web chạy mỗi lần một phiên; nguồn trong phiên chạy tuần tự. Chưa có trigger, webhook/outbox hay đồng bộ thời gian giữa camera. Mất dấu có thể làm đổi ID; ROI, chất lượng hình và tốc độ máy đều ảnh hưởng kết quả. Không dùng kết quả nhận diện thay cho kiểm chứng thủ công.

Codebase kế thừa [demo_stream_detect](https://github.com/HaryNguyn/demo_stream_detect) (`e7a5231`). Hai nguồn tham khảo không được import khi chạy: [LicensePlateDetection](https://github.com/ddkhoa2605/LicensePlateDetection) (`f1c732e2270becc44df1de6529372f63ef251076`) và [cars-logos-plates-detection](https://github.com/TeamXGroup/cars-logos-plates-detection) (`b2d35e76f8d220677a54ee5437635748427a7f1a`). Xem [manifest model](../models/manifest.yaml) và giấy phép của từng nguồn trước khi phân phối lại.
