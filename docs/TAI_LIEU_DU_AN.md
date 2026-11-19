1. Kiến trúc phù hợp
Camera IP
  │ RTSP: luồng hình ảnh
  │ ONVIF: khám phá/cấu hình camera nếu hỗ trợ
  ▼
AI Box
  ├── Quản lý camera, kết nối lại
  ├── Nhận frame, timestamp, bộ đệm
  ├── Nhận diện xe + tracking
  ├── Nhận diện biển số + loại xe + logo/hãng
  ├── Tổng hợp kết quả và chốt sự kiện
  ├── Lưu ảnh bằng chứng + sự kiện cục bộ
  └── Hàng đợi gửi API, retry khi mất mạng
          │ HTTPS
          ▼
Hệ thống trung tâm
  ├── API tiếp nhận sự kiện
  ├── Lưu dữ liệu và ảnh
  └── Hiển thị, tra cứu, tích hợp nghiệp vụ
Box phải hoạt động được khi không mở giao diện và khi mất kết nối với server.
2. Chia module theo nhiệm vụ thực tế
Module	Trách nhiệm
camera_manager	Quản lý camera ID, RTSP, trạng thái, reconnect
capture	Giải mã video, timestamp, bộ đệm frame
inference	Model nhận diện xe, biển, OCR, logo
tracking	Theo dõi xe riêng cho từng camera
event_engine	Tổng hợp nhiều frame, xác định thời điểm chốt sự kiện
evidence_store	Lưu ảnh toàn cảnh, crop xe, crop biển
event_store	Lưu sự kiện và hàng đợi gửi bền vững
api_dispatcher	Gửi kết quả về server, retry, chống gửi trùng
management_api	Cấu hình camera, trạng thái box, chẩn đoán
test_tools	Chạy file video hoặc nguồn livestream thử nghiệm
Logic nhận diện nên dùng chung cho camera và video test, chỉ khác adapter đầu vào.
3. Hợp đồng API nên thiết kế trước
Ví dụ AI Box gửi:
POST /api/v1/recognition-events
Authorization: Bearer <device-token>
Idempotency-Key: <event-id>
{
  "schema_version": "1.0",
  "event_id": "51d3aeb5-1c06-4ff7-b12b-72492206eb32",
  "box_id": "BOX-001",
  "camera_id": "CAM-01",
  "lane_id": "LANE-01",
  "event_type": "vehicle_observed",
  "started_at": "2026-11-19T13:00:00.000Z",
  "ended_at": "2026-11-19T13:00:02.500Z",
  "plate": {
    "text": "30A12345",
    "confidence": 0.96,
    "status": "recognized"
  },
  "vehicle": {
    "type": "car",
    "confidence": 0.94
  },
  "brand": {
    "name": "Toyota",
    "confidence": 0.91,
    "source": "logo"
  },
  "evidence": [
    {
      "kind": "vehicle",
      "object_key": "BOX-001/20261119/event-id/vehicle.jpg"
    },
    {
      "kind": "plate",
      "object_key": "BOX-001/20261119/event-id/plate.jpg"
    }
  ],
  "model_version": "release-001"
}
Ở đây object_key là mã ảnh sau khi upload, không phải đường dẫn ổ đĩa trên Box. Nên gửi ảnh bằng API upload riêng hoặc multipart, tùy hợp đồng server.
Không đọc được biển vẫn gửi sự kiện với text: null và trạng thái unread; không biến một kết quả OCR không chắc chắn thành biển số chính thức.
4. Cơ chế gửi kết quả là phần cốt lõi
Luồng nên là:
Chốt sự kiện
    ↓
Lưu event + outbox trong cùng transaction
    ↓
Dispatcher upload ảnh và gửi API
    ↓
Server xác nhận đã lưu
    ↓
Đánh dấu delivered
- Mất mạng: tiếp tục nhận diện, lưu chờ gửi.
- Khởi động lại: đọc hàng đợi và gửi tiếp.
- Server đã nhận nhưng Box chưa nhận phản hồi: gửi lại cùng event_id.
- Server dùng ràng buộc unique để không tạo bản ghi trùng.
- Không để thời gian gọi API làm đứng luồng camera.
Với Box độc lập, SQLite thường đủ cho lưu đệm sự kiện/outbox. MariaDB có thể đặt ở hệ thống trung tâm; không bắt buộc cài MariaDB trên mỗi Box.
5. “Ghi hình” cần tách khỏi “nhận diện”
Nếu chỉ cần bằng chứng lượt xe:
- Lưu ảnh toàn cảnh, ảnh xe, ảnh biển.
- Có thể thêm clip ngắn trước/sau sự kiện bằng bộ đệm vòng.
Nếu cần ghi hình liên tục:
- Tách recorder khỏi pipeline inference.
- Khi phù hợp, ghi trực tiếp luồng nén để tránh mã hóa lại.
- Có chính sách dung lượng và thời gian lưu.
- Có thể để NVR đảm nhiệm ghi hình, AI Box chỉ nhận diện.
Không nên mặc định lưu toàn bộ video chỉ vì Box đang đọc camera.
6. Docker nên đơn giản ở giai đoạn đầu
AI Box — Docker Compose
├── ai-worker       # Camera + nhận diện + chốt sự kiện
├── api-dispatcher  # Gửi sự kiện và ảnh
└── management-api  # Cấu hình, health, trạng thái

Volumes:
├── models/         # Read-only
├── config/
└── runtime/        # SQLite, ảnh, hàng đợi
UI test có thể chạy thêm khi cần. Không cần container riêng cho từng model.