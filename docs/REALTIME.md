# Phát video và ngân sách nhận diện

Web bật `analyze_video(..., realtime=True)`. Worker đọc/phát hình theo FPS
của nguồn capture; worker phân tích gọi `RecognitionSession.feed()` riêng.
Hàng đợi `Queue(maxsize=1)` thay frame cũ khi AI bận, không tích lũy backlog.
Mốc phát bắt đầu khi nhận được frame đầu tiên, không tính thời gian kết nối.
Frame gốc được đưa vào hàng đợi trước khi vẽ; overlay vẽ trên bản sao.
Overlay là snapshot riêng của kết quả và trạng thái track, không đọc trực tiếp
dictionary đang bị luồng AI sửa. Khung có thể lệch khi AI chậm hoặc xe chạy nhanh.

## Các ngưỡng

Trong chế độ YAML, sửa `configs/vietnam-ocr.yaml`:

| Cấu hình | Mặc định | Ý nghĩa |
| --- | --- | --- |
| detector_imgsz | 640 | Kích thước đầu vào detector xe |
| detector_half | false | Chỉ bật FP16 với GPU/backend hỗ trợ |
| max_recognitions_per_frame | 2 | Tối đa số crop xe chạy nhận diện trong một frame AI |
| min_vehicle_width | 120 | Bỏ qua crop xe nhỏ hơn số pixel này |
| min_plate_width | 32 | Bỏ qua OCR biển quá nhỏ |
| min_plate_sharpness | 0 | Ngưỡng phương sai Laplacian; 0 là tắt |
| plate_confirm_reads | 3 | Số lần đọc cùng chuỗi tối thiểu |
| plate_confirm_score | 3.0 | Tổng điểm của chuỗi thắng tối thiểu |
| plate_confirm_mean | 0.85 | Điểm trung bình của chuỗi thắng tối thiểu |
| plate_confirm_margin | 1.0 | Tổng phiếu chuỗi dẫn đầu trừ tổng phiếu chuỗi thứ hai |

Chỉ OCR đã qua bộ kiểm tra biển được bỏ phiếu. Chuỗi thắng theo tổng confidence
OCR, dùng số lần đọc và điểm cao nhất để phân xử hòa; phải đạt cả bốn điều
kiện và có ảnh biển đủ chất lượng mới chốt.
Sau khi chốt, track vẫn được cập nhật vị trí/thời gian nhưng không chạy lại
detector biển hay OCR. Logo tiếp tục theo ngân sách cho đến khi hãng đạt tiêu chí
chốt riêng. Mẫu ảnh và phiếu hiện có được giữ để xuất kết quả.

Ưu tiên xe lâu chưa được lấy mẫu, sau đó crop lớn hơn; điều này tránh xe lớn
chiếm hết ngân sách. Mọi xe trong ROI vẫn cập nhật tracker kể cả khi không OCR.
Detector biển chạy trên crop xe. Bộ lọc kích thước/độ nét cần chỉnh theo camera,
đặc biệt với xe máy hoặc biển ở xa.

Video phát độc lập với tốc độ AI; FPS thực tế vẫn phụ thuộc khả năng giải mã,
vẽ/JPEG, mạng và trình duyệt. Nguồn qua FFmpeg hiện được chuyển về 30 FPS;
file đọc bằng OpenCV dùng FPS gốc. Chưa có nội suy chuyển động box giữa các kết
quả AI, chưa có lịch detect cách 2–3 frame. ByteTrack vẫn chạy mỗi frame AI
được lấy ra khỏi hàng đợi. Chưa tự export model hay tự chọn backend tăng tốc;
chỉ đổi model/backend sau khi đo trên phần cứng thực tế.

## ROI và OCR batch

`crop_vehicle_roi: true` cắt hộp bao đa giác trước detector xe, nới
`vehicle_roi_padding: 0.05` mỗi chiều. Box được cộng offset về ảnh gốc;
vẫn lọc tâm xe theo đa giác. Tắt tùy chọn này để đo đối chứng trên toàn ảnh.
Không có ROI thì dùng toàn frame. Với đầu vào model cố định, cắt ROI không
đảm bảo thời gian inference giảm tỷ lệ thuận với số pixel.

Detector biển dùng `plate_imgsz: 320`. `plate_lower_fraction: 1.0` giữ toàn xe;
có thể thử `0.6` để tìm ở 60% phía dưới. Khi đã có box biển, ưu tiên vùng quanh
box cũ, nới `plate_reuse_padding: 0.3` mỗi phía. Không thấy thì tìm vùng dưới,
rồi toàn xe. Mỗi `plate_full_scan_interval: 5` lần lấy mẫu luôn quét toàn xe để
tránh mắc kẹt ở box cũ. Có thể đặt interval=1 và imgsz=640 để đo đối chứng.

`ocr_top_k: 4` giữ điểm chất lượng của 4 crop đã đưa vào OCR tốt nhất của track:
chấp nhận 4 mẫu đầu qua bộ lọc, sau đó chỉ OCR nếu chất lượng tốt hơn mẫu kém
nhất đang giữ. Đây là chọn mẫu trực tuyến, không biết trước các frame tương lai.
Ảnh crop trong cùng frame AI được gom lại, tối đa `ocr_batch_size: 8`; không
chờ frame tiếp theo để đủ batch. OCR chạy trong worker AI, không chặn phát hình.
Recognizer khác không hỗ trợ batch tự quay về API từng ảnh.

CCT bật ORT_ENABLE_ALL và `ocr_threads: 4`. Batch cố định được chia/pad cho
đúng kích thước, đầu ra trả về đúng track và thứ tự crop. Hai model ONNX có
sẵn đều nhận NHWC uint8 với batch động. Giữ nguyên chuẩn hóa/giải mã CCT.
Không đổi sang TensorRT/OpenVINO tự động.

Chẩn đoán trên web hiển thị ms detect xe, biển/crop/logo, OCR, kích thước batch,
số track đang theo dõi và đã chốt. Mẫu đo CPU cục bộ (4 ảnh tổng hợp 64x128,
warmup 3 lượt, trung vị 10 lượt; không đánh giá độ chính xác):

| ORT threads | 4 lần OCR riêng | Batch 4 |
| --- | --- | --- |
| 2 | 102.33 ms | 100.12 ms |
| 4 | 71.61 ms | 65.58 ms |
| 8 | 71.19 ms | 58.04 ms |

Giữ mặc định 4 luồng để còn tài nguyên cho detector và phát video. Cần đo lại
toàn pipeline trên máy/camera mục tiêu. Để đo số xe bị sót, dùng cùng đoạn
video, cùng ROI đánh giá và danh sách xe/biển được gán nhãn; đối chiếu lần
chạy đầy đủ với từng tối ưu bật riêng. Số track/biển phát hiện không tự nó
là recall nếu chưa có nhãn chuẩn.

## Chọn ảnh tin cậy và chống trùng

Ảnh dùng cho kết quả xác nhận phải đồng thời đạt:

- `min_image_quality: 0.5`: điểm tổng hợp chất lượng ảnh tối thiểu.
- `min_evidence_detection: 0.6`: độ tin cậy detector tối thiểu.
- `min_evidence_ocr: 0.85`: độ tin cậy OCR của chính ảnh biển được lưu.

Điểm ảnh biển là tích của confidence detector, kích thước pixel chuẩn hóa,
độ nét, điểm không cắt mép và điểm tỷ lệ W/H. Sau OCR nhân thêm confidence
trung bình ký tự. Các hệ số đều trong [0,1]; đây là điểm xếp hạng, không phải
xác suất biển đọc đúng.

- `quality_target_width: 100`, `quality_target_height: 32`: đủ kích thước
  khi đạt cả hai chiều; diện tích pixel cũng được lưu trong metadata.
- `quality_target_sharpness: 150`: phương sai Laplacian tham chiếu.
- `quality_edge_margin: 2`: khoảng cách tối thiểu từ biển đến mép crop xe
  (crop xe đã cắt theo biên frame). Nếu biển chạm biên ROI tìm kiếm hẹp,
  tìm lại trên vùng rộng hơn.
- `quality_expected_ratios: [1.4, 4.5]`: dạng biển hai dòng/một dòng.
  Tỷ lệ W/H chỉ là chỉ báo thô của biến dạng, không phải đo góc phối cảnh.

`evidence_top_k: 5` giữ tối đa 3–5 ảnh mỗi loại/track, theo cấu hình. Nếu chưa
đủ 3 ảnh thì giữ số ảnh có được, không nhân bản. OCR theo ngân sách hiện có,
chọn mẫu tốt trực tuyến; không OCR lại cùng ảnh ở lúc finalize. Ảnh lưu có
chuỗi OCR khớp kết quả thắng khi track đã xác nhận. Thẻ kết quả có gallery
ảnh ứng viên; rê chuột lên ảnh để xem kích thước, độ nét, cách mép và W/H.

`keep_low_confidence: true` giữ track chưa chốt với nhãn **độ tin cậy thấp**:
lấy các ảnh tốt nhất sẵn có kể cả dưới ngưỡng. Biển chưa đọc được để trống,
không biến chuỗi OCR bị từ chối thành biển xác nhận. Xe nhỏ chưa được OCR
vẫn giữ một ảnh xe dự phòng. Có thể tắt fallback trên giao diện/YAML.

Logo bỏ phiếu theo tổng confidence × chất lượng ảnh xe, thay cho lấy một
lần có confidence cao nhất. Điểm hãng hiển thị là confidence trung bình
của các phiếu hãng thắng. Loại xe cộng phiếu confidence có trọng số ưu tiên
box lớn; xe quá nhỏ chỉ có dự đoán ban đầu cho fallback.

Trong một lần chạy, khóa chống trùng là nguồn camera + biển chuẩn hóa.
Tracker đổi ID không tạo thêm bản ghi khi đọc cùng biển. Gộp các ảnh tốt
nhất và loại ảnh JPEG giống hệt; kết quả xác nhận được ưu tiên hơn fallback.
Ảnh bị loại khỏi top K được dọn trong thư mục event. Track chưa có biển
được giữ riêng theo ID, không đoán gộp hai xe chưa đọc được biển.

Lịch sử/CSV hiển thị một kết quả tốt nhất cho mỗi nguồn + biển; dữ liệu lịch
sử cũ trong DB không bị xóa. Phiên chạy mới vẫn có thể ghi lần xuất hiện mới.
Điểm ảnh, tổng phiếu và trạng thái tin cậy được giữ trong JSON metadata của
ảnh trong DB, không cần thêm cột. CSV có `confidence_status` để phân biệt.

Hướng dẫn đo riêng nhóm đã chốt/nhãn vàng, so sánh baseline và tiêu chí riêng
cho loại xe/hãng: xem [EVALUATE_CONFIDENCE.md](EVALUATE_CONFIDENCE.md).
