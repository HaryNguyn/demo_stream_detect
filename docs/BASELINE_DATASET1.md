# Baseline dataset1 — 19/11/2026

Nguồn: `data/raw/dataset1.mp4`; cấu hình: `configs/baseline.yaml`.
Kết quả gốc: `runtime/runs/dataset1-baseline-20261119/`.

## Kết quả chạy

| Chỉ số | Giá trị |
|---|---:|
| Thời lượng video | 38,53 giây |
| Frame đã xử lý | 1156/1156 |
| Thời gian chạy CPU, gồm nạp model | 174,48 giây |
| Tốc độ trung bình | 6,63 FPS |
| Sự kiện theo track | 96 |
| Có OCR đạt ngưỡng (`recognized`) | 8 |
| Có mẫu biển nhưng OCR chưa đạt (`unread`) | 6 |
| Chưa có mẫu biển (`no_plate`) | 82 |
| Có dự đoán hãng | 6 |
| Ảnh bằng chứng | 180 |
| Track ngắn hơn 1 giây | 66 |

Dự đoán loại xe: 33 xe máy, 29 ô tô con, 22 xe tải, 12 xe buýt.
Dự đoán hãng: mazda 2 sự kiện, renault 4 sự kiện.
8 sự kiện OCR trả 6 chuỗi khác nhau; `50F05426` và `43D182249` mỗi chuỗi xuất
hiện ở 2 sự kiện.

Đây là đầu ra model, chưa có ground truth xác nhận. 96 sự kiện không có nghĩa
là 96 xe thật. 8/96 là tỷ lệ có OCR đạt ngưỡng, không phải độ chính xác biển số.
Hai sự kiện cùng biển chỉ là dấu hiệu cần kiểm tra, chưa kết luận xe trùng.

## Điểm cần ưu tiên rà soát

1. 66/96 track ngắn hơn 1 giây: kiểm tra track đứt đoạn, xe nhỏ/xa, che khuất
   và phát hiện nhầm. Không tự lọc bỏ track ngắn trước khi biết có làm mất xe thật.
2. 82 sự kiện chưa có crop biển: kiểm tra bbox xe, kích thước biển và detector biển.
3. Kiểm tra các crop có OCR để xác định lỗi phát hiện, ảnh mờ hay giải mã;
   không dùng kết quả model làm nhãn đúng tự động.
4. Rà soát loại xe và hãng trên ảnh; số liệu dự đoán chưa chứng minh các lớp đó đúng.
5. Tốc độ CPU hiện chậm hơn thời gian thực khoảng 4,53 lần. Tối ưu sau khi có
   dữ liệu đánh giá chất lượng cố định.

## Dữ liệu để gán nhãn

39 frame trích mỗi giây ở `data/annotation/dataset1/`, kèm manifest và hướng dẫn.
Giữ cùng `group_id=dataset1`; chưa chia một video thành train/val/test.
Ảnh đầu có nhiều xe đồng thời: đánh giá temporal IoU phải rà soát ghép từng xe.

Bước tiếp theo: gán nhãn xe/biển trên frame, chuỗi OCR trên crop và lượt xe trên
video; bổ sung video độc lập để lập tập validation/test trước khi train thật.
