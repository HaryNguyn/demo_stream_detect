# So sánh detector gốc và vehicle-reviewed-v001

**Trạng thái:** đã xóa weights và cấu hình của candidate v001 theo yêu cầu
người dùng; giữ báo cáo để tham khảo. Lệnh chạy lại ở cuối tài liệu cần có
checkpoint candidate nên không còn chạy được với các file hiện tại.

## Kết luận

**Giữ `yolo11n.pt` gốc làm detector mặc định.** Trên tập test hiện có, model
gốc tốt hơn rõ rệt so với bản fine-tune v001.
Kết luận chỉ áp dụng detector xe/loại xe, không đánh giá model biển, OCR hoặc logo.

## Phương pháp

- Cùng 15 ảnh test, 53 bbox đã duyệt, từ dataset2.
- Giữ nguyên ảnh, tọa độ nhãn; kiểm tra SHA-256 các ảnh sao chép.
- Remap nhãn 0/1/2/3 sang COCO 2/3/5/7 khi đánh giá baseline.
- Hai model cùng imgsz=640, batch=4, CPU, conf=0.001, NMS IoU=0.7, max_det=300.
- Conf=0.001 phục vụ dựng đường precision/recall và AP, không phải đề xuất
  ngưỡng nhận diện vận hành.
- Không thay checkpoint, không train thêm hoặc chọn model theo kết quả test.

## Chỉ số

| Chỉ số | YOLO11n gốc | Fine-tune v001 |
|---|---:|---:|
| mAP@50 | 85,16% | 11,18% |
| mAP@50:95 | 66,96% | 9,52% |

mAP không phải phần trăm xe nhận diện đúng trong vận hành.

| Lớp | Bbox test | AP@50 gốc | AP@50 fine-tune |
|---|---:|---:|---:|
| Ô tô con | 13 | 74,38% | 33,53% |
| Xe máy | 39 | 81,60% | 0% |
| Xe buýt | 0 | Không đo được | Không đo được |
| Xe tải | 1 | 99,50% | 0% |

Không dùng AP xe tải trên chỉ 1 bbox để kết luận độ chính xác cho xe tải.
Precision/recall thư viện báo cáo được lưu trong JSON; đó là giá trị ở điểm
vận hành được chọn từ đường cong, không phải thống kê tại confidence=0.2 của app.
Precision có thể bằng 1 cho lớp không dự đoán đúng và recall=0; phải đọc cùng AP/recall.

## Diễn giải

- Train v001 chỉ có 2 bbox xe máy, so với 481 bbox ô tô: tập train mất cân đối
  và rất ít bối cảnh; đây là nguyên nhân quan trọng cần khắc phục.
- Nhãn test được tạo gợi ý bằng model gốc rồi người dùng duyệt. Cách tạo nhãn
  này có thể thiên về model gốc nếu còn xe bị bỏ sót hoặc nhãn chưa được sửa hết.
- 15 ảnh thuộc một clip, nhiều frame tương quan. Cần thêm video và nhãn độc lập
  trước khi kết luận độ ổn định triển khai.
- Bản gốc tốt hơn trong phép so sánh này, chưa có nghĩa toàn bộ pipeline AI Box
  đã đạt yêu cầu cho biển số, logo và sự kiện.

## File và chạy lại

Chi tiết: `runtime/evaluation/compare-vehicle-v001/comparison.json`.
Đồ thị/ảnh: thư mục `baseline/` và `candidate/` trong cùng thư mục.

```powershell
.\.venv312\Scripts\python.exe scripts/compare_vehicle_models.py `
  --data runtime/training/vehicle-reviewed-v001/dataset/dataset.yaml `
  --baseline models/yolo11n.pt `
  --candidate runtime/training/vehicle-reviewed-v001/fit/weights/best.pt `
  --output runtime/evaluation/compare-vehicle-v002
```

Output phải là thư mục mới. `baseline_dataset/` là bản đổi ID lớp phục vụ
đánh giá, không dùng làm tập huấn luyện.
