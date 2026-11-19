# Lần train vehicle-reviewed-v001

**Đã loại bỏ bản model này theo yêu cầu người dùng:** `best.pt`, `last.pt`,
`best.onnx` và `configs/vehicle-reviewed-v001.yaml` đã được xóa.
Ứng dụng dùng model gốc `yolo11n.pt` qua `configs/baseline.yaml`.
Các thông tin/lệnh bên dưới là lịch sử lần train, không còn dùng để chạy model v001.

## Dữ liệu đã xác nhận trước khi chạy

- 102 ảnh được người dùng duyệt, 3 ảnh bỏ qua.
- 830 bbox: 580 ô tô con, 124 xe máy, 95 xe buýt, 31 xe tải.
- Split theo video, seed 42: 48 train, 39 validation, 15 test.
- Manifest: `data/review/vehicle-v001/split-reviewed-v001.csv`.
- Kiểm tra nhãn, nhóm và trùng pixel giữa split đã đạt.

## Cấu hình lần chạy

- Fine-tune detector YOLO11n từ `yolo11n.pt`.
- 30 epoch, batch 4, imgsz 640, CPU.
- Chọn `best.pt` bằng validation; test chỉ dùng đánh giá sau khi train.
- Các lớp đầu ra đổi thành 0=ô tô con, 1=xe máy, 2=xe buýt, 3=xe tải.
- Sau train: đánh giá test và export ONNX.
- Không thay weights mặc định. OCR/biển/logo tiếp tục dùng model baseline.

## Đường dẫn kết quả

- Training/history: `runtime/training/vehicle-reviewed-v001/fit/`.
- Checkpoint: `runtime/training/vehicle-reviewed-v001/fit/weights/best.pt`.
- ONNX: `runtime/training/vehicle-reviewed-v001/fit/weights/best.onnx`.
- Test metrics: `runtime/evaluation/vehicle-reviewed-v001/metrics.json`.
- Pipeline config: `configs/vehicle-reviewed-v001.yaml`.

## Kết quả hoàn tất

Đã chạy đủ 30 epoch (~6,9 phút train CPU), đánh giá test và xuất `best.onnx`.
Theo lịch sử `results.csv`, validation mAP50-95 cao nhất tại epoch 2.

| Chỉ số test | Giá trị |
|---|---:|
| Số ảnh / bbox test | 15 / 53 |
| mAP@50 | 0,111766 (~11,18%) |
| mAP@50:95 | 0,095154 (~9,52%) |
| Recall trung bình theo lớp được đánh giá | 0,205128 (~20,51%) |
| Precision trung bình do Ultralytics báo cáo | 0,698915 |

Precision tổng hợp không có nghĩa là 69,89% xe được nhận diện đúng:
báo cáo thư viện có precision=1, recall=0 cho lớp xe máy/xe tải không có
phát hiện đúng, nên cần đọc cùng recall và AP từng lớp.
AP@50 test: ô tô con ~0,335; xe máy=0; xe tải=0.
Test không có xe buýt nên không đo được khả năng lớp này trên test.

## Phân bố nhãn tìm thấy sau train

| Lớp | Train | Validation | Test |
|---|---:|---:|---:|
| Ô tô con | 481 | 86 | 13 |
| Xe máy | 2 | 83 | 39 |
| Xe buýt | 93 | 2 | 0 |
| Xe tải | 21 | 9 | 1 |

Split theo video tránh rò rỉ frame nhưng đang mất cân đối mạnh:
train chỉ có 2 bbox xe máy, trong khi test chủ yếu là xe máy.
Đây là một nguyên nhân rõ ràng cần xử lý; chưa loại trừ ảnh hưởng của nhãn,
bối cảnh, độ phân giải và cấu hình fine-tune.

Model hiện là bản thử nghiệm, chưa nên thay model mặc định.
Đã chạy so sánh trên cùng tập và remap đúng lớp: baseline mAP@50=85,16%,
candidate=11,18%. Xem [SO_SANH_DETECTOR_V001.md](SO_SANH_DETECTOR_V001.md)
để biết phương pháp và giới hạn của phép so sánh.

## Hướng tiếp theo

1. Bổ sung video độc lập có nhiều xe máy/xe tải cho train; kiểm tra lại nhãn.
2. Tăng độ đa dạng camera/bối cảnh và số nhóm, giữ test độc lập.
3. Kiểm tra phân bố lớp ở từng split trước lần train tiếp theo.
4. So sánh baseline và candidate trên cùng nhãn, cùng class mapping.

Không chia ngẫu nhiên frame cùng video sang train/test để làm tăng chỉ số.
Với mỗi split chỉ có một video, kết quả chưa đại diện cho camera triển khai.

## Thử sau khi hoàn tất

Trong giao diện, chọn **Dùng pipeline YAML** và nhập
`configs/vehicle-reviewed-v001.yaml`. ROI dùng tùy chọn JSON riêng.

Hoặc chạy CLI:

```powershell
.\.venv312\Scripts\python.exe -m vehicle_ai run `
  --config configs/vehicle-reviewed-v001.yaml `
  --source data/raw/dataset1.mp4 --source-id dataset1 --camera CAM-01 `
  --output runtime/runs/dataset1-vehicle-reviewed-v001
```

Đây là thử trực quan trên video đã dùng trong tập validation, không phải
báo cáo nghiệm thu độc lập. Ground truth sự kiện/OCR vẫn cần gán nhãn riêng.
