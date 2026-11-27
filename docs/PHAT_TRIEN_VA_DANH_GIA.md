# Phát triển và đánh giá AI

Các lệnh PowerShell dưới đây chạy từ thư mục dự án sau khi [cài môi trường](VAN_HANH.md). Đổi các đường dẫn ví dụ theo dữ liệu thật; không dùng dữ liệu cá nhân chưa được phép. Dùng `& $python -m vehicle_ai <lệnh> --help` để xem đầy đủ tham số.

```powershell
$python = '.\.venv312\Scripts\python.exe'
```

## 1. Chuẩn bị dữ liệu và huấn luyện

1. Dùng `extract` để trích frame và tạo manifest chưa gán nhãn:
   ```powershell
   & $python -m vehicle_ai extract --video data/raw/cam01.mp4 `
     --group-id cam01_day01 --output data/annotation/cam01 --every-seconds 1
   ```
2. Gán nhãn bằng công cụ như CVAT/Label Studio và hoàn thiện CSV theo [mẫu manifest](../data/templates/manifest.csv). Với detector, gán box và lớp; với OCR, dùng crop biển và chuỗi ký tự đúng. Dùng `split --manifest ... --output ...` chia theo `group_id` (camera/clip/đợt quay), không chia các frame cùng nhóm vào các tập khác nhau. Giữ tập test riêng, không dùng để chọn ngưỡng hay model.
3. Chạy `validate-data --manifest ... --task detect --classes ...` hoặc `--task ocr` trước khi train. `build-yolo` chuyển manifest đã gán nhãn sang dataset YOLO nếu cần dùng riêng.

Ví dụ train detector xe từ manifest đã chia:

```powershell
& $python -m vehicle_ai train-detector --manifest data/vehicle/split.csv `
  --classes configs/classes_vehicle.yaml --weights models/yolo11n.pt `
  --device cpu --output runtime/training/vehicle-001
```

Đổi manifest, lớp và trọng số khởi tạo để huấn luyện detector biển/logo. `train-ocr` huấn luyện CRNN riêng; đây không phải OCR mặc định. Dùng `export --engine detector` hoặc `--engine crnn` với `--weights` để xuất ONNX. Tạo bản cấu hình riêng từ `configs/candidate.example.yaml` để thử trọng số mới mà không thay baseline. Dùng `evaluate-detector` đo mAP trên dataset test và `evaluate-ocr` đo đọc ký tự; cả hai không thay cho đánh giá đầu cuối.

## 2. Gán nhãn chuẩn và đánh giá video

Mỗi dòng JSONL là một lượt xe vào/ra vùng đánh giá, kể cả xe khó hoặc không đọc được biển. Mẫu: [ground_truth.example.jsonl](../data/templates/ground_truth.example.jsonl).

```jsonl
{"event_id":"gt-001","source_id":"video-001","camera_id":"CAM-01","started_at_seconds":1.0,"ended_at_seconds":4.0,"plate":"30A12345","vehicle_type":"Ô tô con","brand":"toyota","tags":["day","car"]}
```

- `event_id` duy nhất; `source_id` và `camera_id` khớp với lệnh `run`. Mốc thời gian tính từ đầu video; dùng một quy tắc vào/ra ROI nhất quán. Xe đi khỏi vùng rồi quay lại là lượt mới.
- Biển viết hoa, bỏ dấu chấm, gạch và khoảng trắng; biển hai dòng đọc trên trước. Không đoán: dùng `null` nếu không đọc được. Lượt đó vẫn tính vào recall nhưng không tính độ đúng biển/CER. Loại xe dùng đúng tên lớp; ghi hãng thật ngay cả khi ngoài danh sách model. Dùng `tags` để tách ngày/đêm, xe máy hoặc trường hợp khó.

Chạy baseline và candidate trên cùng video, ROI, camera và thời lượng (lệnh mẫu trong [hướng dẫn vận hành](VAN_HANH.md#3-cli-và-docker)). Không dùng `--realtime` khi so ngưỡng vì việc bỏ frame phụ thuộc tải máy; chỉ bật ở lượt kiểm tra riêng để mô phỏng web. Mỗi lần chạy dùng một thư mục output mới:

- `events.jsonl`: track thô, phù hợp để tìm lỗi tách/trùng track.
- `events.merged.jsonl`: kết quả sau gộp, gần với kết quả web hơn; vẫn cần lưu ý lớp lọc hiển thị của web.
- `run.json` và `evidence/`: metadata của lần chạy (cấu hình, model, nguồn) và ảnh bằng chứng.

```powershell
& $python -m vehicle_ai evaluate-events --truth data/video-001.truth.jsonl `
  --baseline runtime/runs/baseline/events.merged.jsonl `
  --predictions runtime/runs/candidate/events.merged.jsonl `
  --output runtime/evaluation/compare-merged.json
```

Lặp lại với hai file thô để đo ảnh hưởng của tracking; không so raw của một lần với merged của lần khác. Bộ đánh giá ghép một-một theo nguồn, camera và IoU thời gian, không chọn cặp bằng chuỗi biển dự đoán. Đọc recall, số lượt sót/thừa/trùng, độ đúng biển và CER (tỷ lệ lỗi ký tự) cùng nhau. Nhóm `confirmed`, `low`, `unknown` và `tag_groups` giúp tìm chỗ sai; tăng tỷ lệ xác nhận không đồng nghĩa tăng độ chính xác. Báo cả số tuyệt đối, ví dụ `41/52`, và chỉ đổi một ngưỡng mỗi lần.

Khi điều tra xe bị sót, xem `rejection_reasons`/`rejection_detail` trong JSONL: kiểm tra ROI và kích thước xe trước, rồi detector biển, chất lượng ảnh, ngân sách OCR, cuối cùng là điều kiện xác nhận và bộ lọc hiển thị. Xe không thành track phải đối chiếu trực tiếp với video và nhãn chuẩn.

## 3. Đo hiệu năng và kiểm thử

- `scripts/measure_images.py`: đo thông lượng, độ trễ trung vị/p95 và độ chính xác theo các mức `--counts 50 500 5000 10000`. Cần thư mục `--images`, file `--truth` JSONL có `filename` tương đối và nhãn, `--config` và `--output`.
- `scripts/measure_video_matrix.py plan`: tạo manifest và lệnh chạy cho video đặt tên như `10min_144p.mp4`, kèm nhãn `10min_144p.truth.jsonl`. Chạy `commands.ps1` sinh ra, rồi dùng subcommand `report --manifest ... --output ...`. Kiểm tra độ phân giải thực tế và thử clip ngắn trước khi chạy cả ma trận.
- `scripts/benchmark_speed.py`: đo tốc độ pipeline; ghi rõ phần cứng, cấu hình và nguồn khi so sánh.

```powershell
& $python -m pip install -e ".[dev]"
& $python -m ruff check .
& $python -m unittest discover -s tests -v
```
