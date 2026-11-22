# Triển khai theo từng hạng mục

## Phạm vi

Giai đoạn hiện tại: video/camera thử nghiệm → theo dõi xe → loại xe, logo,
biển số → sự kiện và ảnh. Chưa triển khai trigger/API gửi backend.
Code đã có `RecognitionSession.feed(frame, timestamp)` và `finalize()` để
sau này thay cơ chế điều khiển phiên mà không viết lại nhận diện.

Các model hiện có vẫn là baseline của nguồn tham khảo. Công cụ train mới
không đồng nghĩa đã có model được huấn luyện/kiểm chứng trên dữ liệu riêng.

## 0. Cài đặt

Mở PowerShell tại thư mục có `app.py`:

```powershell
$python = ".\.venv312\Scripts\python.exe"
& $python -m pip install -e .
& $python -m vehicle_ai --help
```

Máy mới: tạo `.venv312` bằng Python 3.12 trước. App cũ chạy như trước sau
khi cài package editable. `.env` và database hiện có không được thay đổi.

## 1. Chuẩn dữ liệu

Đặt video vào `data/raw/`, ví dụ `data/raw/cam01.mp4`.

```powershell
& $python -m vehicle_ai extract `
  --video data/raw/cam01.mp4 --group-id cam01_day01 `
  --output data/annotation/cam01 --every-seconds 1 --max-frames 300
```

Lệnh tạo ảnh và manifest chưa có nhãn. Gán nhãn bằng CVAT/Label Studio hoặc
công cụ bạn dùng, rồi hoàn thiện manifest theo [data/README.md](../data/README.md).
Trích ảnh từ video chưa tạo được nhãn huấn luyện.

Gộp dữ liệu đã gán nhãn của ít nhất 3 nhóm độc lập, rồi chia:

```powershell
& $python -m vehicle_ai split `
  --manifest data/vehicle/manifest.csv --output data/vehicle/split.csv
& $python -m vehicle_ai validate-data `
  --manifest data/vehicle/split.csv --task detect --classes configs/classes_vehicle.yaml
```

Giữ nguyên split giữa các lần so sánh; không chọn model bằng test.

## 2. Baseline và cấu trúc module

```powershell
& $python -m vehicle_ai run `
  --config configs/baseline.yaml --source data/raw/cam01.mp4 `
  --source-id video-001 --camera CAM-01 --output runtime/runs/baseline-001
```

Output:

```text
runtime/runs/baseline-001/
├── events.jsonl
├── run.json           # config, model hash, source hash, thời gian xử lý
└── evidence/          # JPEG xe/biển theo event_id
```

Thư mục output phải mới để không ghi đè kết quả cũ. Run CLI không ghi vào
MariaDB đang dùng bởi giao diện. `run.json` ghi trạng thái failed nếu
inference lỗi; file JSONL có thể chứa phần kết quả đã hoàn tất.

Package:

| Module | Vai trò |
|---|---|
| `capture.py` | Đọc nguồn OpenCV/FFmpeg/yt-dlp |
| `inference.py` | Detector, crop và nhận diện trên ảnh xe |
| `ocr.py` | Adapter CCT-VN |
| `crnn.py` | Kiến trúc, tiền xử lý và decoder CRNN/CTC riêng |
| `aggregation.py` | Giới hạn ảnh, chọn mẫu, bỏ phiếu biển |
| `session.py` | Trạng thái track, timeout, feed/finalize |
| `pipeline.py` | Chạy nguồn video, callback |
| `domain.py` | Schema sự kiện tiếng Anh |
| `storage.py` | JSONL và ảnh của lần thực nghiệm |
| `datasets.py` | Trích ảnh, chia nhóm, kiểm tra nhãn, tạo YOLO dataset |
| `training.py` | Train/export detector và CRNN |
| `evaluation.py` | Ghép sự kiện và tính chỉ số |

`app.py` là entry point mỏng của web Flask.
Database nằm trong `vehicle_ai.persistence`; các file compatibility ở root
đã được loại bỏ. Schema MariaDB vẫn tương thích dữ liệu cũ.
Xem [ARCHITECTURE.md](ARCHITECTURE.md) cho cấu trúc hiện tại.

## 3. Train và export

### Detector xe, biển hoặc logo

```powershell
& $python -m vehicle_ai train-detector `
  --manifest data/vehicle/split.csv --classes configs/classes_vehicle.yaml `
  --weights models/yolo11n.pt --epochs 50 --batch 8 --imgsz 640 --device cpu `
  --output runtime/training/vehicle-001
```

Với biển/logo: đổi manifest, classes YAML và weights khởi tạo phù hợp.
Checkpoint tốt nhất: `runtime/training/vehicle-001/fit/weights/best.pt`.
Để dùng NVIDIA GPU đã cài PyTorch CUDA, truyền `--device 0`.
Không tự khởi chạy 50 epoch trên dữ liệu chưa chuẩn bị.

Dataset YOLO được tạo dưới `output/dataset`; nhãn tên lớp gốc không bị sửa.

```powershell
& $python -m vehicle_ai evaluate-detector `
  --weights runtime/training/vehicle-001/fit/weights/best.pt `
  --data runtime/training/vehicle-001/dataset/dataset.yaml `
  --output runtime/evaluation/vehicle-001
```

Lệnh này dùng split test. mAP detector không thay thế đánh giá toàn bộ sự kiện.

### Train logo cho thị trường Việt Nam

Taxonomy mẫu trong `configs/classes_logo.example.yaml` gồm 16 class:

- Bắt buộc: `vinfast`, `toyota`, `hyundai`, `kia`, `mazda`, `mitsubishi`,
  `ford`, `honda`, `suzuki`.
- Bổ sung: `mercedes`, `bmw`, `lexus`, `isuzu`, `peugeot`, `volvo`, `other`.
- Mặc định gộp `opel`, `seat`, `skoda`, `dacia`, `citroen`, `fiat`,
  `chevrolet`, `audi`, `renault`, `volkswagen` vào `other`. Có thể tách riêng
  Chevrolet/Audi/Volkswagen nếu camera mục tiêu có đủ dữ liệu.

**Đây là danh sách cho lần train mới.** Không đổi `model.names` của checkpoint
cũ để thêm VinFast: model cần được học từ ảnh logo VinFast đã gán nhãn.
ID nhãn YOLO là vị trí trong danh sách, bắt đầu từ 0; phải ánh xạ lại toàn bộ
nhãn cũ theo tên hãng trước khi dùng taxonomy này. Giữ bản dataset gốc.

Chuẩn bị dữ liệu:

Remap nhãn cũ sang taxonomy mới bằng script sau; kiểm tra thứ tự class nguồn
bằng `--old-classes` nếu khác danh sách mặc định 24 hãng:

```powershell
& $python scripts/remap_logo_labels.py `
  --input data/logo_old/labels --output data/logo_v2/labels `
  --new-classes configs/classes_logo.example.yaml `
  --report runtime/evaluation/logo_remap_report --dry-run
```

Bỏ `--dry-run` để ghi nhãn và báo cáo JSON/CSV. Thư mục output phải chưa tồn tại.
Dry-run không ghi bất kỳ file nào. Script chỉ remap nhãn detection (5 trường),
giữ cây thư mục và nhãn rỗng; không sao chép ảnh hoặc tạo manifest.
Nissan mặc định vào `other` kèm thông báo; dùng `--map nissan=<class>` để đổi.

1. Gán bounding box quanh logo trong ảnh xe, phù hợp đầu vào crop xe của pipeline.
   `other` chỉ dành cho logo hãng ngoài danh sách; logo mờ/không rõ hãng cần
   duyệt lại, không tự gán thành `other`. Ảnh không có logo là ảnh âm với nhãn rỗng.
2. Loại ảnh trùng/gần trùng; chia train/val/test theo xe, video hoặc phiên camera
   trước khi cân bằng để tránh các frame cùng xe lọt sang nhiều split.
3. Thống kê cả số ảnh và số logo mỗi hãng trong split train. Đặt giới hạn mẫu
   mỗi hãng theo dữ liệu thực có, lấy mẫu đa dạng ngày/đêm, góc nhìn, khoảng cách
   và camera. Bổ sung ảnh thật cho hãng ít mẫu; augmentation chỉ áp dụng train.
   Không suy ra tỷ lệ dataset từ thị phần doanh số bán mới.
4. Giữ test đại diện giao thông thực tế; có thêm tập đánh giá cân bằng theo hãng.
   Theo dõi precision/recall/AP từng hãng và confusion matrix, nhất là VinFast
   bị nhầm sang hãng khác hoặc các hãng khác bị đoán thành VinFast.
   `other` cũng cần nhiều kiểu logo, không chỉ một hãng.

```powershell
& $python -m vehicle_ai train-detector `
  --manifest data/logo/split.csv --classes configs/classes_logo.example.yaml `
  --weights models/yolo11n.pt --epochs 50 --batch 8 --imgsz 640 --device cpu `
  --output runtime/training/logo-vn-001
```

Sau khi đánh giá, chọn `runtime/training/logo-vn-001/fit/weights/best.pt`
trong mục model logo trên web hoặc cấu hình `logo_weights` của pipeline.
Danh sách class nhỏ hơn không làm giảm đáng kể dung lượng toàn bộ model;
lợi ích chính là taxonomy phù hợp và chất lượng dữ liệu.

### OCR riêng

CCT hiện có là model ONNX inference, không có training graph/checkpoint
đủ để fine-tune trực tiếp. Đã cung cấp CRNN/CTC mới để train từ crop đã gán nhãn:

```powershell
& $python -m vehicle_ai validate-data --manifest data/ocr/split.csv --task ocr
& $python -m vehicle_ai train-ocr `
  --manifest data/ocr/split.csv --epochs 50 --batch 32 `
  --device cpu --output runtime/training/ocr-001
```

CRNN lưu `best.pt`, `last.pt`, `history.jsonl`, `training.json`. Chọn checkpoint
bằng exact-match trên val, sau đó CER; test không tham gia chọn checkpoint.
Tiền xử lý train/infer dùng chung: grayscale, letterbox 32×160, chuẩn hóa [-1,1].
Decoder CTC gộp ký tự lặp liên tiếp, nhưng giữ ký tự lặp có blank ở giữa.
Decoder CCT vẫn giữ ký tự theo slot như trước.

### Export ONNX

```powershell
& $python -m pip install -r requirements.txt
& $python -m vehicle_ai export --engine detector `
  --weights runtime/training/vehicle-001/fit/weights/best.pt
& $python -m vehicle_ai export --engine crnn `
  --weights runtime/training/ocr-001/best.pt --output models/ocr-own-v1.onnx
```

CRNN ONNX đi kèm `.onnx.json`, cần giữ cả hai. Bản export CRNN hiện dùng
batch=1 cố định, chạy ONNX CPU. Model PyTorch CRNN có thể chạy CPU/CUDA.

## 4. Đánh giá model mới và sự kiện

Copy `configs/candidate.example.yaml` thành `configs/candidate.yaml` và chỉnh
đường dẫn. Với CRNN: `ocr_engine: crnn`, `ocr_weights` trỏ tới best.pt hoặc ONNX,
`ocr_config: ""`.

**ID lớp xe phải khớp checkpoint**: model COCO dùng 2/3/5/7; dataset xe mẫu
dùng 0/1/2/3. Nếu đổi taxonomy phải đổi `vehicle_classes` tương ứng.

```powershell
& $python -m vehicle_ai evaluate-ocr `
  --manifest data/ocr/split.csv --config configs/candidate.yaml `
  --output runtime/evaluation/ocr-candidate.json

& $python -m vehicle_ai run `
  --config configs/candidate.yaml --source data/raw/cam01.mp4 `
  --source-id video-001 --camera CAM-01 --output runtime/runs/candidate-001

& $python -m vehicle_ai evaluate-events `
  --truth data/ground_truth/cam01.jsonl `
  --predictions runtime/runs/baseline-001/events.jsonl `
  --output runtime/evaluation/baseline-001.json
```

Lặp evaluate-events với predictions của candidate, giữ nguyên ground truth.
Chỉ số: precision/recall/F1 sự kiện, missed/extra/possible duplicate,
biển đúng trên matched events, biển đúng end-to-end, CER, loại xe và hãng.
`null` nghĩa là không có mẫu hợp lệ cho chỉ số, không phải accuracy=0.

Ghép một-một bằng temporal IoU cùng source/camera, không dùng biển dự đoán
để quyết định ghép. Phương pháp phù hợp video một làn/xe tách biệt; khi nhiều
xe chồng thời gian, cần rà soát bảng `matches` hoặc bổ sung ground truth
trajectory/ROI. `possible_duplicate_events` chỉ là ứng viên trùng theo thời gian.

Độ tin cậy model hiện là score gốc, chưa hiệu chuẩn xác suất. Trạng thái
`no_plate` nghĩa là pipeline chưa có mẫu biển, có thể do model chưa bật.

## 5. Giao diện và Docker

```powershell
& $python app.py
```

Chọn chế độ cấu hình YAML để dùng candidate và mapping lớp mới.
CLI là đường chạy chuẩn cho baseline/evaluation; UI web phục vụ kiểm tra trực quan.

Docker Desktop cần bật Linux engine:

```powershell
docker compose config --quiet
docker compose build
docker compose run --rm inference python -m vehicle_ai run `
  --config configs/baseline.yaml --source data/raw/cam01.mp4 `
  --source-id video-001 --camera CAM-01 --output runtime/runs/docker-baseline-001
docker compose --profile ui up -d ui
```

Models/config/data mount read-only; runtime là bind mount ghi kết quả. Linux
host cần cấp quyền ghi runtime cho UID 10001. Không đóng `.env`, dữ liệu, model
hay `.venv` vào image. Docker này dành CPU inference, chưa là image train GPU.
UI đọc `.env`; nếu MariaDB chạy trên Windows host thì cấu hình host là
`host.docker.internal`. CLI baseline không cần DB.

## 6. Kiểm tra và giới hạn hiện tại

```powershell
& $python -m unittest discover -s tests -v
```

Các kiểm thử bao gồm vòng đời phiên, ảnh bằng chứng, lưu sự kiện MariaDB,
chống rò rỉ dataset, đánh giá một-một và train CRNN 1 epoch trên dữ liệu tổng hợp.
Test tổng hợp chỉ xác nhận đường chạy, không chứng minh chất lượng nhận diện.

Timestamp video đang dùng frame_index/FPS. Nguồn live qua FFmpeg quy đổi 30 FPS;
chưa phải timestamp cảm biến, chưa có trigger hoặc cơ chế đồng bộ camera.
Một track vẫn tương ứng một sự kiện; chưa ghép lại xe khi tracker đứt đoạn.
Logo vẫn chọn kết quả confidence cao nhất như baseline.

## Công việc cần dữ liệu thật

1. Cung cấp video camera mục tiêu và taxonomy loại xe/hãng mong muốn.
2. Gán nhãn detector/OCR và ground truth lượt xe.
3. Chạy baseline, phân tích lỗi, chọn thành phần cần fine-tune.
4. Train và so sánh trên test cố định trước khi thay model mặc định.

Giữ thông tin nguồn của code/model đã kế thừa; CRNN mới và logic dự án là phần
phát triển thêm, không làm thay đổi xuất xứ các trọng số YOLO/CCT/logo hiện có.
