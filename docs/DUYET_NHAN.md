# Duyệt nhãn xe từ 3 video

## Dữ liệu đã chuẩn bị

| Nhóm | Video nguồn | Ảnh trích |
|---|---|---:|
| dataset1 | dataset1.mp4 | 39 |
| dataset2 | 1795100286847_1768171110419924287_8411524150686472973.mp4 | 18 |
| dataset3 | videoplayback (1).mp4 | 48 |

Tổng 105 ảnh, lấy mỗi giây. YOLO11n COCO tạo 829 bbox gợi ý với confidence
0.25, imgsz 960. Có 4 ảnh không có bbox gợi ý; không mặc định đó là ảnh âm tính.
Chưa có ảnh nào được người dùng duyệt tại thời điểm tạo bộ nhãn.

Thư mục: `data/review/vehicle-v001/`.

- `review.json`: nhãn gợi ý, trạng thái duyệt, nguồn ảnh, nhóm video.
- `suggested_labels/`: nhãn YOLO tự động, chưa phải ground truth.
- `previews/`: 105 ảnh vẽ khung để xem nhanh.
- `classes.yaml`: taxonomy của bộ nhãn.
- `reviewed_labels/`: chỉ tạo khi duyệt.
- `reviewed_manifest.csv`: chỉ chứa ảnh đã duyệt, tạo/cập nhật khi duyệt.

## Mở giao diện

Trong thư mục chứa `app.py`:

```powershell
.\.venv312\Scripts\python.exe -m streamlit run review_app.py --server.port 8502
```

Mở http://localhost:8502.

### Vẽ ROI vùng nhận diện

Ở đầu ảnh có khung **Vẽ ROI — vùng nhận diện**:

- Hình chữ nhật: giữ và kéo chuột từ một góc đến góc đối diện.
- Đa giác: chọn chế độ, bấm lần lượt các đỉnh bao quanh vùng.
- Bấm **Lưu ROI cho video**; ROI dùng chung cho các frame cùng `group_id`.
- **Bỏ ROI / dùng toàn ảnh** lưu cấu hình nhận diện toàn ảnh.
- Chuyển video sẽ dùng ROI riêng của video đó. Tọa độ chuẩn hóa nên không phụ thuộc độ phân giải hiển thị.

File được lưu trong `data/review/vehicle-v001/rois/`; đường dẫn cụ thể hiện
bên dưới khung vẽ. ROI không tự xóa hoặc duyệt các nhãn xe.

Để chạy baseline với vùng đã vẽ, thêm `--roi-file "<đường dẫn JSON>"` vào lệnh
`python -m vehicle_ai run` và dùng output mới. Trong giao diện nhận diện cổng
8501, nhập đường dẫn vào **ROI JSON (tùy chọn)** trước khi phân tích video.
Pipeline chỉ giữ xe có **tâm bbox** trong ROI; detector vẫn chạy toàn frame.
Vẽ ROI không chạy lại baseline cũ và không tự áp dụng ROI của video này cho video khác.

### Duyệt khung nhãn xe

1. Chọn video/ảnh; so sánh ảnh gốc và ảnh có khung.
2. Kiểm tra mọi xe, không chỉ các khung model đã gợi ý.
3. Sửa ID lớp: 0=Ô tô con, 1=Xe máy, 2=Xe buýt, 3=Xe tải.
4. Sửa bbox trong bảng; tọa độ tâm/kích thước chuẩn hóa 0–1.
   Ví dụ ảnh rộng 1920, bbox x1=192,x2=576 thì x_center=0.2, width=0.2.
5. Xóa khung sai hoặc thêm hàng cho xe bị bỏ sót.
6. Bấm **Lưu và duyệt ảnh**. Nếu bảng rỗng, cần xác nhận không có xe thật.
7. Ảnh không thể gán nhãn đáng tin cậy thì bấm **Bỏ qua ảnh này**.

Có thể chọn trạng thái `approved` để sửa lại ảnh đã duyệt. Chọn `skipped` để
duyệt lại ảnh từng bỏ qua. Dùng một người duyệt tại một thời điểm.
UI hiện chỉnh bảng tọa độ, chưa hỗ trợ kéo/vẽ khung bằng chuột.

## Sau khi duyệt

Chưa train tự động. Sau khi có đủ ảnh đã duyệt ở 3 nhóm, chạy:

```powershell
.\.venv312\Scripts\python.exe -m vehicle_ai split `
  --manifest data/review/vehicle-v001/reviewed_manifest.csv `
  --output data/review/vehicle-v001/split-reviewed.csv
.\.venv312\Scripts\python.exe -m vehicle_ai validate-data `
  --manifest data/review/vehicle-v001/split-reviewed.csv `
  --task detect --classes data/review/vehicle-v001/classes.yaml
```

Chia theo cả video để tránh cùng lượt xe vào nhiều tập. Với chỉ 3 video,
mỗi split chỉ có 1 nhóm nên kết quả còn phụ thuộc mạnh vào từng clip.
Nếu các video là đoạn cắt của cùng nguồn/lượt xe, cần gộp lại group trước khi split.
Chưa có nhãn OCR/logo; không dùng nhãn xe này để train OCR hoặc logo.
