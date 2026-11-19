# Dữ liệu của dự án

Chưa có dataset đã gán nhãn trong thư mục này. File trong `templates/` chỉ là
định dạng mẫu, không phải dữ liệu nghiệm thu.

## 1. Detector xe / biển / logo

Mỗi tác vụ dùng manifest riêng:

```csv
image,label,text,group_id,split
images/a.jpg,labels/a.txt,,camera01_day01,train
images/b.jpg,labels/b.txt,,camera02_day02,val
images/c.jpg,labels/c.txt,,camera03_day03,test
```

Đường dẫn tính từ vị trí file CSV. Mỗi file nhãn YOLO có các dòng:

```text
class_id x_center y_center width height
```

Tọa độ chuẩn hóa [0,1]. Ảnh âm tính phải có file nhãn rỗng đã được người gán
nhãn xác nhận, không được tự xem ảnh chưa gán nhãn là ảnh âm tính.
Tên lớp và ID phải khớp YAML trong `configs/classes_*.yaml`.

Để train logo dùng bbox logo trên crop xe (phù hợp đầu vào inference hiện tại);
để train detector biển dùng ảnh/crop có biển đủ đa dạng và bbox biển.

## 2. OCR

Manifest OCR trỏ tới **ảnh crop biển**, không phải ảnh toàn cảnh:

```csv
image,label,text,group_id,split
crops/a.jpg,,30A12345,camera01_day01,train
crops/b.jpg,,51B67890,camera02_day02,val
crops/c.jpg,,29C11111,camera03_day03,test
```

Nhãn dùng A–Z và 0–9; bỏ dấu gạch/chấm/khoảng trắng. Với biển hai dòng,
chuỗi đọc theo dòng trên rồi dòng dưới. Phải có dữ liệu hai dòng trong train
và báo cáo riêng; CRNN mới là baseline, chưa có bảo đảm xử lý hai dòng tốt.

## 3. Chống rò rỉ

- `group_id` nên theo camera/ngày hoặc toàn bộ video; các frame cùng lượt xe
  không được nằm ở nhiều split.
- Tối thiểu 3 nhóm độc lập để tạo train/val/test. Đây là yêu cầu cấu trúc,
  không phải cỡ dữ liệu đủ để model tốt.
- Bộ kiểm tra chặn trùng group, trùng pixel giữa split, bbox sai và thiếu nhãn.
- Ảnh gần giống nhưng không giống hệt vẫn cần kiểm soát bằng group và rà soát thủ công.
- Gộp CSV từ nhiều nguồn trước khi split; chuẩn hóa lại đường dẫn tương đối.

## 4. Nhãn sự kiện

Mỗi dòng JSONL mô tả một lượt xe; xem `templates/ground_truth.example.jsonl`.
`source_id` và `camera_id` phải giống tham số chạy baseline.
Thời gian bắt đầu/kết thúc là giây trong video. Đánh dấu khoảng xe thực sự
hiện diện, không cộng thêm timeout theo dõi.

Không đọc được biển trong ground truth thì dùng `plate: null`.
Các lượt đó vẫn được tính trong đánh giá phát hiện sự kiện.
