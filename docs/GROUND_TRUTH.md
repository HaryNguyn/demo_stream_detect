# Gán nhãn ground truth

Mỗi dòng JSONL là một lượt xe đi qua vùng đánh giá. Mẫu:
`data/templates/ground_truth.example.jsonl`.

```jsonl
{"event_id":"gt-001","source_id":"video-001","camera_id":"CAM-01","started_at_seconds":1.0,"ended_at_seconds":4.0,"plate":"30A12345","vehicle_type":"Ô tô con","brand":"toyota","tags":["day","car"],"notes":""}
{"event_id":"gt-002","source_id":"video-001","camera_id":"CAM-01","started_at_seconds":3.2,"ended_at_seconds":5.1,"plate":"59X112345","vehicle_type":"Xe máy","brand":"honda","tags":["day","motorcycle"],"notes":""}
{"event_id":"gt-003","source_id":"video-001","camera_id":"CAM-01","started_at_seconds":8.0,"ended_at_seconds":11.5,"plate":null,"vehicle_type":"Ô tô con","brand":"vinfast","tags":["unknown-brand","hard"],"notes":"Biển bị che, không đọc được"}
```

## Quy tắc từng trường

| Trường | Quy tắc |
| --- | --- |
| `event_id` | Duy nhất, tăng dần (`gt-001`, `gt-002`…). |
| `source_id`, `camera_id` | Khớp từng ký tự với `--source-id` và `--camera` của lệnh `run`. Lệch một ký tự là không ghép được. |
| `started_at_seconds`, `ended_at_seconds` | Xe vào và rời vùng đánh giá, tính từ đầu video. Dùng một định nghĩa duy nhất cho mọi xe, ví dụ “tâm xe chạm và rời đa giác ROI”. |
| `plate` | Chữ hoa, bỏ dấu chấm, gạch ngang, khoảng trắng. `30A-123.45` → `30A12345`; `59-X1 123.45` → `59X112345`. Biển hai dòng đọc dòng trên rồi dòng dưới. Không đọc được để `null` hoặc bỏ trường, **không đoán**. |
| `vehicle_type` | Đúng chuỗi pipeline xuất: `Ô tô con`, `Xe máy`, `Xe buýt`, `Xe tải`. Chốt trước bán tải/van/xe tải nhỏ thuộc nhóm nào. |
| `brand` | Hãng thật, viết thường theo taxonomy của model. Hãng model chưa biết (VinFast…) vẫn ghi `vinfast`. Không chắc thì `null` hoặc bỏ trường. |
| `tags` | Tùy chọn: danh sách chuỗi không rỗng, dùng để tách recall theo nhóm (`day`, `night`, `motorcycle`, `hard`…). |
| `notes` | Tùy chọn: chuỗi ghi chú tự do, không tham gia tính điểm. |

## Bộ đánh giá chấp nhận gì

Đã kiểm tra bằng test:

- `plate: null` **được chấp nhận**. Sự kiện vẫn tính vào recall nhưng bị loại khỏi
  tỷ lệ biển đúng và CER. Bỏ hẳn trường `plate` cũng được.
- `vehicle_type: null`, `brand: null` hoặc bỏ trường đều được; sự kiện không bị
  tính vào mẫu số accuracy của trường đó.
- `brand` so sánh không phân biệt hoa/thường và khoảng trắng: `Toyota` khớp `toyota`.
- `plate` được chuẩn hóa khi chấm: bỏ dấu chấm/gạch ngang/khoảng trắng, viết hoa.
  Nhãn vẫn nên ghi đúng chuẩn để dễ đọc và so sánh.
- `tags` sai kiểu (không phải danh sách chuỗi khác rỗng) hoặc `notes` không phải
  chuỗi sẽ bị từ chối khi đọc file, để tránh nhóm rác.
- `event_id` trùng, thời gian âm/ngược, thiếu `source_id`/`camera_id`/mốc thời gian
  đều bị từ chối.

## Nguyên tắc để nhãn đáng tin

- Gán độc lập với model. Không mở kết quả pipeline rồi sửa theo; xem video, tua
  đến frame biển rõ nhất và tự đọc.
- Gán mọi xe trong vùng đánh giá, kể cả xe nhỏ, ở xa hoặc không đọc được biển.
  Chỉ gán xe dễ thì recall sẽ cao giả. Dùng `tags` để sau này tách nhóm.
- Xe chồng nhau về thời gian vẫn gán đủ, nhưng đánh dấu `tags`/`notes` để rà bảng
  `matches`, vì ghép theo IoU thời gian dễ nhầm ở đây.
- Cùng gốc thời gian với pipeline. Pipeline dùng `frame_index/FPS`; nguồn qua
  FFmpeg quy về 30 FPS thì đo mốc theo cùng FPS đó, tránh lệch dần.
- Một lượt xe một sự kiện: dừng đèn đỏ rồi đi tiếp vẫn là một; rời đi rồi quay lại
  là sự kiện mới. Không gộp hai lượt của cùng một biển.
- Người gán thứ hai cho khoảng 20% số xe để đo đồng thuận. Chỗ hai người khác nhau
  chính là chỗ quy tắc còn mơ hồ.
- Chia clip thành tập chỉnh ngưỡng và tập giữ nguyên để báo cáo cuối; không chỉnh
  ngưỡng trên chính tập chấm điểm.

## Bộ nhãn nên có

Khoảng 50–100 lượt xe, đủ: ngày và đêm, ô tô và xe máy, xe nhanh và xe chậm/dừng,
biển một dòng và hai dòng, xe VinFast, hãng ngoài danh sách, và vài xe cố tình khó
(biển mờ, bị che). Gắn `tags` cho các nhóm này để `tag_groups` trong báo cáo tách
recall và độ đúng biển theo nhóm.

## Lỗi hay gặp

- `brand` viết hoa/thường không nhất quán hoặc `plate` còn dấu cách.
- Đo mốc thời gian bằng công cụ khác FPS với pipeline.
- Gán thiếu xe khó nên số liệu đẹp hơn thực tế.
- Gộp hai lượt của cùng một biển thành một sự kiện.
- Ghi `other` thay cho `vinfast`, làm mất dấu lỗi model chưa có lớp VinFast.

## Quyền riêng tư

Biển số là thông tin cá nhân. Lưu video, ảnh và file ground truth ở nơi có kiểm
soát truy cập; không đưa chúng vào kho mã nguồn công khai.
