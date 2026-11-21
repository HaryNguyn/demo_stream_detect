# So sánh đã chốt / nhãn vàng

## Những gì báo cáo đo

`evaluate-events` ghép dự đoán và ground truth một-một theo source, camera và
IoU thời gian, không dùng biển dự đoán để chọn cặp. Chỉ ghép một lần trên toàn
bộ dữ liệu rồi chia nhóm `confirmed`, `low`, `unknown`; không ghép lại từng
nhóm vì như vậy một xe thật có thể bị tính đúng hai lần.

Mỗi nhóm có số dự đoán, tỷ lệ, số matched, extra, possible duplicate,
số biển đúng, tỷ lệ biển đúng trên matched có biển chuẩn và CER. Nhóm không có
biển chuẩn để đối chiếu trả `null` thay vì accuracy=0. `confirmation_rate`
tính confirmed/(confirmed+low); không bao gồm dữ liệu cũ thiếu nhãn `unknown`.
CER nhóm không tính các xe bị sót; luôn đọc kèm missed_events và recall tổng.
`tag_groups` lặp lại recall và độ đúng biển theo từng `tags` của ground truth,
dùng cho các nhóm ngày/đêm, xe máy, xe khó; nhóm không có biển chuẩn vẫn trả `null`.

## Chạy cùng một nguồn

Chuẩn bị video và JSONL ground truth theo
`data/templates/ground_truth.example.jsonl`. Đó chỉ là mẫu schema, không phải
nhãn thực tế của video YouTube. `source_id`, `camera_id`, mốc bắt đầu/kết thúc
trong ground truth phải khớp đầu vào; hai lần chạy dùng cùng video/ROI/thời lượng.

```powershell
.\.venv312\Scripts\vehicle-ai.exe run --source data/test.mp4 --source-id test-01 --camera CAM-01 --config configs/before.yaml --output runtime/eval-before
.\.venv312\Scripts\vehicle-ai.exe run --source data/test.mp4 --source-id test-01 --camera CAM-01 --config configs/after.yaml --output runtime/eval-after
.\.venv312\Scripts\vehicle-ai.exe evaluate-events --truth data/test.truth.jsonl --baseline runtime/eval-before/events.merged.jsonl --predictions runtime/eval-after/events.merged.jsonl --output runtime/compare-merged.json
```

Tạo hai cấu hình `before.yaml`, `after.yaml` trong `configs/` trước khi chạy;
giữ cùng đường dẫn weights và chỉ thay **một** ngưỡng mỗi lần. Các thư mục output
phải chưa tồn tại. `run.json` lưu cấu hình, hash video/model, dependencies và
`brand_classes` thực tế của model hãng.

So sánh ngưỡng **không** dùng `--realtime`: bỏ frame phụ thuộc lịch CPU nên hai
lần chạy có thể khác nhau chỉ vì nhiễu. Chạy không bỏ frame để so ngưỡng, rồi
chạy `--realtime` một lần ở cuối để kiểm tra hành vi thực tế trên web.

Mỗi lần `run` xuất hai file cùng ground truth:

- `events.jsonl`: sự kiện track thô (`event_stage: raw`), dùng để đo lỗi tách/trùng
  track. Không phải danh sách thẻ web.
- `events.merged.jsonl`: sự kiện sau gộp theo camera + biển (`event_stage: merged`),
  do cùng `EvidenceRecorder` với web tạo ra, kèm `raw_event_ids`, `source_track_ids`,
  tổng phiếu và trạng thái tin cậy. Đây mới là thứ phản ánh kết quả web, gồm cả
  việc cộng phiếu khi Track ID đổi.

Chấm cả hai file trên cùng ground truth và cùng IoU. So `events.jsonl` trước/sau
để biết thay đổi có làm tách/trùng track không; so `events.merged.jsonl` trước/sau
để biết việc cộng phiếu và gộp biển có giúp ích không. Không trộn hai cấp trong
một con số recall.

## Tìm đúng khâu làm sót

Mỗi track được gắn `rejection_reasons` (số lần) và `rejection_detail` (giá trị/giới
hạn gần nhất) ngay khi một khâu từ chối mẫu. Hai trường này nằm trong cả
`events.jsonl` và `events.merged.jsonl`, trong `/api/events`, cột CSV
`rejection_reasons`, và mục “Tìm chỗ làm sót xe” trên web. Đọc theo thứ tự:

| Mã | Khâu | Ý nghĩa |
| --- | --- | --- |
| `roi_rejected` | Detector/ROI | Tâm xe nằm ngoài đa giác ROI. |
| `vehicle_too_small` | Lọc xe | Chiều rộng crop xe nhỏ hơn `min_vehicle_width`. |
| `plate_model_disabled`, `ocr_disabled` | Cấu hình | Chưa bật model biển hoặc OCR. |
| `no_plate_detected` | Detector biển | Không tìm thấy biển; kiểm tra `plate_imgsz`, `plate_lower_fraction`, `plate_full_scan_interval`. |
| `plate_too_small`, `plate_too_blurry` | Lọc ảnh biển | Biển dưới `min_plate_width` hoặc `min_plate_sharpness`. |
| `ocr_budget_skipped` | Ngân sách | Ảnh mới không tốt hơn `ocr_top_k` ảnh đã OCR. |
| `empty_vehicle_crop`, `empty_plate_crop` | Crop | Vùng cắt rỗng. |
| `ocr_rejected` | OCR | Chuỗi đọc ra bị từ chối (định dạng hoặc điểm). `rejection_detail` ghi chuỗi và điểm. |
| `quality_below_threshold`, `plate_not_confirmed` | Chốt | Ảnh dưới ngưỡng hoặc phiếu biển chưa đủ điều kiện. |
| `merged_duplicate`, `low_dropped`, `no_qualified_evidence`, `hidden_by_recognized_only` | Gộp/hiển thị | Bị gộp vào track khác, bỏ nhãn vàng, không có ảnh đạt ngưỡng, hoặc ẩn vì chế độ chỉ lấy xe đọc được biển. |

Số sau mỗi mã là **số lần từ chối trong track**, không phải số xe riêng; luôn đọc
kèm `event_id`, `source_track_ids` và `finalization_reason`. Trạng thái web cũng
hiện `detector_boxes`, `roi_rejected_this_frame`, `vehicle_too_small_this_frame`,
`frame_size` và ngưỡng đang dùng. Nếu mắt thấy xe nhưng **không có track nào**,
detector đã loại từ trước khi tạo track; hệ thống không thể tự gán mã cho xe đó.
Khi đó chỉ có thể đối chiếu ground truth với `detector_confidence`, ROI, độ phân
giải và `detector_imgsz`. Các xe bị ẩn/ gộp thì vẫn có mã riêng ở cấp gộp.

Việc thêm mã lý do và thống kê **không thay đổi ngưỡng hay quyết định chốt**.
Khi có ground truth, nối dự đoán đã ghép một-một với từng lần đọc/ảnh rồi vẽ phân
bố điểm của nhóm đúng và nhóm sai (tách riêng ngày/đêm, xe máy/ô tô, hãng biết/
không biết). Chọn ngưỡng tách hai phân bố đó, đổi **một** ngưỡng mỗi lần chạy và
báo số tuyệt đối. Đừng chọn ngưỡng chỉ vì tỷ lệ chốt tăng.

Output giữ chỉ số lần chạy mới ở cấp gốc, lần cũ trong `baseline` và chênh lệch
`sau - trước` ở `delta_from_baseline`. Giảm CER/missed/duplicate là tốt,
tăng accuracy/recall là tốt. **Không kết luận tốt hơn chỉ vì confirmation_rate cao.**

Báo cáo bằng số tuyệt đối, ví dụ `plate_correct_over_total: "41/52"`,
`field_counts.brand.correct_over_total: "12/20"`. Với vài chục lượt xe, chênh
lệch một hai xe chưa đủ kết luận; hãy ghi rõ chênh lệch và chạy lặp.

## Ngưỡng và tiêu chí riêng

- Biển giữ nguyên `3 lần / tổng 3.0 / trung bình .85 / cách biệt 1.0`.
  **`reads=3` gần như không bao giờ là điều kiện chốt thực tế:** ba lần .9 chỉ
  được tổng 2.7, nên cần ít nhất 4 lần đọc mới đạt tổng 3.0. Nếu muốn chốt sau
  đúng 3 lần đọc, phải hạ `plate_confirm_score` (ví dụ 2.7) và ghi rõ đã đổi.
  Thẻ hiển thị điều kiện chưa đạt để không hiểu nhầm.
- `plate_confirm_share: 0.0` mặc định tắt. Nếu bật (ví dụ .7), đây là điều kiện
  **bổ sung AND**, không tự thay cách biệt tuyệt đối. Muốn thử chỉ tương đối,
  đặt `plate_confirm_margin: 0` trong cấu hình thử nghiệm và ghi rõ thay đổi.
- Loại xe/hãng có ngưỡng riêng: `type_confirm_*`, `brand_confirm_*`, gồm
  `reads=3`, `mean=.7`, `share=.7`. Đây là mặc định kỹ thuật chưa hiệu chỉnh
  bằng ground truth; chỉ chỉnh sau khi có nhãn cho cả hai trường. Chốt biển không
  còn khiến logo ngừng sớm khi hãng chưa chốt.
- Web cộng tóm tắt phiếu biển/loại xe/hãng của các Track ID cùng biển chuẩn hóa
  và cùng camera trong phiên. Mỗi Track ID chỉ được cộng một lần; không cộng
  chung nguồn khác hoặc xe chưa có biển. Danh sách `source_track_ids`, tổng phiếu,
  quyết định và điều kiện chưa đạt được lưu trong metadata.
- Giao diện hiển thị số kết quả đã chốt/nhãn vàng và tỷ lệ chốt của phiên hiện tại.
  Đó là thống kê quyết định, không phải tỷ lệ biển đúng; độ đúng cần ground truth.

## Nhãn hãng và nhóm hãng ngoài taxonomy

`models/logo.pt` đang chạy có 24 lớp viết thường (`audi`, `bmw`, … `volvo`),
**không có `vinfast` và không có `other`**. `configs/classes_logo.example.yaml`
16 lớp chỉ là taxonomy cho lần huấn luyện mới, không phải lớp của model hiện tại.
`run.json` ghi `brand_classes` thực tế; `evaluate-events` đọc danh sách này để
tách nhóm. Có thể chỉ định tay bằng `--brand-classes <yaml>`.

- So sánh hãng không phân biệt hoa/thường và khoảng trắng: ground truth `Toyota`
  khớp dự đoán `toyota`. Nếu không, mọi xe đều bị tính sai hãng.
- Ground truth ghi **đúng tên lớp model** cho hãng model biết, và ghi `vinfast`
  (hoặc hãng thật) cho hãng model chưa biết. Không ghi `other` thay cho VinFast,
  vì như vậy báo cáo sẽ che mất lỗi model không có lớp này.
- Báo cáo `brand_groups` chia `known` / `unknown` / `taxonomy_unspecified`.
  Với VinFast, model có thể đoán sai một hãng khác với confidence cao và
  `brand.confirmed=true`; `confident_wrong` đếm đúng các trường hợp này.
  **Không đọc accuracy hãng như một con số chung**; luôn đọc riêng `known` và
  `unknown`, kèm `confident_wrong` của nhóm `unknown`.
- `vehicle_type` phải dùng đúng chuỗi pipeline xuất ra: `Ô tô con`, `Xe máy`,
  `Xe buýt`, `Xe tải` (không phải `Xe khách`). Quy định trước bán tải/van/xe tải
  nhỏ thuộc nhóm nào và ghi vào quy ước gán nhãn.

## Gán nhãn

Quy tắc đầy đủ, cách xử lý `null`, `tags`/`notes` và các lỗi thường gặp:
xem [GROUND_TRUTH.md](GROUND_TRUTH.md). Mẫu hợp lệ nằm ở
`data/templates/ground_truth.example.jsonl`.

Tóm tắt: gán độc lập với model; một lượt xe một sự kiện; gán cả xe khó; biển
không đọc được để `null`; `brand` viết thường; dùng đúng chuỗi loại xe của pipeline.
`tag_groups` trong báo cáo tách recall/độ đúng biển theo `tags` (ngày/đêm, xe máy,
xe khó…). Chia clip thành tập chỉnh ngưỡng và tập giữ nguyên để báo cáo.
