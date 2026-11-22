# Nguồn tham khảo và thành phần kế thừa

Codebase phát triển từ bản gộp
[`HaryNguyn/demo_stream_detect`](https://github.com/HaryNguyn/demo_stream_detect)
commit `e7a5231`. Refactor không thay đổi nguồn gốc mã hoặc trọng số kế thừa.

Các checkout tham khảo không được import trong runtime. Đã chuyển nguyên vẹn,
bao gồm lịch sử Git và các file giấy phép hiện có, sang:

`../reference-archive-20261119/third_party/`

**Cập nhật:** archive bên ngoài đã được xóa theo yêu cầu người dùng.
Các URL và commit dưới đây được giữ để có thể tra cứu/tải lại nguồn.

| Thư mục archive | Nguồn | Commit lúc di chuyển |
|---|---|---|
| LicensePlateDetection | https://github.com/ddkhoa2605/LicensePlateDetection | `f1c732e2270becc44df1de6529372f63ef251076` |
| cars-logos-plates-detection | https://github.com/TeamXGroup/cars-logos-plates-detection | `b2d35e76f8d220677a54ee5437635748427a7f1a` |
| cars-logos-plates-detection-clean | https://github.com/TeamXGroup/cars-logos-plates-detection | `b2d35e76f8d220677a54ee5437635748427a7f1a` |

Hai checkout cars-logos cùng commit nhưng nội dung cục bộ có thể khác nhau;
đã giữ cả hai, không suy luận rằng có thể xóa một bản.

Thông tin trọng số đang dùng ở `models/manifest.yaml`. Tất cả thư viện Python,
kể cả dependency export ONNX, được liệt kê trong `requirements.txt`. Các thành
phần giữ giấy phép riêng từ nguồn.
