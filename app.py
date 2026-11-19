import csv
import io
import os
import json
import re
from datetime import datetime
from base64 import b64decode
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from events import EventStore
from pipeline import analyze_video, optional_yolo
from recognition import VietnamesePlateRecognizer
from sources import stream_source

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
DEFAULT_VEHICLE_MODEL = str(APP_DIR / "yolo11n.pt")
IMAGE_ROOT = Path(os.getenv("VDTC_IMAGE_ROOT", r"D:\VDTC11"))


def safe_name(value):
    return re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("._") or "UNKNOWN"


def save_event_images(row):
    event_id = row["event_id"]
    recorded_at = datetime.now()
    camera = safe_name(row.get("Camera", "camera"))
    identity = safe_name(row.get("Xe ID") or row.get("Biển số") or f"TRACK-{row.get('Track ID', 0)}")
    event_dir = IMAGE_ROOT / recorded_at.strftime("%Y-%m-%d") / camera / identity / event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    saved = {"detected": [], "plate": []}
    for kind, field in (("detected", "Danh sách ảnh xe"), ("plate", "Danh sách ảnh biển")):
        for index, image in enumerate(row.pop(field, []) or [], start=1):
            if not image or image.get("encoding") != "base64":
                continue
            suffix = "jpg"
            path = event_dir / f"{kind}_{index:03d}.{suffix}"
            path.write_bytes(b64decode(image["data"]))
            saved[kind].append({
                "path": str(path),
                "mime_type": image.get("mime_type", "image/jpeg"),
                "width": image.get("width"),
                "height": image.get("height"),
            })

    metadata_path = event_dir / "metadata.json"
    row["Thư mục ảnh"] = str(event_dir)
    row["images"] = saved
    metadata_path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return event_dir, metadata_path, saved

st.set_page_config(page_title="Nhận diện xe từ video", layout="wide")
st.title("Nhận diện xe từ video")
store = EventStore()

with st.sidebar:
    st.header("Model nhận diện")
    vehicle_path = st.text_input("Model loại xe", DEFAULT_VEHICLE_MODEL)
    bundled_logo = MODEL_DIR / "logo.pt"
    logo_path = st.text_input(
        "Đường dẫn model logo (.pt)",
        str(bundled_logo),
    )
    bundled_plate = MODEL_DIR / "DetectLisence_YOLO11.pt"
    plate_path = st.text_input(
        "Đường dẫn model phát hiện biển số (.pt)",
        str(bundled_plate),
    )
    st.subheader("OCR biển số")
    ocr_engine = st.selectbox("Engine OCR", ["CCT-VN ONNX", "PaddleOCR REC"])
    rec_dir = ""
    rec_name = ""
    cct_onnx = ""
    cct_config = ""
    if ocr_engine == "PaddleOCR REC":
        rec_dir = st.text_input("Thư mục PaddleOCR REC")
        rec_name = st.text_input("Tên model PaddleOCR REC")
    else:
        bundled_onnx = MODEL_DIR / "cct_s_v1_vn.onnx"
        bundled_config = MODEL_DIR / "cct_s_v1_vn_plate_config.yaml"
        cct_onnx = st.text_input("Model CCT-VN ONNX", str(bundled_onnx))
        cct_config = st.text_input("Tệp cấu hình CCT-VN (.yaml)", str(bundled_config))

videos = st.file_uploader(
    "Chọn video từ một hoặc nhiều camera",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=True,
)
stream_url = st.text_input(
    "Hoặc nhập link stream (YouTube, RTSP, HTTP...)",
    "",
)
source_mode = st.selectbox("Loại URL", ["Tự động / luồng trực tiếp", "Trang web (yt-dlp)"])
max_seconds = st.number_input("Giới hạn thời lượng video xử lý (giây, 0 = đến hết nguồn)", min_value=0, value=60)
st.caption("Thử nghiệm: chốt sự kiện sau 2 giây mất dấu xe hoặc khi kết thúc phiên. "
           "Mỗi track là một lượt quan sát; xe bị mất dấu lâu có thể tạo sự kiện mới.")
event_preview = st.empty()
event_preview.dataframe(store.recent(), hide_index=True, width="stretch")

if st.button("Phân tích video", type="primary", disabled=not videos and not stream_url.strip()):
    st.session_state.pop("rows", None)
    try:
        logo_model = optional_yolo(logo_path.strip())
        plate_model = optional_yolo(plate_path.strip())
        recognizer = None
        if plate_model is not None:
            if ocr_engine == "PaddleOCR REC":
                ocr_fields = [rec_dir.strip(), rec_name.strip()]
                if not all(ocr_fields):
                    raise ValueError("Để dùng PaddleOCR, nhập đủ thư mục REC và tên model REC.")
                if not Path(rec_dir.strip()).is_dir():
                    raise FileNotFoundError(rec_dir)
                from paddleocr import TextRecognition

                recognizer = TextRecognition(model_dir=rec_dir.strip(), model_name=rec_name.strip())
            else:
                cct_fields = [cct_onnx.strip(), cct_config.strip()]
                if not all(cct_fields):
                    raise ValueError("Để dùng CCT-VN ONNX, nhập đủ model ONNX và tệp YAML.")
                recognizer = VietnamesePlateRecognizer(
                    cct_onnx.strip(),
                    cct_config.strip(),
                )

        with TemporaryDirectory() as temp:
            sources = [(video.name, video) for video in videos]
            if stream_url.strip():
                sources.append(stream_source(stream_url, webpage=source_mode == "Trang web (yt-dlp)"))

            for index, (source_name, source) in enumerate(sources, start=1):
                camera = f"{index} · {source_name}"
                st.subheader(camera)
                status = st.empty()
                preview = st.empty()
                diagnostic_status = st.empty()
                diagnostic_image = st.empty()
                if hasattr(source, "read"):
                    file_path = Path(temp) / f"{index}{Path(source.name).suffix.lower()}"
                    with file_path.open("wb") as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
                    source.seek(0)
                    input_source = file_path
                else:
                    input_source = source

                def show_frame(frame, total, image):
                    count = f"{frame}/{total}" if total > 0 else str(frame)
                    status.write(f"Đã xử lý đến khung hình {count}")
                    preview.image(image, channels="BGR")

                def save_event(row):
                    event = store.append(row)
                    event_dir = None
                    try:
                        event_dir, metadata_path, saved = save_event_images(event)
                        store.update_images(event["event_id"], saved)
                        event["Thư mục ảnh"] = str(event_dir)
                        event["Metadata"] = str(metadata_path)
                        event["Ảnh đã lưu"] = saved
                    except OSError as error:
                        st.warning(f"Không lưu được ảnh sang {IMAGE_ROOT}: {error}")
                    event_preview.dataframe(store.recent(), hide_index=True, width="stretch")

                def show_diagnostic(info):
                    diagnostic_status.write(
                        f"Biển số: {info['status']} · Mẫu xe: {info['samples']} · "
                        f"Phát hiện biển: {info['plate_detections']} · "
                        f"OCR đạt: {info['accepted_reads']} · "
                        f"Chuỗi OCR: {info['text'] or '—'} · Điểm: {info['score']:.2f}"
                    )
                    if info["image"] is not None:
                        diagnostic_image.image(
                            info["image"], channels="BGR", width=360,
                            caption=f"Mẫu gần nhất — Track {info.get('track_id', '')}: {info['status']}",
                        )

                analyze_video(
                    input_source, camera, vehicle_path.strip() or DEFAULT_VEHICLE_MODEL,
                    logo_model, plate_model, recognizer, show_frame,
                    on_event=save_event, max_seconds=max_seconds, collect_results=False,
                    on_diagnostic=show_diagnostic,
                )
                status.write("Hoàn tất phiên · sự kiện đã lưu vào MariaDB")
                if hasattr(source, "read"):
                    file_path.unlink()
        st.session_state.rows = store.recent()
    except Exception as error:
        st.error(f"Phân tích bị gián đoạn. {type(error).__name__}: {error}")

if "rows" in st.session_state:
    st.subheader("Kết quả theo camera")
    rows = st.session_state.rows
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Tải CSV", buffer.getvalue().encode("utf-8-sig"), "ket_qua_xe.csv", "text/csv")
    else:
        st.info("Chưa tìm thấy xe trong video.")
