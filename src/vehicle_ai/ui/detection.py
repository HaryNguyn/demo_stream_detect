"""Recognition UI: presentation and callbacks, without database/file implementation."""
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from ..capture import stream_source
from ..config import PipelineConfig, load_config
from ..evidence import save_event_images
from ..inference import optional_yolo
from ..model_loader import load_recognizer
from ..ocr import VietnamesePlateRecognizer
from ..persistence import create_event_store
from ..pipeline import analyze_video
from ..roi import validate_roi
from ..settings import image_root, project_dir


def render_model_settings(root):
    models = root / "models"
    with st.sidebar:
        st.header("Model nhận diện")
        options = {
            "use_config": st.checkbox("Dùng pipeline YAML (model đã train / CRNN)", value=False),
            "config_path": st.text_input("Pipeline YAML", str(root / "configs" / "baseline.yaml")),
            "roi_path": st.text_input(
                "ROI JSON (tùy chọn)", help="File ROI lưu từ giao diện duyệt nhãn ở cổng 8502."
            ),
            "vehicle_weights": st.text_input("Model loại xe", str(models / "yolo11n.pt")),
            "logo_weights": st.text_input("Đường dẫn model logo (.pt)", str(models / "logo.pt")),
            "plate_weights": st.text_input(
                "Đường dẫn model phát hiện biển số (.pt)", str(models / "DetectLisence_YOLO11.pt")
            ),
        }
        st.subheader("OCR biển số")
        options["ocr_choice"] = st.selectbox(
            "Mô hình OCR",
            ["CCT-VN ONNX", "OCR model Việt Nam ONNX"],
        )
        if options["ocr_choice"] == "CCT-VN ONNX":
            options["ocr_weights"] = st.text_input("Model CCT-VN ONNX", str(models / "cct_s_v1_vn.onnx"))
            options["ocr_config"] = st.text_input(
                "Tệp cấu hình CCT-VN (.yaml)", str(models / "cct_s_v1_vn_plate_config.yaml")
            )
        else:
            options["ocr_weights"] = st.text_input("Model OCR Việt Nam ONNX", str(models / "ocr_model.onnx"))
            options["ocr_config"] = st.text_input(
                "Tệp cấu hình OCR Việt Nam (.yaml)", str(models / "ocr_model.yaml")
            )
    return options


def prepare_models(options):
    """Resolve manual/YAML settings only when the user starts analysis."""
    if options["use_config"]:
        config = load_config(options["config_path"])
    else:
        config = PipelineConfig(
            vehicle_weights=options["vehicle_weights"].strip() or str(project_dir() / "models/yolo11n.pt"),
            logo_weights=options["logo_weights"].strip(),
            plate_weights=options["plate_weights"].strip(),
        )
    if options["roi_path"].strip():
        data = json.loads(Path(options["roi_path"].strip()).read_text(encoding="utf-8"))
        config.roi = validate_roi(data["points"])
    logo_model = optional_yolo(config.logo_weights)
    plate_model = optional_yolo(config.plate_weights)
    recognizer = None
    if options["use_config"]:
        recognizer = load_recognizer(config)
    elif plate_model is not None:
        weights, config_path = options["ocr_weights"].strip(), options["ocr_config"].strip()
        if not weights or not config_path:
            raise ValueError("Để OCR, nhập đủ model ONNX và tệp YAML.")
        recognizer = VietnamesePlateRecognizer(weights, config_path)
    return config, logo_model, plate_model, recognizer


def csv_bytes(rows):
    buffer = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def analyze_source(source, camera, temporary_dir, index, models, store, event_preview, max_seconds):
    config, logo_model, plate_model, recognizer = models
    st.subheader(camera)
    status, preview = st.empty(), st.empty()
    diagnostic_status, diagnostic_image = st.empty(), st.empty()
    uploaded = hasattr(source, "read")
    if uploaded:
        input_source = Path(temporary_dir) / f"{index}{Path(source.name).suffix.lower()}"
        with input_source.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
        source.seek(0)
    else:
        input_source = source
    images = image_root()

    def show_frame(frame, total, image):
        count = f"{frame}/{total}" if total > 0 else str(frame)
        status.write(f"Đã xử lý đến khung hình {count}")
        preview.image(image, channels="BGR")

    def save_event(row):
        event = store.append(row)
        try:
            _, _, saved = save_event_images(event, images)
            store.update_images(event["event_id"], saved)
        except OSError as error:
            st.warning(f"Không lưu được ảnh sang {images}: {error}")
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
        input_source, camera, config.vehicle_weights, logo_model, plate_model, recognizer,
        on_frame=show_frame, on_event=save_event, max_seconds=max_seconds, collect_results=False,
        on_diagnostic=show_diagnostic, config=config,
    )
    status.write("Hoàn tất phiên · sự kiện đã lưu vào database")
    if uploaded:
        input_source.unlink()


def main():
    st.set_page_config(page_title="Nhận diện xe từ video", layout="wide")
    st.title("Nhận diện xe từ video")
    store = create_event_store()
    options = render_model_settings(project_dir())
    videos = st.file_uploader(
        "Chọn video từ một hoặc nhiều camera",
        type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True,
    )
    stream_url = st.text_input("Hoặc nhập link stream (YouTube, RTSP, HTTP...)", "")
    source_mode = st.selectbox("Loại URL", ["Tự động / luồng trực tiếp", "Trang web (yt-dlp)"])
    max_seconds = st.number_input(
        "Giới hạn thời lượng video xử lý (giây, 0 = đến hết nguồn)", min_value=0, value=60
    )
    st.caption(
        "Thử nghiệm: mặc định chốt sự kiện sau 2 giây mất dấu xe hoặc khi kết thúc phiên. "
        "Mỗi track là một lượt quan sát; xe bị mất dấu lâu có thể tạo sự kiện mới."
    )
    event_preview = st.empty()
    event_preview.dataframe(store.recent(), hide_index=True, width="stretch")
    if st.button("Phân tích video", type="primary", disabled=not videos and not stream_url.strip()):
        st.session_state.pop("rows", None)
        try:
            models = prepare_models(options)
            inputs = [(video.name, video) for video in videos]
            if stream_url.strip():
                inputs.append(stream_source(stream_url, webpage=source_mode == "Trang web (yt-dlp)"))
            with TemporaryDirectory() as temporary_dir:
                for index, (name, source) in enumerate(inputs, 1):
                    analyze_source(
                        source, f"{index} · {name}", temporary_dir, index,
                        models, store, event_preview, max_seconds,
                    )
            st.session_state.rows = store.recent()
        except Exception as error:
            st.error(f"Phân tích bị gián đoạn. {type(error).__name__}: {error}")
    if "rows" in st.session_state:
        st.subheader("Kết quả theo camera")
        rows = st.session_state.rows
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
            st.download_button("Tải CSV", csv_bytes(rows), "ket_qua_xe.csv", "text/csv")
        else:
            st.info("Chưa tìm thấy xe trong video.")
