"""Annotation review UI and per-video ROI controls."""
import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from ..annotation import FIELDS, NAMES, approve_record, atomic_json, draw_boxes
from ..roi import validate_roi
from ..roi_editor import roi_editor
from ..settings import project_dir


def render_roi(directory, record, image):
    st.subheader("Vẽ ROI — vùng nhận diện")
    st.caption(
        f"ROI lưu riêng cho video {record['group_id']}, dùng chung các frame của video này. "
        "Kéo hình chữ nhật hoặc chọn đa giác rồi bấm từng đỉnh. ROI không thay đổi nhãn train."
    )
    roi_key = hashlib.sha256(record["group_id"].encode()).hexdigest()[:16]
    roi_path = directory / "rois" / f"{roi_key}.json"
    saved = json.loads(roi_path.read_text(encoding="utf-8")) if roi_path.is_file() else {"points": []}
    context = f"{directory.resolve()}:{record['group_id']}:{record['id']}"
    value = roi_editor(image, saved["points"], context=context, key=f"roi-{roi_key}")
    if value and value.get("context") == context:
        nonce_key = f"roi-last-{directory.resolve()}-{roi_key}"
        if st.session_state.get(nonce_key) != value.get("nonce"):
            try:
                points = validate_roi(value["points"])
                roi_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(roi_path, {
                    "schema_version": "1.0", "group_id": record["group_id"], "points": points,
                    "rule": "vehicle_bbox_center", "reference_image": record["image"],
                })
                st.session_state[nonce_key] = value["nonce"]
                st.success("Đã lưu ROI." if points else "Đã bỏ ROI — nhận diện toàn ảnh.")
            except (ValueError, TypeError) as error:
                st.error(str(error))
    if roi_path.is_file():
        st.code(f'--roi-file "{roi_path.resolve()}"', language="text")
    st.caption(
        "Pipeline chỉ giữ xe có tâm bbox nằm trong ROI. Nhãn xe ở phần dưới vẫn được duyệt trên toàn ảnh."
    )


def render_annotation(directory, record, image):
    st.subheader("Duyệt nhãn xe")
    st.caption(
        "Nhãn tự động có thể sai hoặc bỏ sót. Kiểm tra toàn bộ xe trong ảnh trước khi duyệt. "
        "Ảnh bị mờ/khó gán nhãn có thể bỏ qua."
    )
    st.write(" · ".join(f"**{i}** = {name}" for i, name in enumerate(NAMES)))
    st.caption(
        "Tọa độ chuẩn hóa 0–1: x_center/y_center là tâm, width/height là kích thước. "
        "Có thể sửa loại xe/tọa độ, xóa hàng bằng ô chọn bên trái, thêm hàng ở cuối bảng."
    )
    boxes = record.get("reviewed_boxes", record["suggested_boxes"])
    edited = st.data_editor(
        pd.DataFrame(boxes, columns=FIELDS), num_rows="dynamic", hide_index=True,
        key=f"boxes-{record['id']}-{record.get('reviewed_at', '')}",
        column_config={
            "class_id": st.column_config.NumberColumn(
                "Loại (0–3)", min_value=0, max_value=3, step=1, required=True
            ),
            **{key: st.column_config.NumberColumn(
                key, min_value=0.0, max_value=1.0, step=0.001, format="%.4f", required=True
            ) for key in FIELDS[1:]},
        },
    )
    valid = True
    try:
        preview = draw_boxes(image, edited.to_dict("records"))
    except (ValueError, TypeError) as error:
        st.error(str(error))
        preview, valid = image, False
    left, right = st.columns(2)
    left.image(image, channels="BGR", caption="Ảnh gốc", width="stretch")
    right.image(preview, channels="BGR", caption="Khung sau sửa — số khớp thứ tự hàng", width="stretch")
    confirm_empty = st.checkbox(
        "Xác nhận ảnh không có xe thuộc các lớp trên (chỉ cần khi bảng rỗng)",
        key=f"empty-{record['id']}",
    )
    approve, skip = st.columns(2)
    if approve.button("Lưu và duyệt ảnh", type="primary", disabled=not valid):
        try:
            approve_record(directory, record["id"], edited.to_dict("records"), confirm_empty=confirm_empty)
            st.rerun()
        except (ValueError, TypeError) as error:
            st.error(str(error))
    if skip.button("Bỏ qua ảnh này"):
        approve_record(directory, record["id"], [], action="skip")
        st.rerun()
    st.info("Ảnh đã duyệt được xuất vào reviewed_manifest.csv. Chưa chia tập hoặc train tự động.")


def main():
    st.set_page_config(page_title="Duyệt nhãn xe", layout="wide")
    st.title("Duyệt và sửa nhãn xe trước khi train")
    directory = Path(st.sidebar.text_input(
        "Thư mục nhãn", str(project_dir() / "data/review/vehicle-v001")
    ))
    path = directory / "review.json"
    if not path.is_file():
        st.info("Chưa có review.json. Chạy bước tạo nhãn gợi ý trước.")
        st.stop()
    project = json.loads(path.read_text(encoding="utf-8"))
    if project["generation_status"] != "completed":
        st.info("Đang tạo nhãn gợi ý. Tải lại khi hoàn tất.")
        st.stop()
    records = project["records"]
    approved = sum(r["status"] == "approved" for r in records)
    st.sidebar.write(f"Đã duyệt: {approved}/{len(records)}")
    group = st.sidebar.selectbox("Video", ["Tất cả"] + sorted({r["group_id"] for r in records}))
    status = st.sidebar.selectbox("Trạng thái", ["pending", "approved", "skipped", "all"])
    filtered = [
        record for record in records
        if (group == "Tất cả" or record["group_id"] == group)
        and (status == "all" or record["status"] == status)
    ]
    if not filtered:
        st.success("Không có ảnh trong bộ lọc này.")
        st.stop()
    selected = st.selectbox(
        "Chọn ảnh", range(len(filtered)),
        format_func=lambda i: (
            f"{i + 1}/{len(filtered)} · {filtered[i]['group_id']} · "
            f"{Path(filtered[i]['image']).name}"
        ),
    )
    record = filtered[selected]
    image = cv2.imread(record["image"])
    if image is None:
        st.error("Không đọc được ảnh gốc.")
        st.stop()
    render_roi(directory, record, image)
    st.divider()
    render_annotation(directory, record, image)
