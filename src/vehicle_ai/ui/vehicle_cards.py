"""Vehicle evidence cards shared by live and completed recognition views."""
from pathlib import Path

import streamlit as st


def evidence_bytes(row, kind):
    """Read the latest available saved evidence, tolerating missing files."""
    for image in reversed((row.get("images") or {}).get(kind, [])):
        try:
            return Path(image["path"]).read_bytes()
        except (OSError, KeyError):
            continue
    return None


def render_vehicle_cards(rows, interactive=False):
    st.subheader(f"Xe đã ghi nhận ({len(rows)})")
    st.caption("Sự kiện được chốt khi xe mất dấu hoặc khi kết thúc phiên.")
    with st.container(height=680):
        if not rows:
            st.info("Đang chờ kết quả nhận diện xe.")
        for row in rows:
            identity = str(row["event_id"])
            plate = row.get("Biển số") or "Chưa đọc được biển số"
            plate_image = evidence_bytes(row, "plate")
            vehicle_image = evidence_bytes(row, "detected")
            with st.container(border=True):
                thumbnail, title = st.columns([1, 3])
                with thumbnail:
                    if plate_image or vehicle_image:
                        st.image(plate_image or vehicle_image, width="stretch")
                with title:
                    st.subheader(plate)
                    st.caption(f"Xe #{row.get('Track ID', '—')} · {row.get('Camera', '')}")
                started = float(row.get("Xuất hiện (giây)") or 0)
                ended = row.get("Kết thúc (giây)")
                timing = f"{started:.1f}s"
                if ended is not None:
                    timing += f" → {float(ended):.1f}s"
                st.caption(f"Thời điểm: {timing}")
                st.write(
                    f"**Loại xe:** {row.get('Loại xe') or 'Chưa xác định'} · "
                    f"**Hãng:** {row.get('Hãng') or 'Chưa xác định'}"
                )
                if row.get("Biển số"):
                    score = min(1.0, max(0.0, float(row.get("Điểm biển") or 0)))
                    st.progress(score, text=f"Điểm OCR tốt nhất: {score:.1%}")
                else:
                    st.caption("Chưa có kết quả OCR đạt ngưỡng.")
                if interactive:
                    with st.expander("Xem ảnh xe / biển số"):
                        if vehicle_image:
                            st.image(vehicle_image, caption="Ảnh xe", width="stretch")
                        if plate_image:
                            st.image(plate_image, caption="Ảnh biển số", width="stretch")
                        if not vehicle_image and not plate_image:
                            st.caption("Chưa có ảnh được lưu cho xe này.")
                        st.code(row.get("Biển số") or "", language=None)
                    vehicle_action, plate_action = st.columns(2)
                    if vehicle_image:
                        vehicle_action.download_button(
                            "Tải ảnh xe", vehicle_image, f"{identity}_xe.jpg",
                            "image/jpeg", key=f"vehicle_{identity}", on_click="ignore",
                        )
                    if plate_image:
                        plate_action.download_button(
                            "Tải ảnh biển", plate_image, f"{identity}_bien.jpg",
                            "image/jpeg", key=f"plate_{identity}", on_click="ignore",
                        )
