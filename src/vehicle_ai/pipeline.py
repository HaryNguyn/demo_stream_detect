"""Video orchestration. No Streamlit or database dependency."""
import math

import cv2
from ultralytics import YOLO

from .capture import open_capture
from .config import PipelineConfig
from .inference import draw_recognition
from .session import RecognitionSession


def analyze_video(path, camera, vehicle_weights, logo_model=None, plate_model=None,
                  recognizer=None, on_frame=None, on_event=None, max_seconds=0,
                  collect_results=True, on_diagnostic=None, *, config=None, source_id=None, on_stats=None):
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError("max_seconds phải không âm và hữu hạn.")
    config = config or PipelineConfig(vehicle_weights=vehicle_weights)
    detector = YOLO(config.vehicle_weights)
    session = RecognitionSession(
        detector, config, camera, source_id=source_id,
        logo_model=logo_model, plate_model=plate_model, recognizer=recognizer,
    )
    completed = []

    def publish(rows):
        for row in rows:
            if on_event:
                on_event(row)
            if collect_results:
                completed.append(row)

    capture = open_capture(path)
    try:
        if not capture.isOpened():
            raise ValueError("Không mở được video. Kiểm tra nguồn và codec.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        total = int(count) if math.isfinite(count) and count > 0 else 0
        index = 0
        while not max_seconds or index / fps < max_seconds:
            ok, frame = capture.read()
            if not ok:
                break
            publish(session.feed(frame, index / fps))
            if on_frame and index % 10 == 0:
                on_frame(index + 1, total, draw_recognition(session.last_result, session.tracks, config.roi))
            if on_diagnostic and index % 10 == 0:
                on_diagnostic(session.diagnostics.copy())
            index += 1
        if index == 0:
            detail = capture.error_text() if hasattr(capture, "error_text") else ""
            raise ValueError(f"Video không có khung hình đọc được. {detail}")
        reason = "duration_limit" if max_seconds and index / fps >= max_seconds else "end_of_source"
        publish(session.finalize(reason))
        if on_stats:
            on_stats({"frames": index, "source_fps": fps, "content_seconds": index / fps})
        if on_diagnostic:
            on_diagnostic(session.diagnostics.copy())
    finally:
        capture.release()
    return completed
