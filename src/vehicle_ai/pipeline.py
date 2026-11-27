"""Video orchestration. No Streamlit or database dependency."""
import math
from contextlib import closing
from copy import copy
from threading import Lock
from time import monotonic, perf_counter

import cv2
from ultralytics import YOLO

from .capture import open_capture
from .config import PipelineConfig
from .inference import draw_anpr_preview, draw_recognition
from .session import RecognitionSession
from .playback import realtime_frames
from .roi import draw_roi


def analyze_video(path, camera, vehicle_weights, logo_model=None, plate_model=None,
                  recognizer=None, on_frame=None, on_event=None, max_seconds=0,
                  collect_results=True, on_diagnostic=None, *, config=None, source_id=None, on_stats=None,
                  should_stop=None, realtime=False, recognized_only=False, overlay_style="vehicle"):
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError("max_seconds phải không âm và hữu hạn.")
    config = config or PipelineConfig(vehicle_weights=vehicle_weights)
    if should_stop and should_stop():
        return []
    detector = YOLO(config.vehicle_weights)
    session = RecognitionSession(
        detector, config, camera, source_id=source_id,
        logo_model=logo_model, plate_model=plate_model, recognizer=recognizer,
    )
    completed = []
    pipeline_diagnostics = {"hidden_by_recognized_only": 0}

    def publish(rows):
        for row in rows:
            if recognized_only and not str(row.get("Biển số") or "").strip():
                pipeline_diagnostics["hidden_by_recognized_only"] += 1
                continue
            if on_event:
                on_event(row)
            if collect_results:
                completed.append(row)

    def emit_diagnostics():
        if not on_diagnostic:
            return
        info = dict(session.diagnostics)
        info.update(pipeline_diagnostics)
        on_diagnostic(info)

    capture = open_capture(path)
    last_frame = None
    try:
        if not capture.isOpened():
            raise ValueError("Không mở được video. Kiểm tra nguồn và codec.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        total = int(count) if math.isfinite(count) and count > 0 else 0
        index = 0
        stopped = False
        if realtime:
            overlay_lock = Lock()
            overlay = None
            preview_clock = {"last": -math.inf}

            def preview_frame(number, frame_total, frame):
                if not on_frame:
                    return
                now = monotonic()
                if now - preview_clock["last"] < 1.0 / config.preview_fps:
                    return
                preview_clock["last"] = now
                with overlay_lock:
                    snapshot = overlay
                if overlay_style == "anpr":
                    image = draw_anpr_preview(
                        frame, snapshot[0] if snapshot else None, snapshot[1] if snapshot else {},
                        config.roi, inference_fps=snapshot[2] if snapshot else 0.0,
                    )
                else:
                    image = (draw_recognition(snapshot[0], snapshot[1], config.roi, frame=frame)
                             if snapshot is not None else draw_roi(frame.copy(), config.roi))
                on_frame(number, frame_total, image)

            with closing(realtime_frames(
                capture, fps, total, max_seconds, preview_frame, should_stop,
            )) as frames:
                for frame_index, frame in frames:
                    last_frame = frame
                    index = frame_index + 1
                    if frame_index % config.detector_frame_stride:
                        continue
                    started = perf_counter()
                    publish(session.feed(frame, frame_index / fps))
                    inference_fps = 1 / max(perf_counter() - started, 1e-6)
                    # Preview reads an independent snapshot while OCR updates live tracks.
                    tracks = {}
                    for track_id, vehicle in session.tracks.items():
                        frozen = copy(vehicle)
                        frozen.plates = {text: list(scores) for text, scores in vehicle.plates.items()}
                        tracks[track_id] = frozen
                    with overlay_lock:
                        overlay = (session.last_result.cpu(), tracks, inference_fps)
                    if on_diagnostic:
                        emit_diagnostics()
            stopped = bool(should_stop and should_stop())
        else:
            while not max_seconds or index / fps < max_seconds:
                if should_stop and should_stop():
                    stopped = True
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                last_frame = frame
                if index % config.detector_frame_stride == 0:
                    publish(session.feed(frame, index / fps))
                if on_frame and index % 10 == 0:
                    on_frame(index + 1, total, draw_recognition(session.last_result, session.tracks, config.roi))
                if on_diagnostic and index % 10 == 0:
                    emit_diagnostics()
                index += 1
        if index == 0 and not stopped:
            detail = capture.error_text() if hasattr(capture, "error_text") else ""
            raise ValueError(f"Video không có khung hình đọc được. {detail}")
        reason = "duration_limit" if max_seconds and index / fps >= max_seconds else "end_of_source"
        if stopped:
            reason = "user_stop"
        publish(session.finalize(reason))
        if on_stats:
            stats = {"frames": index, "source_fps": fps, "content_seconds": index / fps}
            shape = getattr(last_frame, "shape", None)
            if shape is not None and len(shape) >= 2:
                stats["frame_size"] = f"{shape[1]}x{shape[0]}"
            on_stats(stats)
        if on_diagnostic:
            emit_diagnostics()
    finally:
        capture.release()
    return completed
