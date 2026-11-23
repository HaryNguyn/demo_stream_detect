"""Local HTTP UI and background video analysis, independent of Streamlit."""
import math
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Condition, Event, Lock, Thread

from flask import Flask, Response, abort, jsonify, request, send_file

from .ip_camera import build_rtsp_url, mask_url, validate_stream_url
from .settings import get_setting, image_root
from .result_filter import EvidenceRecorder, image_score, plates_only_rows, unique_results


def create_app(store=None):
    from .persistence import create_event_store

    app = Flask(__name__, static_folder="web_static", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024**3
    store = store if store is not None else create_event_store()
    lock = Lock()
    frame_changed = Condition(lock)
    stop_requested = Event()
    state = {"status": "idle", "frame": 0, "total": 0, "error": "", "camera": "", "revision": 0,
             "diagnostic": {}, "filter_stats": {}, "plates_only": True,
             "hidden_by_plates_only": 0}
    preview = {"data": None}

    def update(**values):
        with lock:
            state.update(values)

    def run(sources, options, max_seconds, temporary):
        try:
            import cv2

            from .capture import stream_source
            from .pipeline import analyze_video
            from .web_models import prepare_models

            config, logo, plate, recognizer = prepare_models(options)
            update(brand_classes=list(logo.names.values()) if logo is not None else [],
                   plates_only=config.plates_only,
                   hidden_by_plates_only=0)
            recorder = EvidenceRecorder(
                store, image_root(), min_quality=config.min_image_quality,
                min_detection=config.min_evidence_detection, min_ocr=config.min_evidence_ocr,
                top_k=config.evidence_top_k, keep_low_confidence=config.keep_low_confidence,
                config=config,
            )

            def frame_ready(frame, total, image):
                height, width = image.shape[:2]
                if width > 960:
                    image = cv2.resize(image, (960, round(height * 960 / width)))
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with frame_changed:
                        preview["data"] = encoded.tobytes()
                        state.update(frame=frame, total=total, revision=state["revision"] + 1)
                        frame_changed.notify_all()

            def event_ready(row):
                recorder.record(row)
                counts = {"confirmed": 0, "low": 0}
                for result in recorder.recorded.values():
                    counts["low" if result.get("confidence_status") == "low" else "confirmed"] += 1
                update(result_counts=counts, filter_stats=dict(recorder.stats))

            def diagnostic_ready(info):
                update(diagnostic={
                    key: info.get(key) for key in
                    ("status", "samples", "plate_detections", "accepted_reads", "text", "score", "track_id",
                     "timings_ms", "ocr_batch", "active_tracks", "confirmed_tracks", "frame_size",
                     "detector_boxes", "roi_rejected_this_frame", "vehicle_too_small_this_frame",
                     "recognition_budget_skipped_this_frame", "roi_rejected_tracks", "rejections",
                     "rejection_detail", "thresholds", "hidden_by_recognized_only")
                })

            for index, (camera, source, is_url) in enumerate(sources, 1):
                if stop_requested.is_set():
                    break
                update(camera=camera, source_index=index, source_count=len(sources), frame=0, total=0,
                       diagnostic={})
                if is_url:
                    _, source = stream_source(source, webpage=options.get("source_mode") == "webpage")
                analyze_video(
                    source, camera, config.vehicle_weights, logo, plate, recognizer,
                    config=config, max_seconds=max_seconds,
                    on_frame=frame_ready, on_event=event_ready, collect_results=False,
                    should_stop=stop_requested.is_set, on_diagnostic=diagnostic_ready,
                    realtime=True, recognized_only=not config.keep_low_confidence,
                    overlay_style=options.get("overlay_style", "anpr"),
                )
            with lock:
                state["status"] = "stopped" if stop_requested.is_set() else "done"
        except Exception as error:
            update(status="error", error=f"{type(error).__name__}: {error}")
        finally:
            if temporary is not None:
                temporary.cleanup()

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/config")
    def config():
        return jsonify(
            ip_cam_url=get_setting("IP_CAM_URL", ""),
            ip_cam_host=get_setting("IP_CAM_HOST", ""),
            ip_cam_port=get_setting("IP_CAM_PORT", "554"),
            ip_cam_user=get_setting("IP_CAM_USER", ""),
            ip_cam_channel=get_setting("IP_CAM_CHANNEL", "1"),
            ip_cam_stream=get_setting("IP_CAM_STREAM", "main"),
            ip_cam_brand=get_setting("IP_CAM_BRAND", "hikvision"),
            ip_cam_path=get_setting("IP_CAM_PATH", ""),
        )

    @app.get("/api/status")
    def status():
        with lock:
            return jsonify(dict(state))

    @app.get("/api/frame")
    def frame():
        from io import BytesIO

        with lock:
            data = preview["data"]
        if data is None:
            abort(404)
        response = send_file(BytesIO(data), mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/events")
    def events():
        rows = recognized_events()
        for row in rows:
            row["photos"] = {
                kind: f"/api/events/{row['event_id']}/images/{kind}?q={image_score(row, kind):.6f}"
                for kind in ("detected", "plate")
                if (row.get("images") or {}).get(kind)
            }
            row["image_scores"] = {kind: image_score(row, kind) for kind in ("detected", "plate")}
            row["photo_candidates"] = {
                kind: [
                    {"url": f"/api/events/{row['event_id']}/images/{kind}?index={index}&q={item.get('quality_score', 0):.6f}",
                     "quality_score": item.get("quality_score", 0),
                     "ocr_score": item.get("ocr_score", 0), "quality_factors": item.get("quality_factors", {})}
                    for index, item in enumerate((row.get("images") or {}).get(kind, []))
                ]
                for kind in ("detected", "plate")
            }
            row.pop("images", None)
        return jsonify(rows)

    @app.get("/api/events.csv")
    def export_csv():
        rows = recognized_events()
        fields = ["Camera", "Track ID", "Biển số", "Loại xe", "Hãng", "Điểm hãng",
                  "Điểm biển", "confidence_status", "Xuất hiện (giây)", "Kết thúc (giây)",
                  "rejection_reasons", "event_id", "recorded_at"]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "rejection_reasons": json.dumps(row.get("rejection_reasons") or {}, ensure_ascii=False),
            })
        return send_file(io.BytesIO(buffer.getvalue().encode("utf-8-sig")), mimetype="text/csv",
                         as_attachment=True, download_name="ket_qua_xe.csv")

    def recognized_events():
        with lock:
            plates_only = bool(state["plates_only"])
        rows = unique_results(store.recent())
        if not plates_only:
            return rows
        # Expose only rows with an actual plate string, same meaning as
        # analyze_video(recognized_only=True). Unread rows stay in the store so
        # their votes can still merge into a later confirmation.
        visible = plates_only_rows(rows)
        with lock:
            state["hidden_by_plates_only"] = len(rows) - len(visible)
        return visible

    @app.get("/api/stream")
    def video_stream():
        def chunks():
            last_revision = -1
            while True:
                with frame_changed:
                    frame_changed.wait_for(
                        lambda: state["revision"] != last_revision and preview["data"] is not None,
                        timeout=1,
                    )
                    data = preview["data"]
                    current = state["revision"]
                    if data is None or current == last_revision:
                        if state["status"] not in {"running", "stopping"}:
                            return
                        continue
                    last_revision = current
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
        return Response(chunks(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/events/<event_id>/images/<kind>")
    def image(event_id, kind):
        if kind not in ("detected", "plate"):
            abort(404)
        row = next((row for row in store.recent() if row["event_id"] == event_id), None)
        if row is None:
            abort(404)
        candidates = (row.get("images") or {}).get(kind, [])
        if "index" in request.args:
            try:
                index = int(request.args["index"])
                if index < 0 or index >= len(candidates):
                    abort(404)
                candidates = [candidates[index]]
            except ValueError:
                abort(400)
        else:
            candidates = sorted(candidates, key=lambda item: item.get("quality_score", 0), reverse=True)
        for item in candidates:
            path = Path(item["path"]).resolve()
            if path.is_relative_to(image_root().resolve()) and path.is_file():
                # Read into memory first: send_file(path) keeps the file handle open
                # until the client finishes the response, which blocks evidence
                # cleanup with WinError 32 on Windows when the gallery is polling.
                return send_file(
                    io.BytesIO(path.read_bytes()), mimetype="image/jpeg",
                    as_attachment=request.args.get("download") == "1",
                    download_name=f"{event_id}_{kind}.jpg",
                )
        abort(404)

    @app.post("/api/stop")
    def stop():
        with lock:
            if state["status"] in {"running", "stopping"}:
                stop_requested.set()
                state["status"] = "stopping"
            return jsonify(status=state["status"])

    @app.post("/api/analyze")
    def analyze():
        uploads = [item for item in request.files.getlist("video") if item.filename]
        url = request.form.get("url", "").strip()
        options = {key: value.strip() for key, value in request.form.items()}
        if options.get("ip_cam_host"):
            try:
                url = build_rtsp_url(
                    options["ip_cam_host"], username=options.get("ip_cam_user", ""),
                    password=options.get("ip_cam_password", ""),
                    port=options.get("ip_cam_port", 554), channel=options.get("ip_cam_channel", 1),
                    stream=options.get("ip_cam_stream", "main"), brand=options.get("ip_cam_brand", "hikvision"),
                    path=options.get("ip_cam_path", ""),
                )
            except ValueError as error:
                return jsonify(error=str(error)), 400
            options["source_mode"] = "auto"
        if url and options.get("source_mode") != "webpage":
            try:
                url = validate_stream_url(url)
            except ValueError as error:
                return jsonify(error=str(error)), 400
        try:
            max_seconds = float(request.form.get("max_seconds", "60"))
            if not math.isfinite(max_seconds) or max_seconds < 0:
                raise ValueError
        except ValueError:
            return jsonify(error="Thời lượng phải là số hữu hạn không âm."), 400
        if not uploads and not url:
            return jsonify(error="Hãy chọn video hoặc nhập URL."), 400
        with lock:
            if state["status"] in {"running", "stopping"}:
                return jsonify(error="Một video đang được xử lý."), 409
            stop_requested.clear()
            state.update(status="running", frame=0, total=0, error="", revision=0)
            state.update(diagnostic={}, source_index=0, source_count=len(uploads) + bool(url))
            state["result_counts"] = {"confirmed": 0, "low": 0}
            state["filter_stats"] = {}
            preview["data"] = None
        temporary = None
        try:
            sources = []
            if uploads:
                temporary = TemporaryDirectory()
                for index, uploaded in enumerate(uploads, 1):
                    suffix = Path(uploaded.filename).suffix.lower()
                    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
                        raise ValueError("Định dạng video không được hỗ trợ.")
                    source = Path(temporary.name) / f"video{index}{suffix}"
                    uploaded.save(source)
                    sources.append((f"{index} · {uploaded.filename}", source, False))
            if url:
                sources.append((f"{len(sources) + 1} · {mask_url(url)}", url, True))
            update(camera=sources[0][0])
            Thread(
                target=run, args=(sources, options, max_seconds, temporary), daemon=True,
            ).start()
        except Exception as error:
            if temporary is not None:
                temporary.cleanup()
            update(status="error", error=str(error))
            return jsonify(error=str(error)), 400
        return jsonify(status="running"), 202

    return app
