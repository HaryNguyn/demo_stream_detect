"""Vehicle tracking, recognition sampling and event finalization."""
import base64
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from recognition import inspect_plate
from sources import open_capture

VEHICLES = {2: "Ô tô con", 3: "Xe máy", 5: "Xe buýt", 7: "Xe tải"}
MAX_PLATE_SAMPLES = 30
MAX_PLATE_CANDIDATES = 20
TRACK_TIMEOUT_SECONDS = 2
PREVIEW_EVERY_FRAMES = 10
TOP_SAMPLES = 3
MIN_SAMPLE_FRAME_DISTANCE = 5


def encode_crop(image):
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return None
    height, width = image.shape[:2]
    return {
        "mime_type": "image/jpeg",
        "encoding": "base64",
        "width": int(width),
        "height": int(height),
        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def image_quality_score(image, confidence, bbox_area, frame_area):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_norm = float(np.clip(sharpness / 500.0, 0.0, 1.0))
    size_ratio = float(np.clip(bbox_area / max(frame_area, 1), 0.0, 1.0))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return 0.4 * confidence + 0.3 * size_ratio + 0.3 * sharpness_norm


def select_top_samples(samples):
    selected = []
    for candidate in sorted(samples, key=lambda item: item["_score"], reverse=True):
        frame_idx = candidate["_frame_idx"]
        if all(abs(frame_idx - item["_frame_idx"]) >= MIN_SAMPLE_FRAME_DISTANCE for item in selected):
            selected.append(candidate)
        if len(selected) == TOP_SAMPLES:
            break
    if len(selected) < TOP_SAMPLES:
        selected_ids = {id(item) for item in selected}
        for candidate in sorted(samples, key=lambda item: item["_score"], reverse=True):
            if id(candidate) not in selected_ids:
                selected.append(candidate)
                selected_ids.add(id(candidate))
            if len(selected) == TOP_SAMPLES:
                break
    return sorted(selected, key=lambda item: item["_frame_idx"])


@dataclass
class Vehicle:
    time: float
    scores: dict[str, float] = field(default_factory=dict)
    brand: str = ""
    brand_score: float = 0.0
    plates: dict[str, list[float]] = field(default_factory=dict)
    detected_samples: list[dict] = field(default_factory=list)
    plate_samples: list[dict] = field(default_factory=list)
    last_sample: int = -10000
    last_seen: int = 0

    def plate_rank(self, plate):
        scores = self.plates[plate]
        return len(scores), max(scores)

    def add_plate(self, value, score):
        scores = self.plates.setdefault(value, [])
        scores.append(score)
        self.plates[value] = scores[-MAX_PLATE_SAMPLES:]
        if len(self.plates) > MAX_PLATE_CANDIDATES:
            del self.plates[min(self.plates, key=self.plate_rank)]

    def add_sample(self, kind, candidate):
        samples = self.detected_samples if kind == "detected" else self.plate_samples
        samples.append(candidate)

    def row(self, camera, track_id):
        plate = max(self.plates, key=self.plate_rank) if self.plates else ""
        plate_score = max(self.plates[plate]) if plate else 0.0
        detected_samples = select_top_samples(self.detected_samples)
        plate_samples = select_top_samples(self.plate_samples)
        return {
            "Camera": camera,
            "Track ID": track_id,
            "Xe ID": plate if plate else f"TRACK-{track_id}",
            "Xuất hiện (giây)": round(self.time, 2),
            "Loại xe": max(self.scores, key=self.scores.get) if self.scores else "",
            "Hãng": self.brand,
            "Điểm hãng": round(self.brand_score, 4),
            "Biển số": plate,
            "Điểm biển": round(plate_score, 4),
            "Ảnh xe": detected_samples[-1] if detected_samples else None,
            "Ảnh biển số": plate_samples[-1] if plate_samples else None,
            "Danh sách ảnh xe": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in detected_samples
            ],
            "Danh sách ảnh biển": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in plate_samples
            ],
        }


def crop(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    return image[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)]


def best_box(model, image, threshold):
    boxes = model.predict(image, conf=threshold, verbose=False)[0].boxes
    return max(boxes, key=lambda box: float(box.conf.item()), default=None)


def recognize_vehicle(
    vehicle, image, logo_model, plate_model, recognizer,
    frame_idx=0, vehicle_conf=0.0, bbox_area=0.0, frame_area=1.0,
):
    diagnostic = {"status": "Không có ảnh xe", "image": None, "text": "", "score": 0.0}
    if not image.size:
        return diagnostic
    detected = encode_crop(image)
    if detected is not None:
        score = image_quality_score(image, vehicle_conf, bbox_area, frame_area)
        detected["_frame_idx"] = frame_idx
        detected["_score"] = round(score, 6)
        detected["frame_idx"] = frame_idx
        detected["quality_score"] = round(score, 6)
        vehicle.detected_samples.append(detected)
    diagnostic.update(status="Chưa bật model biển/OCR", image=image)
    if logo_model is not None:
        logo = best_box(logo_model, image, 0.4)
        if logo is not None and logo.conf.item() > vehicle.brand_score:
            vehicle.brand = logo_model.names[int(logo.cls.item())]
            vehicle.brand_score = float(logo.conf.item())
    if plate_model is None or recognizer is None:
        return diagnostic
    plate = best_box(plate_model, image, 0.35)
    if plate is None:
        diagnostic["status"] = "Không phát hiện biển trong ảnh xe"
        return diagnostic
    plate_image = crop(image, plate.xyxy[0].tolist())
    if plate_image.size:
        plate_sample = encode_crop(plate_image)
        if plate_sample is not None:
            plate_conf = float(plate.conf.item())
            plate_area = float((plate.xyxy[0][2] - plate.xyxy[0][0]) * (plate.xyxy[0][3] - plate.xyxy[0][1]))
            score = image_quality_score(
                plate_image,
                plate_conf,
                plate_area,
                float(image.shape[0] * image.shape[1]),
            )
            plate_sample["_frame_idx"] = frame_idx
            plate_sample["_score"] = round(score, 6)
            plate_sample["frame_idx"] = frame_idx
            plate_sample["quality_score"] = round(score, 6)
            vehicle.plate_samples.append(plate_sample)
        ocr = inspect_plate(recognizer, plate_image)
        diagnostic.update(
            image=plate_image, text=ocr["text"], score=ocr["score"],
            status="Đọc được biển" if ocr["accepted"] else "Có biển nhưng OCR chưa đạt ngưỡng",
        )
        if ocr["accepted"]:
            vehicle.add_plate(ocr["text"], ocr["score"])
    else:
        diagnostic["status"] = "Vùng biển rỗng"
    return diagnostic


def draw_recognition(result, tracks):
    image = result.plot()
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return image
    for box, track_id in zip(boxes, boxes.id.int().tolist()):
        vehicle = tracks.get(track_id)
        if vehicle is None:
            continue
        plate = max(vehicle.plates, key=vehicle.plate_rank) if vehicle.plates else "OCR: --"
        label = f"{plate} {vehicle.brand}".strip()
        x1, _, _, y2 = map(int, box.xyxy[0].tolist())
        cv2.putText(
            image, label, (max(0, x1), max(20, min(image.shape[0] - 8, y2))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
        )
    return image


def analyze_video(
    path, camera, vehicle_weights, logo_model=None, plate_model=None,
    recognizer=None, on_frame=None, on_event=None, max_seconds=0,
    collect_results=True, on_diagnostic=None,
):
    # Each source owns a detector so ByteTrack state cannot cross video boundaries.
    detector = YOLO(vehicle_weights)
    tracks = {}
    vehicle_identity_by_plate = {}
    completed = []
    diagnostics = {
        "samples": 0, "plate_detections": 0, "accepted_reads": 0,
        "status": "Đang chờ xe để nhận diện", "image": None, "text": "", "score": 0.0,
    }
    capture = open_capture(path)
    try:
        if not capture.isOpened():
            source = str(path)
            detail = ""
            if Path(source).is_file():
                detail = f" Tệp: {Path(source).name} ({Path(source).stat().st_size / 1024 / 1024:.1f} MB)."
            raise ValueError(
                "Không mở được video." + detail +
                " Kiểm tra tệp không hỏng, codec được hỗ trợ, hoặc thử chuyển sang MP4 (H.264)."
            )
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every = max(1, round(fps / 2))
        recognition_enabled = logo_model is not None or (
            plate_model is not None and recognizer is not None
        )

        def finish(track_id):
            vehicle = tracks.pop(track_id)
            row = vehicle.row(camera, track_id)
            row["Kết thúc (giây)"] = round(vehicle.last_seen / fps, 2)
            if on_event:
                on_event(row)
            if collect_results:
                completed.append(row)

        frame_idx = 0
        while not max_seconds or frame_idx < max_seconds * fps:
            ok, frame = capture.read()
            if not ok:
                break
            result = detector.track(
                frame, persist=True, tracker="bytetrack.yaml", classes=list(VEHICLES),
                conf=0.2, verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for box, track_id in zip(boxes, boxes.id.int().tolist()):
                    vehicle = tracks.setdefault(track_id, Vehicle(frame_idx / fps))
                    vehicle.last_seen = frame_idx
                    kind = VEHICLES[int(box.cls.item())]
                    vehicle.scores[kind] = vehicle.scores.get(kind, 0.0) + float(box.conf.item())
                    if not recognition_enabled or frame_idx - vehicle.last_sample < sample_every:
                        continue
                    vehicle.last_sample = frame_idx
                    xyxy = box.xyxy[0].tolist()
                    bbox_area = max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                    vehicle_conf = float(box.conf.item())
                    sample = recognize_vehicle(
                        vehicle, crop(frame, xyxy),
                        logo_model, plate_model, recognizer,
                        frame_idx=frame_idx,
                        vehicle_conf=vehicle_conf,
                        bbox_area=bbox_area,
                        frame_area=float(frame.shape[0] * frame.shape[1]),
                    )
                    diagnostics.update(sample, track_id=track_id)
                    diagnostics["samples"] += 1
                    if sample["status"] in {"Đọc được biển", "Có biển nhưng OCR chưa đạt ngưỡng"}:
                        diagnostics["plate_detections"] += 1
                    if sample["status"] == "Đọc được biển":
                        diagnostics["accepted_reads"] += 1
            for track_id in list(tracks):
                if frame_idx - tracks[track_id].last_seen > round(fps * TRACK_TIMEOUT_SECONDS):
                    finish(track_id)
            if on_frame and frame_idx % PREVIEW_EVERY_FRAMES == 0:
                on_frame(frame_idx + 1, total, draw_recognition(result, tracks))
            if on_diagnostic and frame_idx % PREVIEW_EVERY_FRAMES == 0:
                on_diagnostic(diagnostics.copy())
            frame_idx += 1
        if frame_idx == 0:
            source = str(path)
            detail = f" Nguồn: {source}" if not Path(source).is_file() else (
                f" Tệp: {Path(source).name} ({Path(source).stat().st_size / 1024 / 1024:.1f} MB)."
            )
            capture_error = capture.error_text() if hasattr(capture, "error_text") else ""
            if capture_error:
                detail += f"\nChi tiết nguồn:\n{capture_error}"
            raise ValueError(
                "Video không có khung hình đọc được." + detail +
                " Nguyên nhân thường gặp: tệp hỏng, codec không được hỗ trợ, "
                "link stream đã ngắt, hoặc FFmpeg không lấy được luồng."
                " Với tệp tải lên, thử đổi sang MP4/H.264 bằng FFmpeg."
            )
        for track_id in list(tracks):
            finish(track_id)
        if on_diagnostic:
            on_diagnostic(diagnostics.copy())
    finally:
        capture.release()
    return completed


def optional_yolo(path):
    path = path.strip()
    if not path:
        return None
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return YOLO(path)
