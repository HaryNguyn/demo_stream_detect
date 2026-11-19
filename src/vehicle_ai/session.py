"""One source owns one tracker and one event session.

feed/finalize can later be driven by a trigger without changing recognition.
Timestamps are relative source seconds, never wall-clock UTC.
"""
import math

from .config import PipelineConfig
from .domain import event_from_row
from .aggregation import Vehicle
from .inference import crop, recognize_vehicle
from .roi import center_in_roi


class RecognitionSession:
    def __init__(self, detector, config: PipelineConfig, camera, *, source_id=None,
                 logo_model=None, plate_model=None, recognizer=None):
        self.detector, self.config, self.camera = detector, config, camera
        self.source_id = source_id or camera
        self.logo_model, self.plate_model, self.recognizer = logo_model, plate_model, recognizer
        self.tracks = {}
        self.seen_at = {}
        self.sampled_at = {}
        self.frame_idx = 0
        self.timestamp = -1.0
        self.closed = False
        self.last_result = None
        self.diagnostics = {
            "samples": 0, "plate_detections": 0, "accepted_reads": 0,
            "status": "Đang chờ xe để nhận diện", "image": None, "text": "", "score": 0.0,
        }

    def _finish(self, track_id, reason):
        vehicle = self.tracks.pop(track_id)
        row = vehicle.row(self.camera, track_id)
        row["Kết thúc (giây)"] = round(self.seen_at.pop(track_id), 3)
        self.sampled_at.pop(track_id, None)
        row["status"] = "recognized" if row["Biển số"] else (
            "unread" if vehicle.plate_samples else "no_plate"
        )
        row["finalization_reason"] = reason
        row["event"] = event_from_row(
            row, source_id=self.source_id, model_version=self.config.model_version, reason=reason,
        )
        return row

    def feed(self, frame, timestamp):
        if self.closed:
            raise RuntimeError("Phiên đã finalize; hãy tạo phiên mới.")
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp < self.timestamp:
            raise ValueError("Timestamp phải hữu hạn, không âm và không giảm.")
        self.timestamp = timestamp
        completed = [
            self._finish(track_id, "track_timeout")
            for track_id in list(self.tracks)
            if timestamp - self.seen_at[track_id] > self.config.track_timeout_seconds
        ]
        result = self.detector.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=list(self.config.vehicle_classes),
            conf=self.config.detector_confidence, device=self.config.device, verbose=False,
        )[0]
        self.last_result = result
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id.int().tolist()):
                class_id = int(box.cls.item())
                if class_id not in self.config.vehicle_classes:
                    continue
                if not center_in_roi(box.xyxy[0].tolist(), frame.shape, self.config.roi):
                    continue
                vehicle = self.tracks.setdefault(track_id, Vehicle(timestamp))
                self.seen_at[track_id] = timestamp
                vehicle.last_seen = self.frame_idx
                kind = self.config.vehicle_classes[class_id]
                vehicle.scores[kind] = vehicle.scores.get(kind, 0.0) + float(box.conf.item())
                if timestamp - self.sampled_at.get(track_id, -math.inf) < self.config.sample_interval_seconds:
                    continue
                self.sampled_at[track_id] = timestamp
                box_xyxy = box.xyxy[0].tolist()
                sample = recognize_vehicle(
                    vehicle, crop(frame, box_xyxy), self.logo_model, self.plate_model, self.recognizer,
                    frame_idx=self.frame_idx, vehicle_conf=float(box.conf.item()),
                    bbox_area=max(0.0, (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])),
                    frame_area=float(frame.shape[0] * frame.shape[1]),
                )
                self.diagnostics.update(sample, track_id=track_id)
                self.diagnostics["samples"] += 1
                if sample["status"] in {"Đọc được biển", "Có biển nhưng OCR chưa đạt ngưỡng"}:
                    self.diagnostics["plate_detections"] += 1
                if sample["status"] == "Đọc được biển":
                    self.diagnostics["accepted_reads"] += 1
        self.frame_idx += 1
        return completed

    def finalize(self, reason="end_of_source"):
        if self.closed:
            return []
        self.closed = True
        return [self._finish(track_id, reason) for track_id in list(self.tracks)]
