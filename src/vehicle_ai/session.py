"""One source owns one tracker and one event session.

feed/finalize can later be driven by a trigger without changing recognition.
Timestamps are relative source seconds, never wall-clock UTC.
"""
import math
from collections import Counter
from time import perf_counter

from ultralytics.engine.results import Results

from .config import PipelineConfig
from .domain import event_from_row
from .aggregation import Vehicle
from .inference import add_vehicle_sample, apply_plate_read, crop, recognize_vehicle
from .ocr import inspect_plates
from .roi import center_in_roi, crop_detector_roi
from .consensus import classification_decisions


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
        self.finished_rejections = Counter()
        self.finished_rejection_detail = {}
        self.roi_rejected = {}
        self.thresholds = {
            "detector_confidence": config.detector_confidence,
            "detector_imgsz": config.detector_imgsz,
            "min_vehicle_width": config.min_vehicle_width,
            "plate_imgsz": config.plate_imgsz,
            "plate_lower_fraction": config.plate_lower_fraction,
            "plate_full_scan_interval": config.plate_full_scan_interval,
            "min_plate_width": config.min_plate_width,
            "min_plate_sharpness": config.min_plate_sharpness,
            "min_image_quality": config.min_image_quality,
            "min_evidence_detection": config.min_evidence_detection,
            "min_evidence_ocr": config.min_evidence_ocr,
            "max_recognitions_per_frame": config.max_recognitions_per_frame,
            "plate_confirm_reads": config.plate_confirm_reads,
            "plate_confirm_score": config.plate_confirm_score,
            "plate_confirm_mean": config.plate_confirm_mean,
            "plate_confirm_margin": config.plate_confirm_margin,
        }
        self.diagnostics = {
            "samples": 0, "plate_detections": 0, "accepted_reads": 0,
            "status": "Đang chờ xe để nhận diện", "image": None, "text": "", "score": 0.0,
            "rejections": {}, "rejection_detail": {}, "roi_rejected_tracks": 0,
            "thresholds": dict(self.thresholds), "frame_size": "", "detector_boxes": 0,
            "roi_rejected_this_frame": 0, "vehicle_too_small_this_frame": 0,
            "recognition_budget_skipped_this_frame": 0,
        }

    def _refresh_rejections(self):
        counts = Counter(self.finished_rejections)
        details = dict(self.finished_rejection_detail)
        for vehicle in self.tracks.values():
            counts.update(vehicle.rejections)
            details.update(vehicle.rejection_detail)
        # Tracks rejected before entering self.tracks have no Vehicle.reject call.
        for track_id, occurrences in self.roi_rejected.items():
            if track_id not in self.tracks:
                counts["roi_rejected"] += occurrences
        self.diagnostics["rejections"] = dict(counts.most_common())
        self.diagnostics["rejection_detail"] = details
        self.diagnostics["roi_rejected_tracks"] = len(self.roi_rejected)

    def _finish(self, track_id, reason):
        vehicle = self.tracks.pop(track_id)
        self.finished_rejections.update(vehicle.rejections)
        self.finished_rejection_detail.update(vehicle.rejection_detail)
        row = vehicle.row(
            self.camera, track_id, min_image_quality=self.config.min_image_quality,
            min_detection=self.config.min_evidence_detection, min_ocr=self.config.min_evidence_ocr,
            top_k=self.config.evidence_top_k, keep_low_confidence=self.config.keep_low_confidence,
        )
        row["Kết thúc (giây)"] = round(self.seen_at.pop(track_id), 3)
        self.sampled_at.pop(track_id, None)
        row["status"] = "recognized" if row["Biển số"] else (
            "unread" if vehicle.plate_samples else "no_plate"
        )
        row["finalization_reason"] = reason
        row["event"] = event_from_row(
            row, source_id=self.source_id, model_version=self.config.model_version, reason=reason,
        )
        row["raw_event_ids"] = [row["event"]["event_id"]]
        return row

    def _confirm_vehicle(self, vehicle):
        if not vehicle.plates:
            return
        best = max(vehicle.plates, key=vehicle.plate_rank)
        image_ready = any(
            sample.get("recognized_plate") == best
            and sample.get("quality_score", 0) >= self.config.min_image_quality
            and sample.get("detection_score", 0) >= self.config.min_evidence_detection
            and sample.get("ocr_score", 0) >= self.config.min_evidence_ocr
            for sample in vehicle.plate_samples
        )
        vehicle.confirm_plate(self.config.plate_confirm_reads, self.config.plate_confirm_score,
                              self.config.plate_confirm_mean, self.config.plate_confirm_margin,
                              self.config.plate_confirm_share, image_ready=image_ready)

    def feed(self, frame, timestamp):
        if self.closed:
            raise RuntimeError("Phiên đã finalize; hãy tạo phiên mới.")
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp < self.timestamp:
            raise ValueError("Timestamp phải hữu hạn, không âm và không giảm.")
        self.timestamp = timestamp
        self.diagnostics["frame_size"] = f"{frame.shape[1]}x{frame.shape[0]}"
        completed = [
            self._finish(track_id, "track_timeout")
            for track_id in list(self.tracks)
            if timestamp - self.seen_at[track_id] > self.config.track_timeout_seconds
        ]
        started = perf_counter()
        detector_frame, (offset_x, offset_y) = (
            crop_detector_roi(frame, self.config.roi, self.config.vehicle_roi_padding)
            if self.config.crop_vehicle_roi else (frame, (0, 0))
        )
        result = self.detector.track(
            detector_frame, persist=True, tracker="bytetrack.yaml",
            classes=list(self.config.vehicle_classes),
            conf=self.config.detector_confidence, device=self.config.device, verbose=False,
            imgsz=self.config.detector_imgsz, half=self.config.detector_half,
        )[0]
        if detector_frame is not frame:
            data = result.boxes.data.clone() if result.boxes is not None else None
            if data is not None:
                data[:, [0, 2]] += offset_x
                data[:, [1, 3]] += offset_y
            result = Results(frame, path=result.path, names=result.names, boxes=data)
        vehicle_ms = (perf_counter() - started) * 1000
        sample_started = perf_counter()
        self.last_result = result
        boxes = result.boxes
        candidates = []
        pending_ocr = []
        sampled_tracks = {}
        frame_counts = {
            "detector_boxes": 0, "roi_rejected_this_frame": 0,
            "vehicle_too_small_this_frame": 0, "recognition_budget_skipped_this_frame": 0,
        }
        if boxes is not None and boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id.int().tolist()):
                class_id = int(box.cls.item())
                if class_id not in self.config.vehicle_classes:
                    continue
                frame_counts["detector_boxes"] += 1
                coords = box.xyxy[0].tolist()
                if not center_in_roi(coords, frame.shape, self.config.roi):
                    frame_counts["roi_rejected_this_frame"] += 1
                    if track_id in self.tracks:
                        self.tracks[track_id].reject("roi_rejected")
                    else:
                        self.roi_rejected[track_id] = self.roi_rejected.get(track_id, 0) + 1
                    continue
                self.roi_rejected.pop(track_id, None)
                vehicle = self.tracks.setdefault(track_id, Vehicle(timestamp, candidate_limit=self.config.evidence_top_k))
                self.seen_at[track_id] = timestamp
                vehicle.last_seen = self.frame_idx
                kind = self.config.vehicle_classes[class_id]
                image = crop(frame, coords)
                if not image.size:
                    vehicle.reject("empty_vehicle_crop")
                    continue
                area = float(image.shape[0] * image.shape[1])
                if image.shape[1] < self.config.min_vehicle_width:
                    frame_counts["vehicle_too_small_this_frame"] += 1
                    vehicle.reject("vehicle_too_small",
                                   f"{image.shape[1]}px < {self.config.min_vehicle_width}px")
                    if not vehicle.scores:
                        vehicle.scores[kind] = float(box.conf.item()) * .1
                    if self.config.keep_low_confidence and not vehicle.detected_samples:
                        add_vehicle_sample(vehicle, image, float(box.conf.item()), area,
                                           float(frame.shape[0] * frame.shape[1]), self.frame_idx)
                    continue
                # Weight class votes toward larger, closer observations.
                weight = min(image.shape[1] / max(self.config.min_vehicle_width * 2, 1), 1.0)
                vehicle.add_type(kind, float(box.conf.item()), weight)
                vehicle.classification_status = classification_decisions(vehicle.vote_summaries(), self.config)
                if self.config.keep_low_confidence and not vehicle.detected_samples:
                    add_vehicle_sample(vehicle, image, float(box.conf.item()), area,
                                       float(frame.shape[0] * frame.shape[1]), self.frame_idx)
                if timestamp - self.sampled_at.get(track_id, -math.inf) < self.config.sample_interval_seconds:
                    continue
                if vehicle.confirmed_plate and (
                    self.logo_model is None or vehicle.classification_status.get("brand", {}).get("confirmed")
                ):
                    continue
                candidates.append((self.sampled_at.get(track_id, -math.inf), -area,
                                   track_id, box, image, vehicle))
            # Oldest samples first to prevent starvation; larger crops win ties.
            candidates.sort(key=lambda item: item[:3])
            selected = candidates[:self.config.max_recognitions_per_frame]
            frame_counts["recognition_budget_skipped_this_frame"] = len(candidates) - len(selected)
            for _, negative_area, track_id, box, image, vehicle in selected:
                self.sampled_at[track_id] = timestamp
                sample = recognize_vehicle(
                    vehicle, image, self.logo_model, self.plate_model, self.recognizer,
                    frame_idx=self.frame_idx, vehicle_conf=float(box.conf.item()),
                    bbox_area=-negative_area,
                    frame_area=float(frame.shape[0] * frame.shape[1]),
                    min_plate_width=self.config.min_plate_width,
                    min_plate_sharpness=self.config.min_plate_sharpness,
                    plate_imgsz=self.config.plate_imgsz,
                    plate_lower_fraction=self.config.plate_lower_fraction,
                    plate_reuse_padding=self.config.plate_reuse_padding,
                    plate_full_scan_interval=self.config.plate_full_scan_interval,
                    ocr_top_k=self.config.ocr_top_k, pending_ocr=pending_ocr,
                    quality_options={
                        "target_width": self.config.quality_target_width,
                        "target_height": self.config.quality_target_height,
                        "target_sharpness": self.config.quality_target_sharpness,
                        "edge_margin": self.config.quality_edge_margin,
                        "expected_ratios": self.config.quality_expected_ratios,
                    },
                    skip_plate=bool(vehicle.confirmed_plate),
                )
                vehicle.classification_status = classification_decisions(vehicle.vote_summaries(), self.config)
                sampled_tracks[id(vehicle)] = track_id
                self._confirm_vehicle(vehicle)
                self.diagnostics.update(sample, track_id=track_id)
                self.diagnostics["samples"] += 1
                if vehicle.plate_box is not None:
                    self.diagnostics["plate_detections"] += 1
                if sample["status"] == "Đọc được biển":
                    self.diagnostics["accepted_reads"] += 1
        plate_ms = (perf_counter() - sample_started) * 1000
        ocr_started = perf_counter()
        reads = inspect_plates(self.recognizer, [image for _, image, _ in pending_ocr])
        for (vehicle, _, diagnostic), ocr in zip(pending_ocr, reads):
            apply_plate_read(vehicle, diagnostic, ocr)
            self._confirm_vehicle(vehicle)
            self.diagnostics.update(diagnostic, track_id=sampled_tracks[id(vehicle)])
            if ocr["accepted"]:
                self.diagnostics["accepted_reads"] += 1
        self.diagnostics["timings_ms"] = {
            "vehicle": round(vehicle_ms, 2), "plate_and_crops": round(plate_ms, 2),
            "ocr": round((perf_counter() - ocr_started) * 1000, 2),
            "total": round((perf_counter() - started) * 1000, 2),
        }
        self.diagnostics["ocr_batch"] = len(pending_ocr)
        self.diagnostics["active_tracks"] = len(self.tracks)
        self.diagnostics["confirmed_tracks"] = sum(bool(v.confirmed_plate) for v in self.tracks.values())
        self.diagnostics.update(frame_counts)
        if not frame_counts["detector_boxes"]:
            self.diagnostics["status"] = "Không thấy xe trong ROI/khung hình (kiểm tra conf detector, ROI, độ phân giải)"
        self._refresh_rejections()
        self.frame_idx += 1
        return completed

    def finalize(self, reason="end_of_source"):
        if self.closed:
            return []
        self.closed = True
        rows = [self._finish(track_id, reason) for track_id in list(self.tracks)]
        self._refresh_rejections()
        return rows
