"""Still-image plate recognition backend, isolated from the video session.

This module keeps its own model cache and inference lock. It never touches the
video state, preview frames, stop event or event store owned by ``web.py``.
"""
import base64
import json
from threading import Lock
from time import perf_counter

import cv2
import numpy as np

from .inference import _ascii, _crisp_box, _panel, best_box, crop, encode_crop, optional_yolo
from .ocr import inspect_plates
from .quality import plate_quality
from .web_models import prepare_models

MAX_IMAGE_SIDE = 3000

DEFAULT_OPTIONS = {
    "mode": "manual",
    "config": "configs/vietnam-ocr.yaml",
    "vehicle_weights": "models/yolo11n.pt",
    "logo_weights": "models/logo.pt",
    "plate_weights": "models/best_stripped.onnx",
    "ocr_weights": "models/ocr_model.onnx",
    "ocr_config": "models/ocr_model.yaml",
    "detector_imgsz": "1280",
    "plate_imgsz": "960",
    "ocr_threads": "4",
    "ocr_batch_size": "8",
    "vehicle_conf": "0.15",
    "plate_conf": "0.25",
    "min_vehicle_width": "40",
    "speed_preset": "custom",
}

COLORS = [
    (255, 190, 60), (90, 220, 90), (70, 170, 255), (255, 120, 240),
    (80, 225, 250), (255, 190, 120), (150, 250, 190), (215, 165, 255),
]

_models = {}
_cache_lock = Lock()
_inference_lock = Lock()


def normalize_options(options):
    """Validate form values before they reach the shared model preparation."""
    values = dict(DEFAULT_OPTIONS)
    for key in values:
        value = options.get(key)
        if value is not None and str(value).strip():
            values[key] = str(value).strip()
    if values["mode"] not in {"manual", "yaml"}:
        raise ValueError("Chế độ cấu hình không hợp lệ.")
    if values["speed_preset"] not in {"custom", "fast", "balanced", "accurate"}:
        raise ValueError("Tốc độ xử lý không hợp lệ.")
    for key in ("detector_imgsz", "plate_imgsz", "ocr_threads", "ocr_batch_size"):
        try:
            number = int(values[key])
        except ValueError as error:
            raise ValueError(f"{key} phải là số nguyên.") from error
        if number < 1:
            raise ValueError(f"{key} phải lớn hơn 0.")
        values[key] = str(number)
    for key in ("vehicle_conf", "plate_conf"):
        try:
            number = float(values[key])
        except ValueError as error:
            raise ValueError(f"{key} phải là số.") from error
        if not 0 <= number <= 1:
            raise ValueError(f"{key} phải nằm trong [0, 1].")
        values[key] = str(number)
    try:
        minimum = int(values["min_vehicle_width"])
    except ValueError as error:
        raise ValueError("min_vehicle_width phải là số nguyên.") from error
    if minimum < 0:
        raise ValueError("min_vehicle_width không được âm.")
    values["min_vehicle_width"] = str(minimum)
    return values


def decode_image(payload):
    """Decode an uploaded image and cap very large inputs for safe inference."""
    array = np.frombuffer(payload, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không đọc được ảnh. Dùng file JPG/PNG/BMP/WebP hợp lệ.")
    height, width = image.shape[:2]
    if max(height, width) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(height, width)
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    return image


def get_models(options):
    """Load models once per option set; the lock avoids duplicate model loads."""
    key = json.dumps(options, sort_keys=True)
    with _cache_lock:
        cached = _models.get(key)
        if cached is None:
            config, logo, plate, recognizer = prepare_models(options)
            detector = optional_yolo(config.vehicle_weights)
            if detector is None:
                raise ValueError("Thiếu model phát hiện xe (vehicle_weights).")
            cached = {
                "config": config, "detector": detector, "logo": logo,
                "plate": plate, "recognizer": recognizer,
            }
            _models[key] = cached
        return cached


def find_plate(model, image, imgsz, confidence):
    """Scan the whole vehicle crop; the video pipeline's locator stays untouched."""
    if model is None or not image.size:
        return None
    box = best_box(model, image, confidence, imgsz=imgsz)
    if box is None:
        return None
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return (x1, y1, x2, y2), float(box.conf.item())


def find_plate_boxes(model, image, imgsz, confidence):
    """Return every plate box on a whole image (used only by the still-image page)."""
    if model is None or not image.size:
        return []
    result = model.predict(image, conf=confidence, imgsz=imgsz, verbose=False)[0]
    return [([float(value) for value in box.xyxy[0].tolist()], float(box.conf.item()))
            for box in (getattr(result, "boxes", None) or [])]


def _ocr_plate(recognizer, plate_image):
    """Read a plate with a few safe crop variants; best accepted text wins."""
    if recognizer is None or not plate_image.size:
        return {"text": "", "score": 0.0, "accepted": False}
    variants = [plate_image]
    for padding in (4, 16):
        variants.append(cv2.copyMakeBorder(
            plate_image, padding, padding, padding, padding, cv2.BORDER_REPLICATE,
        ))
    for scale in (1.25, 2.0):
        variants.append(cv2.resize(
            plate_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
        ))
    gray = (cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            if plate_image.ndim == 3 else plate_image)
    for threshold in (160, 180):
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    results = inspect_plates(recognizer, variants)
    accepted = [item for item in results if item["accepted"]]
    if not accepted:
        return max(results, key=lambda item: float(item["score"]))
    # Prefer text supported by several similarly confident crops; a single
    # high-scoring threshold variant should not override the consensus.
    best_score = max(float(item["score"]) for item in accepted)

    def rank(text):
        matches = [item for item in accepted if item["text"] == text]
        return (
            sum(float(item["score"]) >= best_score - 0.05 for item in matches),
            max(float(item["score"]) for item in matches),
            sum(float(item["score"]) for item in matches),
        )

    best_text = max({item["text"] for item in accepted}, key=rank)
    return max((item for item in accepted if item["text"] == best_text),
               key=lambda item: float(item["score"]))


def _iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(area_first + area_second - intersection, 1e-6)


def _clamp_bbox(bbox, shape):
    height, width = shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = min(width, max(x2, x1 + 1)), min(height, max(y2, y1 + 1))
    return x1, y1, x2, y2


def _vehicle_boxes(result, names):
    boxes = []
    for box in (getattr(result, "boxes", None) or []):
        boxes.append({
            "class_id": int(box.cls.item()),
            "name": names.get(int(box.cls.item()), str(int(box.cls.item()))),
            "confidence": float(box.conf.item()),
            "bbox": [float(value) for value in box.xyxy[0].tolist()],
        })
    return boxes


def _full_image_plates(plate_model, image, options):
    """Multi-scale whole-image plate scan; the video pipeline keeps its own logic."""
    found = []
    minimum_confidence = max(float(options["plate_conf"]), 0.45)
    for scale in (1.0, 0.75, 0.55):
        resized = (cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                   if scale != 1.0 else image)
        for box, confidence in find_plate_boxes(
            plate_model, resized, int(options["plate_imgsz"]), minimum_confidence,
        ):
            mapped = _clamp_bbox([value / scale for value in box], image.shape)
            if mapped[2] - mapped[0] < 50 or mapped[3] - mapped[1] < 20:
                continue  # Drop tiny false positives on full-image scans.
            found.append((mapped, confidence))
    found.sort(key=lambda item: item[1], reverse=True)
    unique = []
    for box, confidence in found:
        if any(_iou(box, kept[0]) > 0.6 for kept in unique):
            continue
        unique.append((box, confidence))
    return unique


def _vehicle_from_plate(detector, image, plate_box, config, options):
    """Infer the vehicle region around a plate when the detector missed the vehicle."""
    px1, py1, px2, py2 = plate_box
    plate_width, plate_height = max(1.0, px2 - px1), max(1.0, py2 - py1)
    attempts = [
        (px1 - plate_width * 2.5, py1 - plate_height * 11.0, px2 + plate_width * 2.5,
         py2 + plate_height * 3.0),
        (px1 - plate_width * 4.0, py1 - plate_height * 7.0, px2 + plate_width * 4.0,
         py2 + plate_height * 3.0),
        (0, max(0, py1 - plate_height * 12.0), image.shape[1], py2 + plate_height * 3.0),
    ]
    fallback_box = _clamp_bbox(attempts[0], image.shape)
    for attempt in attempts:
        box = _clamp_bbox(attempt, image.shape)
        region = crop(image, box)
        if region.size == 0:
            continue
        probe = detector.predict(region, conf=0.1,
                                 imgsz=min(int(options["detector_imgsz"]), 960), verbose=False)[0]
        candidates = _vehicle_boxes(probe, getattr(detector, "names", None) or {})
        vehicles = [item for item in candidates if item["class_id"] in config.vehicle_classes]
        riders = [item for item in candidates if item["class_id"] in {0, 1, 3}]
        best_vehicle = max(vehicles, key=lambda item: item["confidence"]) if vehicles else None
        best_rider = max(riders, key=lambda item: item["confidence"]) if riders else None
        if best_rider and (best_vehicle is None
                           or best_rider["confidence"] >= best_vehicle["confidence"] + 0.1):
            label = ("Xe máy" if best_rider["class_id"] == 3 else "Xe máy (suy luận)")
            return list(box), label, best_rider["confidence"]
        if best_vehicle is not None:
            return list(box), config.vehicle_classes[best_vehicle["class_id"]], best_vehicle["confidence"]
    return list(fallback_box), "Xe máy (suy luận)", 0.0


def _label(image, text, left, top, color):
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.7, 2)
    left = max(2, min(int(left), image.shape[1] - text_width - 12))
    top = max(text_height + 8, min(int(top), image.shape[0] - baseline - 4))
    right, bottom = left + text_width + 10, top + 6
    _panel(image, left - 4, top - text_height - 4, right + 2, bottom, 0.68)
    cv2.putText(image, text, (left + 3, top - 2), font, 0.7, (240, 240, 240), 2, cv2.LINE_AA)


def analyze_image(image, models, options):
    """Detect vehicles, classify them, read their plates and draw an overlay."""
    started = perf_counter()
    config = models["config"]
    detector = models["detector"]
    logo_model = models["logo"]
    plate_model = models["plate"]
    recognizer = models["recognizer"]
    height, width = image.shape[:2]
    vehicle_conf = float(options["vehicle_conf"])
    detector_imgsz = int(options["detector_imgsz"])
    plate_imgsz = int(options["plate_imgsz"])
    min_vehicle_width = int(options["min_vehicle_width"])

    overlay = image.copy()
    cv2.putText(overlay, _ascii(f"Vehicle AI | IMAGE ANPR | {width}x{height}"), (10, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (210, 230, 255), 2, cv2.LINE_AA)
    vehicles = []
    seen_texts = set()

    def blank_vehicle(index, vehicle_type, confidence, bbox, fallback=False):
        return {
            "index": index,
            "vehicle_type": vehicle_type,
            "vehicle_confidence": round(float(confidence), 4),
            "bbox": list(bbox),
            "brand": "", "brand_confidence": 0.0,
            "plate_detected": False, "plate_confidence": 0.0,
            "plate_bbox": None, "plate_text": "", "plate_score": 0.0,
            "accepted": False, "status": "Không phát hiện biển trong ảnh xe",
            "plate_image": "", "plate_quality": None,
            "fallback": bool(fallback),
        }

    def read_plate(vehicle, plate_image, plate_conf, relative_box, vehicle_image, ocr_override=None):
        vx1, vy1, _, _ = vehicle["bbox"]
        vehicle.update(
            plate_detected=True,
            plate_confidence=round(float(plate_conf), 4),
            plate_bbox=[vx1 + round(relative_box[0]), vy1 + round(relative_box[1]),
                        vx1 + round(relative_box[2]), vy1 + round(relative_box[3])],
            plate_image="data:image/jpeg;base64," + (encode_crop(plate_image) or {"data": ""})["data"],
            plate_quality=plate_quality(
                plate_image, plate_conf, relative_box, vehicle_image.shape,
                target_width=config.quality_target_width,
                target_height=config.quality_target_height,
                target_sharpness=config.quality_target_sharpness,
                edge_margin=config.quality_edge_margin,
                expected_ratios=tuple(config.quality_expected_ratios),
            ),
        )
        if recognizer is None:
            vehicle["status"] = "Có biển nhưng chưa bật OCR"
            return ""
        ocr = ocr_override or _ocr_plate(recognizer, plate_image)
        vehicle.update(
            plate_text=ocr["text"], plate_score=round(float(ocr["score"]), 4),
            accepted=bool(ocr["accepted"]),
            status="Đọc được biển" if ocr["accepted"] else "OCR chưa đạt ngưỡng",
        )
        return vehicle["plate_text"]

    def add_brand(vehicle, vehicle_image):
        if logo_model is None or not vehicle_image.size:
            return
        logo = best_box(logo_model, vehicle_image, 0.4)
        if logo is not None:
            vehicle["brand"] = str(logo_model.names[int(logo.cls.item())])
            vehicle["brand_confidence"] = round(float(logo.conf.item()), 4)

    def finish(vehicle, vehicle_image, bbox):
        if not vehicle["plate_detected"]:
            return
        if vehicle["plate_bbox"] and any(
            _iou(vehicle["plate_bbox"], item["plate_bbox"]) > 0.3
            for item in vehicles if item["plate_bbox"]
        ):
            return  # Same physical plate, different detector crop.
        text = vehicle["plate_text"]
        if text and text in seen_texts:
            return
        if text:
            seen_texts.add(text)
        vehicles.append(vehicle)
        x1, y1, x2, y2 = bbox
        color = COLORS[(len(vehicles) - 1) % len(COLORS)]
        _crisp_box(overlay, (x1, y1, x2, y2), color, 2)
        caption = " | ".join(part for part in (
            _ascii(vehicle["vehicle_type"]), _ascii(vehicle["brand"]),
            _ascii(vehicle["plate_text"] or "OCR --"),
        ) if part)
        _label(overlay, caption, x1, y1 - 8, color)
        if vehicle["plate_bbox"]:
            plate_color = (60, 220, 120) if vehicle["accepted"] else (90, 180, 255)
            _crisp_box(overlay, vehicle["plate_bbox"], plate_color, 2)

    # Whole-image plate scan first: it is more reliable than vehicle crops for
    # close-up bike photos where the generic detector sees only the rider.
    full_plates = _full_image_plates(plate_model, image, options) if plate_model is not None else []
    plate_groups = []
    for plate_box, plate_conf in full_plates:
        for group in plate_groups:
            if any(_iou(plate_box, item[0]) > 0.5 for item in group):
                group.append((plate_box, plate_conf))
                break
        else:
            plate_groups.append([(plate_box, plate_conf)])

    group_best = []
    for group in plate_groups:
        best = None
        for plate_box, plate_conf in group:
            plate_image = crop(image, plate_box)
            if plate_image.size == 0:
                continue
            ocr = _ocr_plate(recognizer, plate_image)
            rank = (bool(ocr["accepted"]), float(ocr["score"]), float(plate_conf))
            if best is None or rank > best[0]:
                best = (rank, plate_box, plate_conf, plate_image, ocr)
        if best is not None:
            group_best.append(best)
    accepted_groups = [item for item in group_best if item[4]["accepted"]]
    for _, plate_box, plate_conf, plate_image, ocr in (accepted_groups or group_best):
        vehicle_box, vehicle_type, vehicle_confidence = _vehicle_from_plate(
            detector, image, plate_box, config, options,
        )
        vehicle_image = crop(image, vehicle_box)
        if vehicle_image.size == 0:
            continue
        relative_box = (plate_box[0] - vehicle_box[0], plate_box[1] - vehicle_box[1],
                        plate_box[2] - vehicle_box[0], plate_box[3] - vehicle_box[1])
        crop_for_vehicle = crop(vehicle_image, relative_box)
        if crop_for_vehicle.size == 0:
            continue
        vehicle = blank_vehicle(len(vehicles) + 1, vehicle_type, vehicle_confidence,
                                vehicle_box, fallback=True)
        read_plate(vehicle, plate_image, plate_conf, relative_box, vehicle_image, ocr_override=ocr)
        add_brand(vehicle, vehicle_image)
        finish(vehicle, vehicle_image, vehicle_box)

    result = detector.predict(image, conf=vehicle_conf, imgsz=detector_imgsz, verbose=False)[0]
    names = getattr(result, "names", None) or {}

    for box in (getattr(result, "boxes", None) or []):
        class_id = int(box.cls.item())
        person_candidate = (
            class_id not in config.vehicle_classes
            and str(names.get(class_id, "")).strip().lower() == "person"
        )
        if class_id not in config.vehicle_classes and not person_candidate:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
        x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
        x2, y2 = min(width, max(x2, x1 + 1)), min(height, max(y2, y1 + 1))
        detection_box = (x1, y1, x2, y2)
        if person_candidate:
            box_width, box_height = x2 - x1, y2 - y1
            x1, y1 = max(0, int(x1 - box_width * 0.15)), max(0, int(y1 - box_height * 0.1))
            x2, y2 = min(width, int(x2 + box_width * 0.15)), min(height, int(y2 + box_height * 0.5))
        vehicle_box = (x1, y1, x2, y2)
        vehicle_image = crop(image, vehicle_box)
        if vehicle_image.size == 0 or vehicle_image.shape[1] < min_vehicle_width:
            continue
        vehicle = blank_vehicle(
            len(vehicles) + 1, config.vehicle_classes.get(class_id, "Xe máy (suy luận)"),
            float(box.conf.item()), vehicle_box,
        )
        plate = find_plate(plate_model, vehicle_image, plate_imgsz, float(options["plate_conf"]))
        if plate is not None:
            relative_box, plate_conf = plate
            if person_candidate:
                detection_x1, detection_y1, detection_x2, detection_y2 = detection_box
                plate_x = x1 + (relative_box[0] + relative_box[2]) / 2
                plate_y = y1 + relative_box[1]
                if (plate_y < detection_y2 - (detection_y2 - detection_y1) * 0.25
                        or not detection_x1 - (detection_x2 - detection_x1) * 0.15
                        <= plate_x
                        <= detection_x2 + (detection_x2 - detection_x1) * 0.15):
                    plate = None
            if plate is not None:
                relative_box, plate_conf = plate
                plate_image = crop(vehicle_image, relative_box)
                if plate_image.size:
                    read_plate(vehicle, plate_image, plate_conf, relative_box, vehicle_image)
        add_brand(vehicle, vehicle_image)
        if full_plates and not vehicle["accepted"]:
            continue  # Do not duplicate a rejected crop read over a good full-image read.
        finish(vehicle, vehicle_image, vehicle_box)

    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    annotated = base64.b64encode(encoded.tobytes()).decode("ascii") if ok else ""
    return {
        "width": width, "height": height, "vehicles": vehicles, "annotated": annotated,
        "summary": {
            "vehicles": len(vehicles),
            "plates_detected": sum(1 for item in vehicles if item["plate_detected"]),
            "plates_accepted": sum(1 for item in vehicles if item["accepted"]),
        },
        "elapsed_ms": round((perf_counter() - started) * 1000, 1),
    }


def recognize_image(payload, options=None):
    """Public entry point used by the image-only HTTP endpoint."""
    normalized = normalize_options(options or {})
    image = decode_image(payload)
    with _inference_lock:
        models = get_models(normalized)
        return analyze_image(image, models, normalized)
