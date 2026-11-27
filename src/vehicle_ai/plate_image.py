"""Still-image plate recognition backend, isolated from the video session.

This module keeps its own model cache and inference lock. It never touches the
video state, preview frames, stop event or event store owned by ``web.py``.
"""
import base64
import json
import re
from threading import Lock
from time import perf_counter

import cv2
import numpy as np

from .inference import _ascii, _crisp_box, _panel, crop, encode_crop, optional_yolo
from .ocr import inspect_plates
from .quality import plate_quality
from .read_log import timed_stage
from .web_models import prepare_models

MAX_IMAGE_SIDE = 3000

DEFAULT_OPTIONS = {
    "input_kind": "auto",
    "mode": "manual",
    "config": "configs/vietnam-ocr.yaml",
    "vehicle_weights": "models/yolo11n.pt",
    "plate_weights": "models/best_stripped.onnx",
    "ocr_weights": "models/ocr_model.onnx",
    "ocr_config": "models/ocr_model.yaml",
    "detector_imgsz": "640",
    "plate_imgsz": "640",
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
    if values["input_kind"] not in {"auto", "scene", "crop"}:
        raise ValueError("input_kind phải là auto, scene hoặc crop.")
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


def get_models(options, *, ocr_only=False):
    """Load models once per option set; the lock avoids duplicate model loads."""
    key = json.dumps({k: v for k, v in options.items() if k != "input_kind"}, sort_keys=True)
    with _cache_lock:
        cached = _models.get(key)
        if cached is None:
            config, _, plate, recognizer = prepare_models(options, ocr_only=True)
            cached = {
                "config": config,
                "plate": plate, "recognizer": recognizer,
            }
            _models[key] = cached
        if not ocr_only and cached["plate"] is None:
            cached["plate"] = optional_yolo(cached["config"].plate_weights)
            if cached["plate"] is None:
                raise ValueError("Cần model phát hiện biển số cho ảnh toàn cảnh.")
        return cached


def find_plate_boxes(model, image, imgsz, confidence):
    """Return every plate box on a whole image (used only by the still-image page)."""
    if model is None or not image.size:
        return []
    result = timed_stage("plate_detect", model.predict, image, conf=confidence, imgsz=imgsz, verbose=False)[0]
    return [([float(value) for value in box.xyxy[0].tolist()], float(box.conf.item()))
            for box in (getattr(result, "boxes", None) or [])]


def _ocr_plate(recognizer, plate_image):
    """Read a plate with a few safe crop variants; best accepted text wins."""
    if recognizer is None or not plate_image.size:
        return {"text": "", "score": 0.0, "accepted": False}
    original = timed_stage("ocr", inspect_plates, recognizer, [plate_image])[0]
    if original["accepted"] and original["score"] >= 0.92 and _standard_plate(original["text"]):
        return {**original, "variant": "original"}
    variants = []
    for padding in (4, 16):
        variants.append((f"padding_{padding}", cv2.copyMakeBorder(
            plate_image, padding, padding, padding, padding, cv2.BORDER_REPLICATE,
        )))
    variants.append(("upscale", cv2.resize(
        plate_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC,
    )))
    gray = (cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            if plate_image.ndim == 3 else plate_image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    variants.append(("threshold", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))
    orientations = [("original", plate_image), ("rotate_180", cv2.rotate(plate_image, cv2.ROTATE_180))]
    if plate_image.shape[1] / plate_image.shape[0] <= 1.8:
        orientations.extend([
            ("rotate_90", cv2.rotate(plate_image, cv2.ROTATE_90_CLOCKWISE)),
            ("rotate_270", cv2.rotate(plate_image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ])
    variants.extend(orientations[1:])
    for name, image in orientations:
        height, width = image.shape[:2]
        if height >= 12 and 0.85 <= width / height <= 1.8:
            top = image[:(height + 1) // 2]
            bottom = image[height // 2:]
            gap = np.full((len(top), max(2, width // 16), 3), 255, dtype=image.dtype)
            variants.append((f"{name}_two_lines", np.concatenate((top, gap, bottom), axis=1)))
    reads = timed_stage("ocr_retry", inspect_plates, recognizer, [image for _, image in variants])
    results = [{**original, "variant": "original"}]
    results.extend({**read, "variant": name} for (name, _), read in zip(variants, reads))
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

    plausible = [
        item for item in accepted
        if _standard_plate(item["text"]) and float(item["score"]) >= best_score - 0.10
    ]
    best_text = max(dict.fromkeys(item["text"] for item in (plausible or accepted)), key=rank)
    return max((item for item in accepted if item["text"] == best_text),
               key=lambda item: float(item["score"]))


def _standard_plate(text):
    return re.fullmatch(r"\d{2}[A-Z]{1,2}\d{4,6}", text) is not None


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


def _full_image_plates(plate_model, image, options):
    """Multi-scale whole-image plate scan; the video pipeline keeps its own logic."""
    found = []
    minimum_confidence = float(options["plate_conf"])

    def scan(candidate, restore):
        for box, confidence in find_plate_boxes(
            plate_model, candidate, int(options["plate_imgsz"]), minimum_confidence,
        ):
            mapped = _clamp_bbox(restore(box), image.shape)
            long_side, short_side = sorted((mapped[2] - mapped[0], mapped[3] - mapped[1]), reverse=True)
            if long_side >= 16 and short_side >= 8:
                found.append((mapped, confidence))

    for scale in (1.0, 0.75, 0.55):
        resized = (cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                   if scale != 1.0 else image)
        scan(resized, lambda box: [value / scale for value in box])
        if found:
            break
    height, width = image.shape[:2]
    if not found and height > width:
        scan(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
             lambda box: (box[1], height - box[2], box[3], height - box[0]))
        if not found:
            scan(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
                 lambda box: (width - box[3], box[0], width - box[1], box[2]))
    found.sort(key=lambda item: item[1], reverse=True)
    unique = []
    for box, confidence in found:
        if any(_iou(box, kept[0]) > 0.6 for kept in unique):
            continue
        unique.append((box, confidence))
    return unique


def _label(image, text, left, top, color):
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.7, 2)
    left = max(2, min(int(left), image.shape[1] - text_width - 12))
    top = max(text_height + 8, min(int(top), image.shape[0] - baseline - 4))
    right, bottom = left + text_width + 10, top + 6
    _panel(image, left - 4, top - text_height - 4, right + 2, bottom, 0.68)
    cv2.putText(image, text, (left + 3, top - 2), font, 0.7, (240, 240, 240), 2, cv2.LINE_AA)


def analyze_image(image, models, options, *, preview="full"):
    """Locate plates directly in the scene and read each distinct plate."""
    started = perf_counter()
    config = models["config"]
    height, width = image.shape[:2]
    overlay = image.copy() if preview == "full" else None
    vehicles = []
    for box, confidence in _full_image_plates(models["plate"], image, options):
        plate_image = crop(image, box)
        if not plate_image.size:
            continue
        read = _ocr_plate(models["recognizer"], plate_image)
        item = _crop_result(plate_image, read, preview)["vehicles"][0]
        item.update(
            index=len(vehicles) + 1, plate_bbox=list(box), plate_detected=True,
            plate_confidence=round(confidence, 4),
            status="Đọc được biển" if read["accepted"] else "Đã phát hiện biển, OCR chưa đọc được",
            plate_quality=plate_quality(
                plate_image, confidence, box, image.shape, target_width=config.quality_target_width,
                target_height=config.quality_target_height, target_sharpness=config.quality_target_sharpness,
                edge_margin=config.quality_edge_margin, expected_ratios=tuple(config.quality_expected_ratios),
            ),
        )
        vehicles.append(item)
        if overlay is not None:
            color = COLORS[(len(vehicles) - 1) % len(COLORS)]
            _crisp_box(overlay, box, color, 2)
            _label(overlay, _ascii(read["text"] or "OCR --"), box[0], box[1] - 8, color)

    annotated = ""
    if overlay is not None:
        ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            annotated = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "width": width, "height": height, "vehicles": vehicles, "annotated": annotated,
        "summary": {
            "vehicles": 0,
            "plates_detected": sum(1 for item in vehicles if item["plate_detected"]),
            "plates_accepted": sum(1 for item in vehicles if item["accepted"]),
        },
        "elapsed_ms": round((perf_counter() - started) * 1000, 1),
    }


def _crop_result(image, read, preview):
    height, width = image.shape[:2]
    encoded = encode_crop(image) if preview != "none" else None
    data = encoded["data"] if encoded else ""
    item = {
        "index": 1, "vehicle_type": "", "vehicle_confidence": 0, "brand": "", "brand_confidence": 0,
        "bbox": None, "plate_bbox": [0, 0, width, height], "plate_detected": False,
        "plate_confidence": None, "plate_text": read["text"], "plate_score": read["score"],
        "accepted": read["accepted"], "ocr_variant": read.get("variant", "original"),
        "plate_image": "data:image/jpeg;base64," + data if data else "",
        "status": "OCR trên ảnh cắt" if read["accepted"] else "OCR ảnh cắt chưa đạt ngưỡng",
    }
    return {"width": width, "height": height, "input_route": "crop", "vehicles": [item],
            "annotated": "", "summary": {"vehicles": 0, "plates_detected": 0,
                                         "plates_accepted": int(read["accepted"])}}


def recognize_image(payload, options=None, *, preview="full"):
    """Public entry point used by the image-only HTTP endpoint."""
    started = perf_counter()
    normalized = normalize_options(options or {})
    image = timed_stage("decode", decode_image, payload)
    decoded = perf_counter()
    with _inference_lock:
        kind = normalized["input_kind"]
        height, width = image.shape[:2]
        ratio = max(height, width) / min(height, width)
        candidate = kind == "crop" or (kind == "auto" and (max(height, width) <= 400 or ratio >= 2.2))
        models = get_models(normalized, ocr_only=candidate)
        analysis_started = perf_counter()
        result = None
        if candidate and models["recognizer"] is not None:
            read = _ocr_plate(models["recognizer"], image)
            if kind == "crop" or (read["accepted"] and read["score"] >= 0.92 and _standard_plate(read["text"])):
                result = _crop_result(image, read, preview)
        elif kind == "crop":
            raise ValueError("Cần bật model OCR để đọc ảnh đã cắt.")
        if result is None:
            if candidate:
                models = get_models(normalized)
            result = analyze_image(image, models, normalized, preview=preview)
            result["input_route"] = "scene"
            if kind == "auto" and not candidate and not result["vehicles"] and models["recognizer"] is not None:
                read = _ocr_plate(models["recognizer"], image)
                if read["accepted"] and read["score"] >= 0.92 and _standard_plate(read["text"]):
                    result = _crop_result(image, read, preview)
    ended = perf_counter()
    result["timings_ms"] = {
        "decode": round((decoded - started) * 1000, 2),
        "setup_and_wait": round((analysis_started - decoded) * 1000, 2),
        "analysis": round((ended - analysis_started) * 1000, 2),
    }
    result["elapsed_ms"] = round((ended - started) * 1000, 2)
    return result
