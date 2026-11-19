"""Model adapters and per-vehicle recognition."""
import base64
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .ocr import inspect_plate
from .roi import center_in_roi


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
        vehicle.add_sample("detected", detected)
    diagnostic.update(status="Chưa bật model biển/OCR", image=image)
    if logo_model is not None:
        logo = best_box(logo_model, image, 0.4)
        if logo is not None and logo.conf.item() > vehicle.brand_score:
            vehicle.brand = logo_model.names[int(logo.cls.item())]
            vehicle.brand_score = float(logo.conf.item())
    if plate_model is None:
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
            vehicle.add_sample("plate", plate_sample)
        if recognizer is None:
            diagnostic.update(image=plate_image, status="Có biển nhưng chưa bật OCR")
            return diagnostic
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


def draw_recognition(result, tracks, roi=None):
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return result.orig_img.copy()
    mask = [center_in_roi(box.xyxy[0].tolist(), result.orig_img.shape, roi)
            for box in boxes]
    result = result[mask]
    boxes = result.boxes
    image = result.plot()
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


def optional_yolo(path):
    path = path.strip()
    if not path:
        return None
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return YOLO(path)
