"""Model adapters and per-vehicle recognition."""
import base64
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .ocr import inspect_plate
from .roi import center_in_roi, draw_roi


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
    image = draw_roi(result.orig_img.copy(), roi)
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return image
    mask = [center_in_roi(box.xyxy[0].tolist(), result.orig_img.shape, roi)
            for box in boxes]
    result = result[mask]
    boxes = result.boxes
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    height, width = image.shape[:2]
    placed, captions, overflow = [], [], []
    labels = []
    for box, track_id in sorted(zip(boxes, boxes.id.int().tolist()), key=lambda item: item[1]):
        color = ((37 * track_id) % 156 + 100, (67 * track_id) % 156 + 100,
                 (97 * track_id) % 156 + 100)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        lines = []
        vehicle = tracks.get(track_id)
        if vehicle is not None:
            plate = max(vehicle.plates, key=vehicle.plate_rank) if vehicle.plates else "OCR: --"
            lines.append(plate)
        lines.append(f"{result.names[int(box.cls.item())]} #{track_id}")
        if vehicle is not None and vehicle.brand:
            lines.append(vehicle.brand)
        labels.append((lines, color, x1, y1, y2))

    # Reserve the whole multiline caption, including the black text outline.
    for lines, color, x1, y1, y2 in labels:
        scale = min(0.5, max(1, width - 10) /
                    max(1, max(cv2.getTextSize(line, font, 1.0, 3)[0][0] for line in lines)))
        sizes = [cv2.getTextSize(line, font, scale, 3) for line in lines]
        text_width = max(size[0][0] for size in sizes)
        text_height = max(size[0][1] for size in sizes)
        line_height = text_height + max(size[1] for size in sizes) + 4
        label_width, label_height = text_width + 10, line_height * len(lines) + 8
        x = max(0, min(x1, width - label_width))
        candidates = [y1 + 3, y1 - label_height - 3, y2 + 3]
        candidates.extend(sorted(range(0, height - label_height + 1, label_height + 3),
                                 key=lambda y: abs(y - y1)))
        position = None
        for y in candidates:
            if label_width > width or y < 0 or y + label_height > height:
                continue
            rect = (x, y, x + label_width, y + label_height)
            if all(rect[2] + 3 <= a or c + 3 <= rect[0] or
                   rect[3] + 3 <= b or d + 3 <= rect[1] for a, b, c, d in placed):
                position = rect
                break
        if position is None:
            overflow.append((lines, color, label_width, label_height, text_height, scale, line_height))
            continue
        placed.append(position)
        captions.append((position, lines, color, text_height, scale, line_height))
        left, top, right, bottom = position
        if top != y1 + 3:
            cv2.line(image, (max(0, min(x1, width - 1)), max(0, min(y1, height - 1))),
                     (left, top + label_height // 2), color, 1, cv2.LINE_AA)
        # Draw labels after all connectors so lines cannot cross their text.
    for (left, top, right, bottom), lines, color, text_height, scale, line_height in captions:
        for index, line in enumerate(lines):
            origin = (left + 5, top + 4 + text_height + index * line_height)
            cv2.putText(image, line, origin, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, origin, font, scale, color, 1, cv2.LINE_AA)
    # Dense scenes retain all information in separate rows below the original frame.
    if overflow:
        extra_height = sum(item[3] + 3 for item in overflow)
        extra_width = max(0, max(item[2] for item in overflow) - width)
        image = cv2.copyMakeBorder(image, 0, extra_height, 0, extra_width,
                                   cv2.BORDER_CONSTANT, value=(25, 25, 25))
        top = height
        for lines, color, _, label_height, text_height, scale, line_height in overflow:
            for index, line in enumerate(lines):
                origin = (5, top + 4 + text_height + index * line_height)
                cv2.putText(image, line, origin, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, line, origin, font, scale, color, 1, cv2.LINE_AA)
            top += label_height + 3
    return image


def optional_yolo(path):
    path = path.strip()
    if not path:
        return None
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return YOLO(path)
