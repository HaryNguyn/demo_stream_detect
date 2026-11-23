"""Model adapters and per-vehicle recognition."""
import base64
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .ocr import inspect_plate
from .roi import center_in_roi, draw_roi
from .quality import plate_quality


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


def best_box(model, image, threshold, **options):
    boxes = model.predict(image, conf=threshold, verbose=False, **options)[0].boxes
    return max(boxes, key=lambda box: float(box.conf.item()), default=None)


def locate_plate(model, image, previous=None, *, imgsz=320, lower_fraction=1.0,
                 padding=0.3, full_scan=False):
    height, width = image.shape[:2]
    regions = []
    if not full_scan:
        if previous:
            x1, y1, x2, y2 = previous
            dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
            regions.append((max(0, int((x1 - dx) * width)), max(0, int((y1 - dy) * height)),
                            min(width, int(np.ceil((x2 + dx) * width))),
                            min(height, int(np.ceil((y2 + dy) * height)))))
        if lower_fraction < 1:
            regions.append((0, int(height * (1 - lower_fraction)), width, height))
    regions.append((0, 0, width, height))
    tried = set()
    for region in regions:
        if region in tried:
            continue
        tried.add(region)
        x1, y1, x2, y2 = region
        if x2 <= x1 or y2 <= y1:
            continue
        box = best_box(model, image[y1:y2, x1:x2], .35, imgsz=imgsz)
        if box is not None:
            px1, py1, px2, py2 = box.xyxy[0].tolist()
            if region != (0, 0, width, height) and min(
                px1, py1, x2 - x1 - px2, y2 - y1 - py2,
            ) < 2:
                continue  # Re-search wider instead of accepting a plate clipped by the search ROI.
            return (px1 + x1, py1 + y1, px2 + x1, py2 + y1), float(box.conf.item())
    return None


def apply_plate_read(vehicle, diagnostic, ocr):
    vehicle.plate_ocr_score = float(ocr["score"])
    diagnostic.update(text=ocr["text"], score=ocr["score"],
                      status="Đọc được biển" if ocr["accepted"] else "Có biển nhưng OCR chưa đạt ngưỡng")
    if ocr["accepted"]:
        vehicle.add_plate(ocr["text"], ocr["score"])
    else:
        vehicle.reject("ocr_rejected", f"{ocr['text']} ({float(ocr['score']):.2f})")
    sample = diagnostic.get("_plate_sample")
    if sample is not None:
        sample["recognized_plate"] = ocr["text"] if ocr["accepted"] else ""
        sample["ocr_text"] = ocr["text"]
        sample["ocr_score"] = float(ocr["score"])
        sample["quality_score"] = round(sample.get("visual_quality_score", sample["quality_score"])
                                        * float(ocr["score"]), 6)
        sample["_score"] = sample["quality_score"]
        vehicle.add_sample("plate", sample)


def add_vehicle_sample(vehicle, image, confidence, area, frame_area, frame_idx):
    score = image_quality_score(image, confidence, area, frame_area)
    if not vehicle.keeps_sample("detected", score):
        return None
    sample = encode_crop(image)
    if sample is not None:
        sample.update(_frame_idx=frame_idx, _score=round(score, 6), frame_idx=frame_idx,
                      quality_score=round(score, 6), detection_score=round(confidence, 6))
        vehicle.add_sample("detected", sample)
    return sample


def recognize_vehicle(
    vehicle, image, logo_model, plate_model, recognizer,
    frame_idx=0, vehicle_conf=0.0, bbox_area=0.0, frame_area=1.0,
    min_plate_width=0, min_plate_sharpness=0.0,
    plate_imgsz=320, plate_lower_fraction=1.0, plate_reuse_padding=0.3,
    plate_full_scan_interval=5, ocr_top_k=4, pending_ocr=None,
    quality_options=None,
    skip_plate=False,
):
    diagnostic = {"status": "Không có ảnh xe", "image": None, "text": "", "score": 0.0}
    previous = vehicle.plate_box
    if not skip_plate:
        vehicle.plate_box = None
        vehicle.plate_detection_score = 0.0
        vehicle.plate_ocr_score = 0.0
    if not image.size:
        return diagnostic
    detected = add_vehicle_sample(vehicle, image, vehicle_conf, bbox_area, frame_area, frame_idx)
    diagnostic.update(status="Chưa bật model biển/OCR", image=image)
    if logo_model is not None and not vehicle.classification_status.get("brand", {}).get("confirmed"):
        logo = best_box(logo_model, image, 0.4)
        if logo is not None:
            vehicle.add_brand(logo_model.names[int(logo.cls.item())], float(logo.conf.item()),
                              detected["quality_score"] if detected else 0.0)
    if skip_plate:
        diagnostic.update(status="Biển đã chốt, đang hoàn thiện hãng", text=vehicle.confirmed_plate)
        return diagnostic
    if plate_model is None:
        vehicle.reject("plate_model_disabled")
        return diagnostic
    vehicle.plate_searches += 1
    plate = locate_plate(
        plate_model, image, previous, imgsz=plate_imgsz, lower_fraction=plate_lower_fraction,
        padding=plate_reuse_padding, full_scan=vehicle.plate_searches % plate_full_scan_interval == 0,
    )
    if plate is None:
        vehicle.reject("no_plate_detected")
        diagnostic["status"] = "Không phát hiện biển trong ảnh xe"
        return diagnostic
    height, width = image.shape[:2]
    (px1, py1, px2, py2), plate_conf = plate
    vehicle.plate_box = (
        max(0.0, min(1.0, px1 / width)), max(0.0, min(1.0, py1 / height)),
        max(0.0, min(1.0, px2 / width)), max(0.0, min(1.0, py2 / height)),
    )
    vehicle.plate_detection_score = plate_conf
    plate_image = crop(image, (px1, py1, px2, py2))
    if plate_image.size:
        assessment = plate_quality(
            plate_image, plate_conf, (px1, py1, px2, py2), image.shape, **(quality_options or {}),
        )
        plate_sample = None
        score = assessment["quality_score"]
        if vehicle.keeps_sample("plate", score):
            plate_sample = encode_crop(plate_image)
        if plate_sample is not None:
            plate_sample["_frame_idx"] = frame_idx
            plate_sample["_score"] = round(score, 6)
            plate_sample["frame_idx"] = frame_idx
            plate_sample["quality_score"] = round(score, 6)
            plate_sample["detection_score"] = round(plate_conf, 6)
            plate_sample.update(assessment)
            diagnostic["_plate_sample"] = plate_sample
        if recognizer is None:
            if plate_sample:
                vehicle.add_sample("plate", plate_sample)
            vehicle.reject("ocr_disabled")
            diagnostic.update(image=plate_image, status="Có biển nhưng chưa bật OCR")
            return diagnostic
        sharpness = assessment["quality_factors"]["sharpness"]
        too_small = plate_image.shape[1] < min_plate_width
        too_blurry = sharpness < min_plate_sharpness
        if too_small:
            vehicle.reject("plate_too_small", f"{plate_image.shape[1]}px < {min_plate_width}px")
        if too_blurry:
            vehicle.reject("plate_too_blurry", f"{sharpness:.1f} < {min_plate_sharpness}")
        if too_small or too_blurry:
            if plate_sample:
                vehicle.add_sample("plate", plate_sample)
            diagnostic.update(image=plate_image, status="Biển quá nhỏ hoặc mờ, chờ mẫu tốt hơn")
            return diagnostic
        quality = assessment["quality_score"]
        if len(vehicle.ocr_qualities) >= ocr_top_k and quality <= min(vehicle.ocr_qualities):
            if plate_sample:
                vehicle.add_sample("plate", plate_sample)
            vehicle.reject("ocr_budget_skipped", f"{quality:.3f} <= {min(vehicle.ocr_qualities):.3f}")
            diagnostic.update(image=plate_image, status="Chờ ảnh biển tốt hơn các mẫu đã OCR")
            return diagnostic
        vehicle.ocr_qualities.append(quality)
        vehicle.ocr_qualities = sorted(vehicle.ocr_qualities, reverse=True)[:ocr_top_k]
        diagnostic.update(image=plate_image, status="Chờ OCR batch")
        if pending_ocr is not None:
            pending_ocr.append((vehicle, plate_image, diagnostic))
        else:
            apply_plate_read(vehicle, diagnostic, inspect_plate(recognizer, plate_image))
    else:
        vehicle.reject("empty_plate_crop")
        diagnostic["status"] = "Vùng biển rỗng"
    return diagnostic


def _ascii(text):
    """Hershey fonts only draw ASCII, so drop accents before putText."""
    text = str(text).replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.encode("ascii", "ignore").decode() or "?"


def _accent(track_id):
    return [
        (255, 190, 60), (90, 220, 90), (70, 170, 255), (255, 120, 240),
        (80, 225, 250), (255, 190, 120), (150, 250, 190), (215, 165, 255),
    ][track_id % 8]


def _panel(image, left, top, right, bottom, alpha=0.62):
    left, top = max(0, left), max(0, top)
    region = image[top:max(0, bottom), left:max(0, right)]
    if region.size:
        shade = np.empty_like(region)
        shade[:] = (18, 18, 18)
        cv2.addWeighted(shade, alpha, region, 1.0 - alpha, 0, region)


def _crisp_box(image, box, color, thickness):
    """Thin outline with brighter corner ticks so edges stay readable when scaled."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    tick = max(6, min(x2 - x1, y2 - y1) // 6)
    weight = thickness + 1
    for start, end in (((x1, y1), (x1 + tick, y1)), ((x1, y1), (x1, y1 + tick)),
                       ((x2, y1), (x2 - tick, y1)), ((x2, y1), (x2, y1 + tick)),
                       ((x1, y2), (x1 + tick, y2)), ((x1, y2), (x1, y2 - tick)),
                       ((x2, y2), (x2 - tick, y2)), ((x2, y2), (x2, y2 - tick))):
        cv2.line(image, start, end, color, weight, cv2.LINE_AA)


def _text_style(width, height):
    scale = float(np.clip(min(width / 1500.0, height / 750.0), 0.5, 1.05))
    return scale, (2 if scale >= 0.85 else 1), max(3, round(6 * scale)), max(4, round(6 * scale))


def draw_anpr_preview(frame, result, tracks, roi=None, *, inference_fps=0.0):
    """Compact plate-level overlays on the current video frame."""
    image = draw_roi(frame.copy(), roi)
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, weight, _, pad = _text_style(width, height)

    def caption(text, x, y, color):
        (tw, th), baseline = cv2.getTextSize(text, font, scale, weight)
        x = max(0, min(int(x), width - tw - 2 * pad))
        y = max(th + pad, min(int(y), height - baseline - pad))
        left, top = x, y - th - pad
        right, bottom = x + tw + 2 * pad, y + baseline + pad - 2
        _panel(image, left, top, right, bottom, 0.68)
        cv2.rectangle(image, (left, top), (right - 1, bottom - 1), color, 1, cv2.LINE_AA)
        cv2.putText(image, text, (x + pad, y), font, scale, (240, 240, 240), weight, cv2.LINE_AA)

    caption(f"Vehicle AI | ANPR | AI FPS: {inference_fps:.1f} | {width}x{height}",
            6, 20, (36, 88, 26))
    if result is None or result.boxes is None or result.boxes.id is None:
        return image
    for box, track_id in zip(result.boxes, result.boxes.id.int().tolist()):
        vehicle = tracks.get(track_id)
        if vehicle is None or vehicle.plate_box is None:
            continue
        coords = box.xyxy[0].tolist()
        if not center_in_roi(coords, frame.shape, roi):
            continue
        x1, y1, x2, y2 = coords
        # recognize_vehicle crops to frame boundaries before locating the plate.
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))
        px1, py1, px2, py2 = vehicle.plate_box
        left, top = int(x1 + px1 * (x2 - x1)), int(y1 + py1 * (y2 - y1))
        right, bottom = int(x1 + px2 * (x2 - x1)), int(y1 + py2 * (y2 - y1))
        if right <= left or bottom <= top:
            continue
        text = max(vehicle.plates, key=vehicle.plate_rank) if vehicle.plates else "Dang doc"
        color = (40, 130, 35) if vehicle.confirmed_plate else (30, 100, 155)
        if vehicle.plates and not vehicle.confirmed_plate:
            text += " [LOW]"
        _crisp_box(image, (left, top, right, bottom), color, weight)
        caption(f"{text} D{vehicle.plate_detection_score * 100:.0f}"
                f"/R{vehicle.plate_ocr_score * 100:.0f} #{track_id}",
                left, top - 7, color)
    return image


def draw_recognition(result, tracks, roi=None, *, frame=None):
    image = draw_roi((result.orig_img if frame is None else frame).copy(), roi)
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return image
    mask = [center_in_roi(box.xyxy[0].tolist(), result.orig_img.shape, roi)
            for box in boxes]
    result = result[mask]
    boxes = result.boxes
    height, width = image.shape[:2]
    scale, weight, gap, pad = _text_style(width, height)
    plate_font, info_font = cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX
    accent_width = max(2, round(3 * scale))
    entries = []
    for box, track_id in sorted(zip(boxes, boxes.id.int().tolist()), key=lambda item: item[1]):
        color = _accent(track_id)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        _crisp_box(image, (x1, y1, x2, y2), color, weight)
        vehicle = tracks.get(track_id)
        plate = max(vehicle.plates, key=vehicle.plate_rank) if vehicle and vehicle.plates else ""
        plate_text = _ascii(plate or "OCR --")
        class_name = _ascii(result.names[int(box.cls.item())])
        brand = _ascii(vehicle.brand) if vehicle and vehicle.brand else ""
        info_text = " | ".join(part for part in (brand, class_name, f"#{track_id}") if part)
        # Fit the chip inside narrow frames instead of clipping the text.
        widest = max(cv2.getTextSize(plate_text, plate_font, 1.0, weight)[0][0],
                     cv2.getTextSize(info_text, info_font, 1.0, 1)[0][0])
        fit = min(1.0, max(0.25, (width - 2 * pad - accent_width) / max(1, widest)))
        entry_scale, entry_info_scale = scale * fit, scale * fit * 0.82
        (plate_w, plate_h), _ = cv2.getTextSize(plate_text, plate_font, entry_scale, weight)
        (info_w, info_h), info_base = cv2.getTextSize(
            info_text, info_font, entry_info_scale, 1)
        entries.append({
            "color": color, "box": (x1, y1, x2, y2), "plate": plate_text,
            "info": info_text, "plate_h": plate_h, "info_h": info_h,
            "info_base": info_base, "confirmed": bool(vehicle and vehicle.confirmed_plate),
            "scale": entry_scale, "info_scale": entry_info_scale,
            "w": max(plate_w, info_w) + 2 * pad + accent_width,
            "h": plate_h + info_h + info_base + 3 * gap,
        })

    # Chips reserve their full area, so information never overlaps in dense scenes.
    placed, drawings = [], []
    for entry in entries:
        x1, y1, x2, y2 = entry["box"]
        w, h = min(entry["w"], width), min(entry["h"], height)
        x = max(0, min(x1, width - w))
        candidates = [y1 - h - gap, y2 + gap, y1 + gap]
        candidates.extend(sorted(range(0, max(1, height - h + 1), max(8, h // 2)),
                                 key=lambda y: (abs(y - y1), y)))
        chosen = None
        for y in candidates:
            if not 0 <= y <= height - h:
                continue
            rect = (x, int(y), x + w, int(y) + h)
            if all(rect[2] + gap <= a or c + gap <= rect[0] or
                   rect[3] + gap <= b or d + gap <= rect[1] for a, b, c, d in placed):
                chosen = rect
                break
        if chosen is None:
            y = max(0, min(y1, height - h))
            chosen = (x, y, x + w, y + h)
        placed.append(chosen)
        drawings.append((entry, chosen))

    for entry, (left, top, right, bottom) in drawings:
        _, y1, _, y2 = entry["box"]
        if bottom <= y1 or top >= y2:
            cv2.line(image, (max(0, min(entry["box"][0], width - 1)),
                             max(0, min(y1, height - 1))),
                     (left, top + entry["h"] // 2), entry["color"], 1, cv2.LINE_AA)
    for entry, (left, top, right, bottom) in drawings:
        _panel(image, left, top, right, bottom, 0.66)
        cv2.rectangle(image, (left, top), (right - 1, bottom - 1), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.rectangle(image, (left, top), (left + accent_width - 1, bottom - 1),
                      entry["color"], -1)
        text_x = left + accent_width + pad
        plate_y = top + gap + entry["plate_h"]
        plate_color = ((120, 255, 130) if entry["confirmed"] else
                       (120, 220, 255) if entry["plate"] != "OCR --" else (190, 190, 190))
        cv2.putText(image, entry["plate"], (text_x, plate_y), plate_font,
                    entry["scale"], plate_color, weight, cv2.LINE_AA)
        info_y = plate_y + entry["info_h"] + entry["info_base"] + gap
        cv2.putText(image, entry["info"], (text_x, info_y), info_font,
                    entry["info_scale"], (228, 228, 228), 1, cv2.LINE_AA)
    return image


def optional_yolo(path):
    path = path.strip()
    if not path:
        return None
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return YOLO(path)
