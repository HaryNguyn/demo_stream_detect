"""Shared helpers for repeatable image-batch and resolution-matrix measurements."""
import json
import math
import os
from pathlib import Path
import re
import statistics

from .evaluation import edit_distance, normalize_label, normalize_plate

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
RESOLUTION_SIZES = {
    "144p": (256, 144), "240p": (426, 240), "360p": (640, 360),
    "480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080),
}
_SCALED_FIELDS = (
    "min_vehicle_width", "min_plate_width", "quality_target_width", "quality_target_height",
)
_DEFAULT_BASE = {
    "min_vehicle_width": 120, "min_plate_width": 32,
    "quality_target_width": 100, "quality_target_height": 32,
}
_MIN_VALUES = {
    "min_vehicle_width": 8, "min_plate_width": 8,
    "quality_target_width": 20, "quality_target_height": 8,
}
_PATH_FIELDS = (
    "vehicle_weights", "logo_weights", "plate_weights", "ocr_weights", "ocr_config",
)


def read_jsonl(path):
    """Read a JSONL file, tolerating a UTF-8 BOM and blank lines."""
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def resolution_overrides(base, height, reference_height=720):
    """Scale pixel thresholds from a reference height down to ``height``.

    Detector input size is intentionally untouched: it is a model input, not a
    source-pixel threshold. Generation metadata records every changed field.
    """
    height, reference = int(height), float(reference_height)
    if height < 1 or reference <= 0:
        raise ValueError("Chiều cao và reference_height phải dương.")
    scale = height / reference
    overrides = {}
    for name in _SCALED_FIELDS:
        value = float(base.get(name, _DEFAULT_BASE[name]))
        overrides[name] = max(_MIN_VALUES[name], int(round(value * scale)))
    return overrides


def rebase_config_paths(data, source_dir, target_dir):
    """Rewrite relative model paths so a config can live in another folder."""
    source_dir, target_dir = Path(source_dir).resolve(), Path(target_dir).resolve()
    for name in _PATH_FIELDS:
        value = data.get(name)
        if not value or Path(value).is_absolute():
            continue
        resolved = (source_dir / value).resolve()
        data[name] = Path(os.path.relpath(resolved, target_dir)).as_posix()
    return data


def parse_video_name(path):
    """Read duration and resolution from names like ``10min_144p.mp4``."""
    stem = Path(path).stem
    duration = re.search(r"(?<!\d)(\d+)\s*min(?:ute)?s?(?!\d)", stem, re.IGNORECASE)
    resolution = re.search(r"(?<!\d)(\d{3,4})\s*p(?!\d)", stem, re.IGNORECASE)
    if not duration or not resolution:
        return None
    return {
        "source_id": stem,
        "duration_min": int(duration.group(1)),
        "resolution": f"{int(resolution.group(1))}p",
    }


def compact_result(result):
    """Drop base64 payloads before writing thousands of per-image results."""
    vehicles = []
    for item in result.get("vehicles") or []:
        vehicles.append({
            "index": item.get("index"),
            "vehicle_type": item.get("vehicle_type", ""),
            "vehicle_confidence": item.get("vehicle_confidence", 0),
            "brand": item.get("brand", ""),
            "brand_confidence": item.get("brand_confidence", 0),
            "plate_detected": bool(item.get("plate_detected")),
            "plate_text": item.get("plate_text", ""),
            "plate_score": item.get("plate_score", 0),
            "plate_confidence": item.get("plate_confidence", 0),
            "ocr_variant": item.get("ocr_variant", ""),
            "accepted": bool(item.get("accepted")),
            "bbox": item.get("bbox"),
        })
    return {
        "width": result.get("width", 0), "height": result.get("height", 0),
        "input_route": result.get("input_route", "scene"),
        "vehicles": vehicles, "summary": result.get("summary", {}),
        "elapsed_ms": result.get("elapsed_ms", 0),
        "timings_ms": result.get("timings_ms", {}),
    }


def _pick_vehicle(row):
    vehicles = row.get("vehicles") or []
    accepted = [item for item in vehicles if item.get("accepted") and item.get("plate_text")]
    if accepted:
        return max(accepted, key=lambda item: float(item.get("plate_score") or 0))
    return vehicles[0] if vehicles else None


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))]


def _accuracy_bucket():
    return {"images": 0, "labels": 0, "correct": 0}


def _count_labels(bucket, expected, predicted):
    expected, predicted = normalize_label(expected), normalize_label(predicted)
    if not expected:
        return
    bucket["images"] += 1
    if predicted:
        bucket["labels"] += 1
        bucket["correct"] += int(expected == predicted)


def summarize_images(rows, truth=None, warmup=0):
    """Aggregate compact per-image rows, optionally scored against image truth.

    Warmup only affects latency statistics; accuracy always uses every row.
    ``elapsed_total_s`` on each row gives a real cumulative wall time.
    """
    truth = truth or {}
    warmup = min(len(rows), max(0, int(warmup)))
    measured = len(rows) - warmup
    timings = [float(row["elapsed_ms"]) for row in rows[warmup:]
               if not row.get("error") and row.get("elapsed_ms") is not None]
    if rows and all("elapsed_total_s" in row for row in rows):
        start = float(rows[warmup - 1]["elapsed_total_s"]) if warmup else 0.0
        wall = max(float(rows[-1]["elapsed_total_s"]) - start, 0.0)
    else:
        wall = sum(timings) / 1000
    summary = {
        "images": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "vehicles": sum(len(row.get("vehicles") or []) for row in rows if row.get("input_route") != "crop"),
        "plates_detected": sum(1 for row in rows for item in row.get("vehicles") or []
                               if item.get("plate_detected")),
        "plates_accepted": sum(1 for row in rows for item in row.get("vehicles") or []
                               if item.get("accepted")),
        "wall_seconds": round(wall, 3),
        "images_per_second": round(measured / wall, 3) if wall else None,
        "measured_images": measured,
        "mean_ms": round(statistics.mean(timings), 2) if timings else None,
        "wall_ms_per_image": round(wall * 1000 / measured, 2) if wall and measured else None,
        "median_ms": round(statistics.median(timings), 2) if timings else None,
        "p95_ms": round(_percentile(timings, 0.95), 2) if timings else None,
        "max_ms": round(max(timings), 2) if timings else None,
    }
    groups = {"plate": _accuracy_bucket(), "vehicle_type": _accuracy_bucket(),
              "brand": _accuracy_bucket()}
    tag_groups = {}

    def score(bucket_group, expected_row, vehicle):
        target = normalize_plate(expected_row.get("plate"))
        if target:
            bucket_group["plate"]["images"] += 1
            text = normalize_plate((vehicle or {}).get("plate_text"))
            if vehicle and text:
                bucket_group["plate"]["labels"] += 1
                bucket_group["plate"]["correct"] += int(target == text)
                edits_bucket = bucket_group.setdefault("_edits", [0, 0])
                edits_bucket[0] += edit_distance(target, text)
                edits_bucket[1] += len(target)
        _count_labels(bucket_group["vehicle_type"], expected_row.get("vehicle_type"),
                      (vehicle or {}).get("vehicle_type"))
        _count_labels(bucket_group["brand"], expected_row.get("brand"),
                      (vehicle or {}).get("brand"))

    for row in rows:
        name = row.get("filename")
        expected_row = truth.get(name) or truth.get(Path(str(name)).name)
        if not expected_row:
            continue
        vehicle = _pick_vehicle(row)
        score(groups, expected_row, vehicle)
        for tag in expected_row.get("tags") or []:
            tag = normalize_label(tag)
            if not tag:
                continue
            group = tag_groups.setdefault(tag, {
                **{name: _accuracy_bucket() for name in groups},
                "_edits": [0, 0],
            })
            score(group, expected_row, vehicle)

    def finish(group):
        plate = group["plate"]
        edits_count, chars_count = group.get("_edits", [0, 0])
        group["plate"]["exact_over_total"] = (
            f"{plate['correct']}/{plate['images']}" if plate["images"] else None)
        group["plate"]["exact_on_read"] = (
            plate["correct"] / plate["labels"] if plate["labels"] else None)
        group["plate"]["cer_on_read"] = (
            edits_count / chars_count if chars_count else None)
        for name in ("vehicle_type", "brand"):
            item = group[name]
            item["accuracy_on_matched"] = item["correct"] / item["labels"] if item["labels"] else None
        group.pop("_edits", None)
        return group

    result = {**summary, "truth_images": 0}
    if truth:
        result["truth_images"] = sum(
            1 for row in rows
            if row.get("filename") in truth or Path(str(row.get("filename"))).name in truth)
        result["groups"] = finish(groups)
        result["tag_groups"] = {name: finish(group) for name, group in tag_groups.items()}
    return result
