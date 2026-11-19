"""Dataset preparation with group-isolated splits and label validation."""
import csv
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import yaml

SPLITS = ("train", "val", "test")
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "group_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest cần cột image,group_id.")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest rỗng; cần ảnh đã gán nhãn.")
    for row in rows:
        for key in ("image", "label"):
            if row.get(key):
                row[key] = str((path.parent / row[key]).resolve())
        if not row["group_id"].strip():
            raise ValueError("group_id không được rỗng.")
    return rows


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_manifest(source, output, seed=42):
    rows = read_manifest(source)
    groups = sorted({row["group_id"] for row in rows})
    if len(groups) < 3:
        raise ValueError("Cần ít nhất 3 nhóm độc lập để chia train/val/test.")
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * 0.15))
    n_test = max(1, round(len(groups) * 0.15))
    assigned = {
        group: "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for i, group in enumerate(groups)
    }
    for row in rows:
        row["split"] = assigned[row["group_id"]]
    write_manifest(output, rows)
    return {split: sum(row["split"] == split for row in rows) for split in SPLITS}


def validate_manifest(path, task, names=None, require_all=True):
    rows = read_manifest(path)
    groups, hashes, paths = {}, {}, set()
    counts = dict.fromkeys(SPLITS, 0)
    for index, row in enumerate(rows, 2):
        image = Path(row["image"])
        split = row.get("split")
        if split not in SPLITS:
            raise ValueError(f"Dòng {index}: split phải là train/val/test.")
        counts[split] += 1
        group = row["group_id"]
        if group in groups and groups[group] != split:
            raise ValueError(f"Rò rỉ group_id giữa các split: {group}")
        groups[group] = split
        if image in paths:
            raise ValueError(f"Ảnh xuất hiện nhiều lần: {image}")
        paths.add(image)
        pixels = cv2.imread(str(image))
        if pixels is None:
            raise ValueError(f"Không đọc được ảnh: {image}")
        digest = hashlib.sha256(pixels.tobytes() + str(pixels.shape).encode()).hexdigest()
        if digest in hashes and hashes[digest] != split:
            raise ValueError(f"Ảnh trùng nội dung giữa các split: {image}")
        hashes[digest] = split
        if task == "ocr":
            text = row.get("text", "")
            if not text or any(char not in ALPHABET for char in text) or len(text) > 16:
                raise ValueError(f"Dòng {index}: text OCR cần 1–16 ký tự A–Z, 0–9, không dấu phân cách.")
        elif task == "detect":
            if not names:
                raise ValueError("Cần danh sách names cho detector.")
            label = Path(row.get("label") or "__missing_label__")
            if not label.is_file():
                raise ValueError(f"Dòng {index}: thiếu file nhãn YOLO. Ảnh âm tính cần file nhãn rỗng.")
            for line in label.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f"Nhãn phải là class x_center y_center width height: {label}")
                cls, x, y, w, h = map(float, values)
                if not all(math.isfinite(v) for v in (cls, x, y, w, h)):
                    raise ValueError(f"Nhãn chứa NaN/Infinity: {label}")
                if not cls.is_integer() or not 0 <= cls < len(names):
                    raise ValueError(f"Class ID không hợp lệ: {label}")
                if not (0 < w <= 1 and 0 < h <= 1 and
                        -1e-6 <= x - w / 2 <= x + w / 2 <= 1 + 1e-6 and
                        -1e-6 <= y - h / 2 <= y + h / 2 <= 1 + 1e-6):
                    raise ValueError(f"Bbox vượt biên hoặc rỗng: {label}")
        else:
            raise ValueError("task phải là detect hoặc ocr.")
    if require_all and any(count == 0 for count in counts.values()):
        raise ValueError("Cần đủ train/val/test; không dùng test để chọn checkpoint.")
    return {"images": len(rows), "groups": len(groups), "splits": counts}


def build_yolo(manifest, output, names):
    report = validate_manifest(manifest, "detect", names)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for row in read_manifest(manifest):
        image = Path(row["image"])
        name = hashlib.sha256(str(image).encode()).hexdigest()[:20]
        for kind in ("images", "labels"):
            (output / kind / row["split"]).mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, output / "images" / row["split"] / (name + image.suffix.lower()))
        shutil.copy2(row["label"], output / "labels" / row["split"] / (name + ".txt"))
    config = {"path": str(output), **{s: f"images/{s}" for s in SPLITS}, "names": names}
    (output / "dataset.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return config


def extract_frames(video, output, group_id, every_seconds=1.0, max_frames=300):
    if every_seconds <= 0 or not math.isfinite(every_seconds) or max_frames <= 0:
        raise ValueError("every_seconds và max_frames phải > 0.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video))
    rows = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Không mở được video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video không có FPS hợp lệ.")
        step = max(1, round(fps * every_seconds))
        index = 0
        while len(rows) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                name = f"frame_{index:08d}.jpg"
                if not cv2.imwrite(str(output / name), frame):
                    raise OSError(f"Không ghi được {name}")
                rows.append({"image": name, "label": "", "text": "", "group_id": group_id,
                             "split": "", "timestamp_seconds": round(index / fps, 6)})
            index += 1
    finally:
        capture.release()
    if not rows:
        raise ValueError("Video không có frame đọc được.")
    write_manifest(output / "manifest.csv", rows)
    return {"frames": len(rows), "manifest": str(output / "manifest.csv")}
