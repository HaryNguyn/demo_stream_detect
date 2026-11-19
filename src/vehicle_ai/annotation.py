"""Model-assisted annotations; only explicit human approvals enter the train manifest."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import cv2
import yaml

from .datasets import read_manifest, sha256

NAMES = ["Ô tô con", "Xe máy", "Xe buýt", "Xe tải"]
DISPLAY = ["car", "motorbike", "bus", "truck"]
COCO_TO_CUSTOM = {2: 0, 3: 1, 5: 2, 7: 3}
FIELDS = ["class_id", "x_center", "y_center", "width", "height"]


def validate_boxes(boxes):
    validated = []
    for index, box in enumerate(boxes, 1):
        values = [float(box[key]) for key in FIELDS]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Khung {index}: tọa độ không hợp lệ.")
        cls, x, y, w, h = values
        if not cls.is_integer() or not 0 <= cls < len(NAMES):
            raise ValueError(f"Khung {index}: class_id phải là 0–3.")
        if not (0 < w <= 1 and 0 < h <= 1 and
                -1e-6 <= x - w / 2 <= x + w / 2 <= 1 + 1e-6 and
                -1e-6 <= y - h / 2 <= y + h / 2 <= 1 + 1e-6):
            raise ValueError(f"Khung {index}: bbox rỗng hoặc vượt ảnh.")
        validated.append(dict(zip(FIELDS, [int(cls), x, y, w, h])))
    return validated


def label_text(boxes):
    return "".join(f"{b['class_id']} {b['x_center']:.8f} {b['y_center']:.8f} "
                   f"{b['width']:.8f} {b['height']:.8f}\n" for b in validate_boxes(boxes))


def draw_boxes(image, boxes):
    image = image.copy()
    h, w = image.shape[:2]
    for index, box in enumerate(validate_boxes(boxes), 1):
        x, y, bw, bh = (box[key] for key in FIELDS[1:])
        x1, y1 = round((x - bw / 2) * w), round((y - bh / 2) * h)
        x2, y2 = round((x + bw / 2) * w), round((y + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{index}: {DISPLAY[box['class_id']]}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if w > 1280:
        image = cv2.resize(image, (1280, round(h * 1280 / w)))
    return image


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def approve_record(directory, record_id, boxes, *, action="approve", confirm_empty=False):
    directory = Path(directory).resolve()
    path = directory / "review.json"
    project = json.loads(path.read_text(encoding="utf-8"))
    if project["generation_status"] != "completed":
        raise ValueError("Chờ tạo nhãn gợi ý hoàn tất.")
    record = next(item for item in project["records"] if item["id"] == record_id)
    if action == "approve":
        boxes = validate_boxes(boxes)
        if not boxes and not confirm_empty:
            raise ValueError("Cần xác nhận ảnh âm tính trước khi duyệt nhãn rỗng.")
        label = directory / "reviewed_labels" / f"{record_id}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        temporary = label.with_suffix(".tmp")
        temporary.write_text(label_text(boxes), encoding="utf-8")
        temporary.replace(label)
        record.update(status="approved", reviewed_boxes=boxes, label=str(label))
    elif action == "skip":
        record.update(status="skipped", label="")
    else:
        raise ValueError("action phải là approve hoặc skip.")
    record["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, project)
    # Always regenerate from authoritative review state. Skipped/pending never enter this manifest.
    manifest = directory / "reviewed_manifest.csv"
    tmp = manifest.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "label", "text", "group_id", "split"])
        writer.writeheader()
        for item in project["records"]:
            if item["status"] == "approved":
                writer.writerow({"image": item["image"], "label": item["label"],
                                 "text": "", "group_id": item["group_id"], "split": ""})
    tmp.replace(manifest)
    return project


def prepare(manifests, output, weights, confidence=0.25, imgsz=960):
    from ultralytics import YOLO

    if not 0 < confidence <= 1 or imgsz <= 0:
        raise ValueError("confidence phải trong (0,1], imgsz phải > 0.")
    rows = [row for path in manifests for row in read_manifest(path)]
    images = [row["image"] for row in rows]
    if len(set(images)) != len(images):
        raise ValueError("Manifest chứa đường dẫn ảnh trùng.")
    model = YOLO(weights)
    # This adapter uses the bundled COCO vehicle taxonomy, not an arbitrary custom checkpoint.
    expected = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    if any(model.names.get(key) != value for key, value in expected.items()):
        raise ValueError("Model phải dùng taxonomy COCO car/motorcycle/bus/truck.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "suggested_labels").mkdir()
    (output / "previews").mkdir()
    project = {
        "task": "vehicle_detection", "generation_status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(weights).resolve()), "model_sha256": sha256(weights),
        "confidence": confidence, "imgsz": imgsz, "names": NAMES, "records": [],
        "note": "Machine proposals only. Review all boxes and missed vehicles before approving.",
    }
    atomic_json(output / "review.json", project)
    for index, row in enumerate(rows, 1):
        image = cv2.imread(row["image"])
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {row['image']}")
        result = model.predict(image, classes=list(COCO_TO_CUSTOM), conf=confidence,
                               imgsz=imgsz, device="cpu", verbose=False)[0]
        boxes, scores = [], []
        for detected in result.boxes:
            coords = detected.xywhn[0].tolist()
            box = dict(zip(FIELDS, [COCO_TO_CUSTOM[int(detected.cls.item())], *coords]))
            boxes.append(box)
            scores.append(float(detected.conf.item()))
        boxes = validate_boxes(boxes)
        record_id = hashlib.sha256(row["image"].encode()).hexdigest()[:20]
        (output / "suggested_labels" / f"{record_id}.txt").write_text(label_text(boxes), encoding="utf-8")
        preview = output / "previews" / f"{record_id}.jpg"
        if not cv2.imwrite(str(preview), draw_boxes(image, boxes)):
            raise OSError(f"Không ghi được {preview}")
        project["records"].append({
            "id": record_id, "image": row["image"], "group_id": row["group_id"],
            "timestamp_seconds": row.get("timestamp_seconds"),
            "status": "pending", "suggested_boxes": boxes, "scores": scores, "label": "",
            "preview": str(preview),
        })
        atomic_json(output / "review.json", project)
        if index % 10 == 0 or index == len(rows):
            print(f"Prepared {index}/{len(rows)} images", flush=True)
    project["generation_status"] = "completed"
    atomic_json(output / "review.json", project)
    (output / "classes.yaml").write_text(yaml.safe_dump({"names": NAMES}, allow_unicode=True), encoding="utf-8")
    summary = {
        "images": len(rows),
        "groups": sorted({row["group_id"] for row in rows}),
        "suggested_boxes": sum(len(r["suggested_boxes"]) for r in project["records"]),
        "images_without_suggestions": sum(not r["suggested_boxes"] for r in project["records"]),
        "approved_images": 0, "output": str(output),
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="models/yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifests, args.output, args.weights, args.confidence, args.imgsz),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
