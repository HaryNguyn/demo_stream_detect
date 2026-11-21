"""Evidence files and metadata, independent of Streamlit and the DB backend."""
from base64 import b64decode
from datetime import datetime
import json
from pathlib import Path
import re
from hashlib import sha256

IMAGE_FIELDS = {"detected": "Danh sách ảnh xe", "plate": "Danh sách ảnh biển"}
RECOGNITION_FIELDS = (
    "confidence_status", "plate_votes", "brand_votes", "Biển số", "Điểm biển",
    "Hãng", "Điểm hãng", "Loại xe", "Xuất hiện (giây)", "Kết thúc (giây)",
    "vote_summaries", "classification_status", "plate_decision", "source_track_ids",
    "raw_event_ids", "rejection_reasons", "rejection_detail", "finalization_reason",
)
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def safe_name(value) -> str:
    name = re.sub(r"[^\w.-]+", "_", str(value)).strip("._")[:100] or "UNKNOWN"
    return f"_{name}" if name.split(".")[0].upper() in RESERVED_NAMES else name


def save_event_images(row: dict, root: str | Path, *, previous_images=None) -> tuple[Path, Path, dict]:
    camera = safe_name(row.get("Camera", "camera"))
    identity = safe_name(row.get("Xe ID") or row.get("Biển số") or f"TRACK-{row.get('Track ID', 0)}")
    event_dir = (
        Path(root) / datetime.now().strftime("%Y-%m-%d")
        / camera / identity / safe_name(row["event_id"])
    )
    event_dir.mkdir(parents=True, exist_ok=True)
    saved = {kind: list((previous_images or {}).get(kind, [])) for kind in IMAGE_FIELDS}
    for kind, field in IMAGE_FIELDS.items():
        if field in row:
            saved[kind] = []
        for index, image in enumerate(row.get(field, []) or [], 1):
            if image and image.get("path") and image.get("encoding") != "base64":
                saved[kind].append(dict(image))
                continue
            if not image or image.get("encoding") != "base64":
                continue
            data = b64decode(image["data"], validate=True)
            digest = sha256(data).hexdigest()
            path = event_dir / f"{kind}_{digest[:20]}.jpg"
            path.write_bytes(data)
            saved[kind].append({
                "path": str(path), "mime_type": image.get("mime_type", "image/jpeg"),
                "width": image.get("width"), "height": image.get("height"),
                "quality_score": image.get("quality_score", 0),
                "detection_score": image.get("detection_score", 0),
                "ocr_score": image.get("ocr_score", 0),
                "recognized_plate": image.get("recognized_plate", ""),
                "ocr_text": image.get("ocr_text", ""),
                "quality_factors": image.get("quality_factors", {}),
                "visual_quality_score": image.get("visual_quality_score", image.get("quality_score", 0)),
                "image_id": digest,
            })
    saved["_recognition"] = {key: row[key] for key in RECOGNITION_FIELDS if key in row}
    for kind in IMAGE_FIELDS:
        retained = {item["path"] for item in saved[kind]}
        for old in (previous_images or {}).get(kind, []):
            path = Path(old["path"]).resolve()
            if (str(old["path"]) not in retained and path.is_relative_to(Path(root).resolve())
                    and path.parent.name == safe_name(row["event_id"])):
                path.unlink(missing_ok=True)
    excluded = {*IMAGE_FIELDS.values(), "Ảnh xe", "Ảnh biển số"}
    metadata = {key: value for key, value in row.items() if key not in excluded}
    metadata.update({"Thư mục ảnh": str(event_dir), "images": saved})
    metadata_path = event_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return event_dir, metadata_path, saved
