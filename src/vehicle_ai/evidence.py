"""Evidence files and metadata, independent of Streamlit and the DB backend."""
from base64 import b64decode
from datetime import datetime
import json
from pathlib import Path
import re

IMAGE_FIELDS = {"detected": "Danh sách ảnh xe", "plate": "Danh sách ảnh biển"}
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def safe_name(value) -> str:
    name = re.sub(r"[^\w.-]+", "_", str(value)).strip("._")[:100] or "UNKNOWN"
    return f"_{name}" if name.split(".")[0].upper() in RESERVED_NAMES else name


def save_event_images(row: dict, root: str | Path) -> tuple[Path, Path, dict]:
    camera = safe_name(row.get("Camera", "camera"))
    identity = safe_name(row.get("Xe ID") or row.get("Biển số") or f"TRACK-{row.get('Track ID', 0)}")
    event_dir = (
        Path(root) / datetime.now().strftime("%Y-%m-%d")
        / camera / identity / safe_name(row["event_id"])
    )
    event_dir.mkdir(parents=True, exist_ok=True)
    saved = {kind: [] for kind in IMAGE_FIELDS}
    for kind, field in IMAGE_FIELDS.items():
        for index, image in enumerate(row.get(field, []) or [], 1):
            if not image or image.get("encoding") != "base64":
                continue
            path = event_dir / f"{kind}_{index:03d}.jpg"
            path.write_bytes(b64decode(image["data"], validate=True))
            saved[kind].append({
                "path": str(path), "mime_type": image.get("mime_type", "image/jpeg"),
                "width": image.get("width"), "height": image.get("height"),
            })
    excluded = {*IMAGE_FIELDS.values(), "Ảnh xe", "Ảnh biển số"}
    metadata = {key: value for key, value in row.items() if key not in excluded}
    metadata.update({"Thư mục ảnh": str(event_dir), "images": saved})
    metadata_path = event_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return event_dir, metadata_path, saved
